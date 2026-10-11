# -*- coding:utf-8 -*-
"""
项目当前的核心跟踪器实现。

可以把这个文件理解成两层：
1. 原始 ByteTrack 主体：
   - 高分检测框第一阶段关联
   - 低分检测框第二阶段补关联
   - Kalman 预测 + 匈牙利匹配 + 轨迹状态管理
2. 本项目新增增强：
   - ReID 外观特征
   - gallery / prototype 外观记忆
   - long-lost recovery
   - temporal matcher
   - close-contact anti-swap

如果你后面要重新理解整个项目，最值得先读的就是这个文件。
"""

from collections import deque
from itertools import islice

import numpy as np
import torch

from .basetrack import BaseTrack, TrackState
from .gmc import SparseOptFlowGMC
from .kalman_filter import KalmanFilter
from .reid_extractor import get_reid_extractor
from .temporal_matcher import get_temporal_matcher
from yolox.tracker import matching


class STrack(BaseTrack):
    """
    单目标轨迹类。

    一条 STrack 对应视频中一个正在被跟踪或曾经被跟踪的目标。
    它不仅保存几何状态，还保存外观状态和时序状态：
    - 几何状态：tlwh、Kalman mean/covariance、score
    - 生命周期状态：Tracked / Lost / Removed
    - 外观状态：curr_feat、smooth_feat、gallery、prototype
    - 时序状态：temporal_features、temporal_boxes、temporal_frames

    你可以把它理解成“某个行人的长期记忆对象”。
    """

    shared_kalman = KalmanFilter()

    def __init__(
        self,
        tlwh,
        score,
        temp_feat=None,
        feat_history=50,
        gallery_size=20,
        prototype_count=4,
        gallery_recent_count=3,
        gallery_update_thresh=0.92,
        prototype_sim_thresh=0.80,
        prototype_momentum=0.8,
    ):
        # STrack 表示“一条轨迹”。
        # 除了原始 ByteTrack 的几何状态外，这里还额外维护：
        # - smooth_feat: 平滑后的外观特征
        # - gallery_features / prototype_features: 外观记忆库
        # - temporal_*: 供 Transformer 使用的时序历史
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.smooth_feat = None
        self.curr_feat = None
        self.features = deque([], maxlen=feat_history)
        self.gallery_features = deque([], maxlen=max(int(gallery_size), 0))
        self.prototype_features = []
        self.prototype_counts = []
        self.prototype_last_update = []
        self.temporal_features = deque([], maxlen=feat_history)
        self.temporal_boxes = deque([], maxlen=feat_history)
        self.temporal_frames = deque([], maxlen=feat_history)
        self.gallery_recent_count = max(int(gallery_recent_count), 0)
        self.gallery_update_thresh = float(gallery_update_thresh)
        self.prototype_count = max(int(prototype_count), 0)
        self.prototype_sim_thresh = float(prototype_sim_thresh)
        self.prototype_momentum = float(prototype_momentum)
        self.feature_update_index = 0
        self.alpha = 0.9

        if temp_feat is not None:
            self.update_features(temp_feat)

    def update_features(self, feat):
        """
        更新轨迹的外观特征记忆。

        输入 feat 是当前帧检测框通过 OSNet 提取出的行人 embedding。
        这里会同时更新三种外观表示：
        1. curr_feat：当前帧最新特征。
        2. smooth_feat：EMA 平滑特征，用于普通 ReID 匹配。
        3. gallery/prototype：多外观记忆，用于减少单一平滑特征被遮挡污染。
        """
        feat = np.asarray(feat, dtype=np.float32)
        norm = np.linalg.norm(feat)
        if norm == 0:
            return
        feat = feat / norm
        self.feature_update_index += 1
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            # 用 EMA 保持外观特征稳定，减少单帧遮挡/模糊造成的抖动。
            self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat
            self.smooth_feat /= np.linalg.norm(self.smooth_feat)
        self.features.append(feat)
        self._update_gallery(feat)
        self._update_prototypes(feat)

    def _update_gallery(self, feat):
        # gallery 更偏“保留近期但彼此不太重复的多个外观快照”。
        if self.gallery_features.maxlen == 0:
            return
        if len(self.gallery_features) == 0:
            self.gallery_features.append(feat.copy())
            return

        gallery_array = np.asarray(self.gallery_features, dtype=np.float32)
        max_similarity = float(np.max(gallery_array @ feat))
        if len(self.gallery_features) < 2 or max_similarity < self.gallery_update_thresh:
            self.gallery_features.append(feat.copy())

    def _update_prototypes(self, feat):
        # prototype 更偏“把历史外观聚成少量原型中心”，用于稳住近似外观。
        if self.prototype_count <= 0:
            return
        if len(self.prototype_features) == 0:
            self.prototype_features.append(feat.copy())
            self.prototype_counts.append(1)
            self.prototype_last_update.append(self.feature_update_index)
            return

        prototype_array = np.asarray(self.prototype_features, dtype=np.float32)
        similarities = prototype_array @ feat
        best_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_index])

        if best_similarity >= self.prototype_sim_thresh:
            updated = (
                self.prototype_momentum * self.prototype_features[best_index]
                + (1.0 - self.prototype_momentum) * feat
            )
            updated /= np.linalg.norm(updated)
            self.prototype_features[best_index] = updated.astype(np.float32)
            self.prototype_counts[best_index] += 1
            self.prototype_last_update[best_index] = self.feature_update_index
            return

        if len(self.prototype_features) < self.prototype_count:
            self.prototype_features.append(feat.copy())
            self.prototype_counts.append(1)
            self.prototype_last_update.append(self.feature_update_index)
            return

        replace_index = int(np.argmin(self.prototype_last_update))
        self.prototype_features[replace_index] = feat.copy()
        self.prototype_counts[replace_index] = 1
        self.prototype_last_update[replace_index] = self.feature_update_index

    def get_gallery_bank(self, include_recent=True, include_gallery=True, include_prototypes=True):
        """
        取出当前轨迹的外观记忆库。

        gallery bank 会把近期特征、稳定 gallery 和 prototype 拼成一个候选集合。
        后续 matching.gallery_embedding_distance 会用检测框特征和这个 bank 计算距离。
        """
        bank = []
        if include_recent and self.gallery_recent_count > 0:
            bank.extend(list(self.features)[-self.gallery_recent_count :])
        if include_gallery:
            bank.extend(list(self.gallery_features))
        if include_prototypes:
            bank.extend(list(self.prototype_features))
        if len(bank) == 0:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(bank, dtype=np.float32)

    def push_temporal_state(self, frame_id):
        """
        把当前轨迹状态压入 temporal matcher 的历史序列。

        temporal matcher 需要的不只是外观，还需要：
        - 每次观测的外观 embedding
        - 每次观测的框位置
        - 每次观测发生在哪一帧
        """
        # 只有在拿到有效外观特征后，才把这一帧压入时序历史。
        if self.curr_feat is None:
            return
        self.temporal_features.append(np.asarray(self.curr_feat, dtype=np.float32).copy())
        self.temporal_boxes.append(np.asarray(self.tlwh, dtype=np.float32).copy())
        self.temporal_frames.append(int(frame_id))

    def lost_frames(self, frame_id):
        return max(frame_id - self.end_frame, 0)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            # Lost 轨迹在预测时把速度衰减为 0，避免长时间丢失后漂得太远。
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        valid_stracks = [st for st in stracks if st.mean is not None and st.covariance is not None]
        if len(valid_stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in valid_stracks])
            multi_covariance = np.asarray([st.covariance for st in valid_stracks])
            for i, st in enumerate(valid_stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                valid_stracks[i].mean = mean
                valid_stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, warp):
        if warp is None or len(stracks) == 0:
            return

        linear = warp[:, :2]
        translation = warp[:, 2]
        for track in stracks:
            track.apply_affine(linear, translation)

    def apply_affine(self, linear, translation):
        # GMC 估计出来的是全局仿射，这里把它作用到单条轨迹的框和速度上。
        tlbr = self.tlbr
        x1, y1, x2, y2 = tlbr
        corners = np.array(
            [[x1, y1], [x2, y1], [x1, y2], [x2, y2]],
            dtype=np.float32,
        )
        warped = corners @ linear.T + translation
        min_xy = warped.min(axis=0)
        max_xy = warped.max(axis=0)
        warped_tlbr = np.r_[min_xy, max_xy]
        warped_tlwh = self.tlbr_to_tlwh(warped_tlbr)

        if self.mean is None:
            self._tlwh = warped_tlwh
            return

        xyah = self.tlwh_to_xyah(warped_tlwh)
        self.mean[:4] = xyah
        self.mean[4:6] = linear @ self.mean[4:6]

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.push_temporal_state(frame_id)

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            self.push_temporal_state(frame_id)

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            self.push_temporal_state(frame_id)

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker:
    """
    多目标跟踪器主类。

    这个类每处理一帧，都会完成从检测框到轨迹 ID 的完整关联过程。
    当前项目在原始 ByteTrack 基础上，额外支持：
    - ReID 外观融合
    - gallery/prototype 外观记忆
    - long-lost recovery
    - temporal matcher
    - close-contact anti-swap
    - association debug 日志
    """

    def __init__(self, args, frame_rate=30):
        # 这里集中读取所有运行时开关。
        # 你以后看配置文件时，可以把参数和这里一一对应起来理解。
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.feature_history = max(
            int(getattr(args, "track_feature_history", 50)),
            int(getattr(args, "temporal_history_len", 8)) * 4,
            int(getattr(args, "gallery_size", 20)),
        )
        self.with_reid = getattr(args, "with_reid", False)
        self.with_gallery = self.with_reid and getattr(args, "with_gallery", False)
        self.gallery_size = max(int(getattr(args, "gallery_size", 20)), 0)
        self.prototype_count = max(int(getattr(args, "prototype_count", 4)), 0)
        self.gallery_recent_count = max(int(getattr(args, "gallery_recent_count", 3)), 0)
        self.gallery_match_weight = float(np.clip(getattr(args, "gallery_match_weight", 0.4), 0.0, 1.0))
        self.gallery_topk = max(int(getattr(args, "gallery_topk", 3)), 1)
        self.gallery_update_thresh = float(getattr(args, "gallery_update_thresh", 0.92))
        self.prototype_sim_thresh = float(getattr(args, "prototype_sim_thresh", 0.80))
        self.prototype_momentum = float(getattr(args, "prototype_momentum", 0.8))
        self.with_long_recovery = self.with_reid and getattr(args, "with_long_recovery", False)
        self.long_track_buffer = int(
            frame_rate / 30.0 * max(getattr(args, "long_track_buffer", args.track_buffer), args.track_buffer)
        )
        self.max_time_lost = self.long_track_buffer if self.with_long_recovery else self.buffer_size
        self.recent_lost_window = min(
            int(frame_rate / 30.0 * getattr(args, "recent_lost_window", args.track_buffer)),
            self.max_time_lost,
        )
        self.kalman_filter = KalmanFilter()
        self.reid_extractor = get_reid_extractor(args, device=getattr(args, "device", "cpu"))
        self.appearance_thresh = getattr(args, "appearance_thresh", 0.25)
        self.reid_weight = getattr(args, "reid_weight", 0.5)
        self.lost_appearance_thresh = getattr(args, "lost_appearance_thresh", 0.18)
        self.long_match_thresh = getattr(args, "long_match_thresh", 0.35)
        self.long_motion_lambda = getattr(args, "long_motion_lambda", 0.98)
        self.with_temporal_matcher = self.with_reid and getattr(args, "with_temporal_matcher", False)
        self.temporal_matcher = (
            get_temporal_matcher(args, device=getattr(args, "device", "cpu"))
            if self.with_temporal_matcher
            else None
        )
        if self.reid_extractor is not None and self.temporal_matcher is not None:
            reid_feat_dim = int(getattr(self.reid_extractor, "feature_dim", 0))
            temporal_feat_dim = int(getattr(self.temporal_matcher, "feat_dim", 0))
            if reid_feat_dim > 0 and temporal_feat_dim > 0 and reid_feat_dim != temporal_feat_dim:
                raise ValueError(
                    "ReID feature dim {} does not match temporal matcher feat_dim {}. "
                    "Please disable temporal matcher or retrain it for the current ReID backbone.".format(
                        reid_feat_dim, temporal_feat_dim
                    )
                )
        self.with_temporal_ambiguity = self.with_temporal_matcher and getattr(args, "with_temporal_ambiguity", False)
        self.temporal_weight = getattr(args, "temporal_weight", 0.5)
        self.temporal_score_thresh = getattr(args, "temporal_score_thresh", 0.55)
        self.temporal_ambiguity_weight = getattr(args, "temporal_ambiguity_weight", 0.6)
        self.temporal_ambiguity_score_thresh = getattr(args, "temporal_ambiguity_score_thresh", 0.45)
        self.temporal_ambiguity_score_margin = float(getattr(args, "temporal_ambiguity_score_margin", 0.12))
        self.temporal_ambiguity_gap = getattr(args, "temporal_ambiguity_gap", 0.05)
        self.temporal_ambiguity_cost_thresh = getattr(args, "temporal_ambiguity_cost_thresh", 0.7)
        self.temporal_ambiguity_override_margin = float(getattr(args, "temporal_ambiguity_override_margin", 0.04))
        self.temporal_ambiguity_topk = max(int(getattr(args, "temporal_ambiguity_topk", 2)), 2)
        self.temporal_ambiguity_tracked_only = bool(getattr(args, "temporal_ambiguity_tracked_only", True))
        self.temporal_proximity_center_thresh = float(getattr(args, "temporal_proximity_center_thresh", 1.2))
        self.temporal_proximity_iou_thresh = float(getattr(args, "temporal_proximity_iou_thresh", 0.05))
        self.with_anti_swap = bool(getattr(args, "with_anti_swap", False))
        self.anti_swap_motion_weight = float(getattr(args, "anti_swap_motion_weight", 0.35))
        self.anti_swap_order_weight = float(getattr(args, "anti_swap_order_weight", 0.35))
        self.anti_swap_margin = float(getattr(args, "anti_swap_margin", 0.05))
        self.anti_swap_topk = max(int(getattr(args, "anti_swap_topk", 2)), 2)
        self.anti_swap_track_center_thresh = float(getattr(args, "anti_swap_track_center_thresh", 1.25))
        self.anti_swap_det_center_thresh = float(getattr(args, "anti_swap_det_center_thresh", 1.35))
        self.anti_swap_det_iou_thresh = float(getattr(args, "anti_swap_det_iou_thresh", 0.02))
        self.anti_swap_order_margin = float(getattr(args, "anti_swap_order_margin", 0.12))
        self.with_cmc = getattr(args, "with_cmc", False)
        self.cmc = SparseOptFlowGMC(downscale=getattr(args, "cmc_downscale", 2)) if self.with_cmc else None
        self.debug_association = bool(getattr(args, "debug_association", False))
        self.debug_ambiguous_only = bool(getattr(args, "debug_ambiguous_only", False))
        self.debug_topk = max(int(getattr(args, "debug_topk", 3)), 1)
        self.last_debug_frame = None

    def _empty_output(self):
        return np.empty((0, 6), dtype=np.float32)

    def _prepare_output(self, output_results):
        """
        统一整理 detector 输出。

        detector 可能返回 None、torch.Tensor、numpy.ndarray 或空数组。
        这里统一转换成形状为 N x C 的 numpy 数组，方便后续按列取 bbox 和 score。
        """
        if output_results is None:
            return self._empty_output()
        if isinstance(output_results, torch.Tensor):
            output_results = output_results.cpu().numpy()
        output_results = np.asarray(output_results)
        if output_results.size == 0:
            return self._empty_output()
        if output_results.ndim == 1:
            output_results = output_results.reshape(1, -1)
        return output_results

    def _split_lost_stracks(self):
        """
        划分短时丢失轨迹和长时丢失轨迹。

        原始 ByteTrack 会把 lost 轨迹放在同一个集合里。
        本项目为了增强长遮挡恢复，把它拆成两类：
        - recent_lost：仍参与常规第一阶段关联。
        - long_lost：进入更严格的外观/时序恢复分支。
        """
        # 启用 long recovery 后，把 lost 轨迹切成“最近丢失”和“长时丢失”两组，
        # 后者会进入更严格的恢复分支。
        if not self.with_long_recovery:
            return list(self.lost_stracks), []

        recent_lost = []
        long_lost = []
        for track in self.lost_stracks:
            lost_frames = track.lost_frames(self.frame_id)
            if lost_frames <= self.recent_lost_window:
                recent_lost.append(track)
            else:
                long_lost.append(track)
        return recent_lost, long_lost

    def _extract_features(self, image, dets):
        # ReID 是跟踪增强项，不影响 detector 输出；这里只对高分框提外观特征。
        if not self.with_reid or self.reid_extractor is None or image is None or len(dets) == 0:
            return [None] * len(dets)
        return self.reid_extractor(image, dets)

    def _to_debug_float(self, value):
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return round(value, 6)

    def _serialize_tlwh(self, tlwh):
        return [round(float(v), 3) for v in np.asarray(tlwh).tolist()]

    def _track_state_name(self, track):
        if track.state == TrackState.Tracked:
            return "Tracked"
        if track.state == TrackState.Lost:
            return "Lost"
        if track.state == TrackState.Removed:
            return "Removed"
        return "New"

    def _candidate_center_distance_norm(self, det_a, det_b):
        tlwh_a = np.asarray(det_a.tlwh, dtype=np.float32)
        tlwh_b = np.asarray(det_b.tlwh, dtype=np.float32)
        ac = tlwh_a[:2] + tlwh_a[2:] / 2.0
        bc = tlwh_b[:2] + tlwh_b[2:] / 2.0
        scale = max(float(tlwh_a[3]), float(tlwh_b[3]), 1.0)
        return float(np.linalg.norm(ac - bc) / scale)

    def _candidate_iou(self, det_a, det_b):
        return float(1.0 - matching.iou_distance([det_a], [det_b])[0, 0])

    def _center_distance_norm(self, tlwh_a, tlwh_b):
        tlwh_a = np.asarray(tlwh_a, dtype=np.float32)
        tlwh_b = np.asarray(tlwh_b, dtype=np.float32)
        center_a = tlwh_a[:2] + tlwh_a[2:] / 2.0
        center_b = tlwh_b[:2] + tlwh_b[2:] / 2.0
        scale = max(float(tlwh_a[3]), float(tlwh_b[3]), 1.0)
        return float(np.linalg.norm(center_a - center_b) / scale)

    def _track_det_center_distance_norm(self, track, det):
        return self._center_distance_norm(track.tlwh, det.tlwh)

    def _topk_candidate_indices(self, row_cost, topk):
        valid_cols = np.flatnonzero(np.isfinite(row_cost))
        if valid_cols.size == 0:
            return []
        order = np.argsort(row_cost[valid_cols])
        return valid_cols[order][:topk].tolist()

    def _pair_order_penalty(self, track_a, det_a, track_b, det_b):
        """
        计算两条近距离轨迹是否发生了相对顺序翻转。

        举例：
        - 遮挡前 A 在左、B 在右。
        - 遮挡后如果 A 被分到右侧检测框、B 被分到左侧检测框，
          就可能是一次 ID 互换。

        返回 order_penalty：
        - 0：相对顺序保持一致，或者二者距离太近无法判断顺序。
        - 1：相对顺序明显翻转，需要惩罚。
        """
        track_tlwh_a = np.asarray(track_a.tlwh, dtype=np.float32)
        track_tlwh_b = np.asarray(track_b.tlwh, dtype=np.float32)
        det_tlwh_a = np.asarray(det_a.tlwh, dtype=np.float32)
        det_tlwh_b = np.asarray(det_b.tlwh, dtype=np.float32)

        track_center_a = track_tlwh_a[:2] + track_tlwh_a[2:] / 2.0
        track_center_b = track_tlwh_b[:2] + track_tlwh_b[2:] / 2.0
        det_center_a = det_tlwh_a[:2] + det_tlwh_a[2:] / 2.0
        det_center_b = det_tlwh_b[:2] + det_tlwh_b[2:] / 2.0

        track_delta = track_center_a - track_center_b
        det_delta = det_center_a - det_center_b
        axis = 0 if abs(float(track_delta[0])) >= abs(float(track_delta[1])) else 1
        scale = max(
            float(track_tlwh_a[3]),
            float(track_tlwh_b[3]),
            float(det_tlwh_a[3]),
            float(det_tlwh_b[3]),
            1.0,
        )
        track_gap = abs(float(track_delta[axis])) / scale
        det_gap = abs(float(det_delta[axis])) / scale
        if track_gap < self.anti_swap_order_margin or det_gap < self.anti_swap_order_margin:
            return 0.0, axis, track_gap, det_gap

        track_sign = 1 if float(track_delta[axis]) >= 0.0 else -1
        det_sign = 1 if float(det_delta[axis]) >= 0.0 else -1
        penalty = 0.0 if track_sign == det_sign else 1.0
        return penalty, axis, track_gap, det_gap

    def _pair_assignment_metrics(
        self,
        track_idx_a,
        det_idx_a,
        track_idx_b,
        det_idx_b,
        tracks,
        detections,
        cost_matrix,
    ):
        # anti-swap 不是重新训练一个新模型，而是显式比较两种 2x2 分配：
        # 当前分配 vs 交换后的分配。
        track_a = tracks[track_idx_a]
        track_b = tracks[track_idx_b]
        det_a = detections[det_idx_a]
        det_b = detections[det_idx_b]
        base_cost = float(cost_matrix[track_idx_a, det_idx_a] + cost_matrix[track_idx_b, det_idx_b])
        motion_cost = self._track_det_center_distance_norm(track_a, det_a) + self._track_det_center_distance_norm(
            track_b, det_b
        )
        order_penalty, axis, track_gap, det_gap = self._pair_order_penalty(track_a, det_a, track_b, det_b)
        total_cost = (
            base_cost
            + motion_cost * self.anti_swap_motion_weight
            + order_penalty * self.anti_swap_order_weight
        )
        return {
            "base_cost": base_cost,
            "motion_cost": motion_cost,
            "order_penalty": order_penalty,
            "order_axis": "x" if axis == 0 else "y",
            "track_gap": track_gap,
            "det_gap": det_gap,
            "total_cost": total_cost,
        }

    def _collect_anti_swap_pairs(self, matches, tracks, detections, cost_matrix, ambiguous_map=None):
        # 只对“彼此很近、且互相把对方候选也看成可选项”的轨迹对做反互换检查，
        # 避免把全局匹配问题搞得过于激进。
        candidate_pairs = []
        if len(matches) < 2:
            return candidate_pairs

        ambiguous_map = ambiguous_map or {}
        match_map = {int(track_idx): int(det_idx) for track_idx, det_idx in matches}
        matched_track_indices = sorted(match_map.keys())

        for pos_a, track_idx_a in enumerate(matched_track_indices):
            track_a = tracks[track_idx_a]
            if track_a.state != TrackState.Tracked:
                continue
            det_idx_a = match_map[track_idx_a]
            det_a = detections[det_idx_a]
            topk_a = set(self._topk_candidate_indices(cost_matrix[track_idx_a], self.anti_swap_topk))
            ambiguous_a = set(ambiguous_map.get(track_idx_a, {}).get("det_indices", []))
            for track_idx_b in matched_track_indices[pos_a + 1 :]:
                track_b = tracks[track_idx_b]
                if track_b.state != TrackState.Tracked:
                    continue
                det_idx_b = match_map[track_idx_b]
                if det_idx_a == det_idx_b:
                    continue
                det_b = detections[det_idx_b]
                topk_b = set(self._topk_candidate_indices(cost_matrix[track_idx_b], self.anti_swap_topk))
                ambiguous_b = set(ambiguous_map.get(track_idx_b, {}).get("det_indices", []))

                cross_candidate = (
                    (det_idx_b in topk_a or det_idx_b in ambiguous_a)
                    and (det_idx_a in topk_b or det_idx_a in ambiguous_b)
                )
                if not cross_candidate:
                    continue

                track_center_gap = self._center_distance_norm(track_a.tlwh, track_b.tlwh)
                if track_center_gap > self.anti_swap_track_center_thresh:
                    continue

                det_center_gap = self._candidate_center_distance_norm(det_a, det_b)
                det_iou = self._candidate_iou(det_a, det_b)
                if det_center_gap > self.anti_swap_det_center_thresh and det_iou < self.anti_swap_det_iou_thresh:
                    continue

                candidate_pairs.append(
                    {
                        "track_indices": (int(track_idx_a), int(track_idx_b)),
                        "det_indices": (int(det_idx_a), int(det_idx_b)),
                        "track_center_gap": float(track_center_gap),
                        "det_center_gap": float(det_center_gap),
                        "det_iou": float(det_iou),
                    }
                )

        candidate_pairs.sort(key=lambda item: (item["det_center_gap"], item["track_center_gap"]))
        return candidate_pairs

    def _refine_close_pair_matches(self, matches, tracks, detections, cost_matrix, temporal_debug=None):
        """
        对近距离双人匹配结果做 2x2 局部复核。

        原始匈牙利匹配是全局最优，但在近距离双人交错遮挡场景中，
        单帧代价可能会把两个人互换。这里不重做全局匹配，只检查小范围候选：
        - 当前匹配：track_a -> det_a, track_b -> det_b
        - 交换匹配：track_a -> det_b, track_b -> det_a

        如果交换匹配在“原始代价 + 运动连续性 + 相对顺序”联合代价上明显更优，
        就翻转这两个匹配。
        """
        empty_debug = {"pairs": [], "overrides": []}
        if (
            not self.with_anti_swap
            or len(matches) < 2
            or len(tracks) == 0
            or len(detections) == 0
        ):
            return matches, empty_debug

        ambiguous_map = temporal_debug.get("ambiguous_map", {}) if temporal_debug is not None else {}
        candidate_pairs = self._collect_anti_swap_pairs(matches, tracks, detections, cost_matrix, ambiguous_map)
        if len(candidate_pairs) == 0:
            return matches, empty_debug

        match_map = {int(track_idx): int(det_idx) for track_idx, det_idx in matches}
        used_tracks = set()
        pair_records = []
        overrides = []

        for pair in candidate_pairs:
            track_idx_a, track_idx_b = pair["track_indices"]
            if track_idx_a in used_tracks or track_idx_b in used_tracks:
                continue

            det_idx_a = match_map.get(track_idx_a)
            det_idx_b = match_map.get(track_idx_b)
            if det_idx_a is None or det_idx_b is None or det_idx_a == det_idx_b:
                continue

            current_metrics = self._pair_assignment_metrics(
                track_idx_a,
                det_idx_a,
                track_idx_b,
                det_idx_b,
                tracks,
                detections,
                cost_matrix,
            )
            swapped_metrics = self._pair_assignment_metrics(
                track_idx_a,
                det_idx_b,
                track_idx_b,
                det_idx_a,
                tracks,
                detections,
                cost_matrix,
            )

            chosen = "current"
            overridden = False
            # 只有在“交换后确实更优”且优势超过 margin 时才翻转，
            # 防止近距离场景下频繁来回抖动。
            if swapped_metrics["total_cost"] + self.anti_swap_margin < current_metrics["total_cost"]:
                match_map[track_idx_a], match_map[track_idx_b] = det_idx_b, det_idx_a
                chosen = "swap"
                overridden = True
                overrides.append(
                    {
                        "track_pool_index": int(track_idx_a),
                        "track_id": int(getattr(tracks[track_idx_a], "track_id", -1)),
                        "new_det_index": int(det_idx_b),
                    }
                )
                overrides.append(
                    {
                        "track_pool_index": int(track_idx_b),
                        "track_id": int(getattr(tracks[track_idx_b], "track_id", -1)),
                        "new_det_index": int(det_idx_a),
                    }
                )

            used_tracks.add(track_idx_a)
            used_tracks.add(track_idx_b)
            pair_records.append(
                {
                    "track_pool_indices": [int(track_idx_a), int(track_idx_b)],
                    "track_ids": [
                        int(getattr(tracks[track_idx_a], "track_id", -1)),
                        int(getattr(tracks[track_idx_b], "track_id", -1)),
                    ],
                    "original_det_indices": [int(det_idx_a), int(det_idx_b)],
                    "track_center_gap": self._to_debug_float(pair["track_center_gap"]),
                    "det_center_gap": self._to_debug_float(pair["det_center_gap"]),
                    "det_iou": self._to_debug_float(pair["det_iou"]),
                    "current": {
                        "det_indices": [int(det_idx_a), int(det_idx_b)],
                        "base_cost": self._to_debug_float(current_metrics["base_cost"]),
                        "motion_cost": self._to_debug_float(current_metrics["motion_cost"]),
                        "order_penalty": self._to_debug_float(current_metrics["order_penalty"]),
                        "order_axis": current_metrics["order_axis"],
                        "track_gap": self._to_debug_float(current_metrics["track_gap"]),
                        "det_gap": self._to_debug_float(current_metrics["det_gap"]),
                        "total_cost": self._to_debug_float(current_metrics["total_cost"]),
                    },
                    "swap": {
                        "det_indices": [int(det_idx_b), int(det_idx_a)],
                        "base_cost": self._to_debug_float(swapped_metrics["base_cost"]),
                        "motion_cost": self._to_debug_float(swapped_metrics["motion_cost"]),
                        "order_penalty": self._to_debug_float(swapped_metrics["order_penalty"]),
                        "order_axis": swapped_metrics["order_axis"],
                        "track_gap": self._to_debug_float(swapped_metrics["track_gap"]),
                        "det_gap": self._to_debug_float(swapped_metrics["det_gap"]),
                        "total_cost": self._to_debug_float(swapped_metrics["total_cost"]),
                    },
                    "chosen": chosen,
                    "overridden": overridden,
                }
            )

        refined_matches = np.asarray(sorted(match_map.items()), dtype=int) if len(match_map) > 0 else np.empty((0, 2), dtype=int)
        return refined_matches, {"pairs": pair_records, "overrides": overrides}

    def _build_first_stage_debug(
        self,
        tracks,
        detections,
        iou_cost,
        score_cost,
        appearance_debug,
        temporal_debug,
        final_cost,
        matches,
        unmatched_tracks,
        unmatched_dets,
        anti_swap_debug=None,
    ):
        if not self.debug_association:
            return None

        ambiguous_map = temporal_debug.get("ambiguous_map", {}) if temporal_debug is not None else {}
        anti_swap_pairs = [] if anti_swap_debug is None else anti_swap_debug.get("pairs", [])
        if self.debug_ambiguous_only and len(ambiguous_map) == 0 and len(anti_swap_pairs) == 0:
            return None

        appearance_cost = None if appearance_debug is None else appearance_debug.get("appearance_cost")
        smooth_cost = None if appearance_debug is None else appearance_debug.get("smooth_cost")
        gallery_cost = None if appearance_debug is None else appearance_debug.get("gallery_cost")
        reid_fused_cost = None if appearance_debug is None else appearance_debug.get("fused_cost")
        temporal_scores = None if temporal_debug is None else temporal_debug.get("temporal_scores")
        temporal_fused_cost = None if temporal_debug is None else temporal_debug.get("fused_cost")

        row_records = []
        for track_idx, track in enumerate(tracks):
            candidate_indices = np.flatnonzero(np.isfinite(final_cost[track_idx]))
            if candidate_indices.size == 0:
                continue
            ordered_indices = candidate_indices[np.argsort(final_cost[track_idx][candidate_indices])]
            top_indices = list(islice(ordered_indices.tolist(), self.debug_topk))
            candidate_records = []
            ambiguous_entry = ambiguous_map.get(track_idx, {})
            ambiguous_candidates = set(ambiguous_entry.get("det_indices", []))
            for det_idx in top_indices:
                det = detections[det_idx]
                candidate_records.append(
                    {
                        "det_index": int(det_idx),
                        "det_score": self._to_debug_float(det.score),
                        "det_tlwh": self._serialize_tlwh(det.tlwh),
                        "iou_cost": self._to_debug_float(iou_cost[track_idx, det_idx]),
                        "score_cost": self._to_debug_float(score_cost[track_idx, det_idx]),
                        "smooth_cost": None if smooth_cost is None else self._to_debug_float(smooth_cost[track_idx, det_idx]),
                        "gallery_cost": None if gallery_cost is None else self._to_debug_float(gallery_cost[track_idx, det_idx]),
                        "appearance_cost": None if appearance_cost is None else self._to_debug_float(appearance_cost[track_idx, det_idx]),
                        "reid_fused_cost": None if reid_fused_cost is None else self._to_debug_float(reid_fused_cost[track_idx, det_idx]),
                        "temporal_score": None if temporal_scores is None else self._to_debug_float(temporal_scores[track_idx, det_idx]),
                        "temporal_fused_cost": None if temporal_fused_cost is None else self._to_debug_float(temporal_fused_cost[track_idx, det_idx]),
                        "final_cost": self._to_debug_float(final_cost[track_idx, det_idx]),
                        "ambiguous_candidate": bool(det_idx in ambiguous_candidates),
                    }
                )

            matched_det = None
            for matched_track_idx, matched_det_idx in matches:
                if int(matched_track_idx) == track_idx:
                    matched_det = int(matched_det_idx)
                    break
            row_records.append(
                {
                    "track_pool_index": int(track_idx),
                    "track_id": int(getattr(track, "track_id", -1)),
                    "state": self._track_state_name(track),
                    "track_tlwh": self._serialize_tlwh(track.tlwh),
                    "matched_det_index": matched_det,
                    "ambiguous": track_idx in ambiguous_map,
                    "ambiguous_reason": ambiguous_entry.get("reason"),
                    "candidates": candidate_records,
                }
            )

        debug_record = {
            "frame_id": int(self.frame_id),
            "stage": "first_stage",
            "num_tracks": int(len(tracks)),
            "num_detections": int(len(detections)),
            "ambiguous_track_indices": [int(idx) for idx in sorted(ambiguous_map.keys())],
            "ambiguous_tracks": [
                {
                    "track_pool_index": int(idx),
                    "track_id": int(getattr(tracks[idx], "track_id", -1)),
                    "reason": ambiguous_map[idx].get("reason"),
                    "candidate_det_indices": [int(v) for v in ambiguous_map[idx].get("det_indices", [])],
                }
                for idx in sorted(ambiguous_map.keys())
            ],
            "matches": [
                {
                    "track_pool_index": int(track_idx),
                    "track_id": int(getattr(tracks[track_idx], "track_id", -1)),
                    "det_index": int(det_idx),
                    "final_cost": self._to_debug_float(final_cost[track_idx, det_idx]),
                }
                for track_idx, det_idx in matches
            ],
            "unmatched_track_indices": [int(idx) for idx in unmatched_tracks],
            "unmatched_det_indices": [int(idx) for idx in unmatched_dets],
            "tracks": row_records,
            "anti_swap_pairs": anti_swap_pairs,
            "anti_swap_overrides": [] if anti_swap_debug is None else anti_swap_debug.get("overrides", []),
        }
        return debug_record

    def _build_detection(self, tlbr, score, feat=None):
        return STrack(
            STrack.tlbr_to_tlwh(tlbr),
            score,
            temp_feat=feat,
            feat_history=self.feature_history,
            gallery_size=self.gallery_size,
            prototype_count=self.prototype_count,
            gallery_recent_count=self.gallery_recent_count,
            gallery_update_thresh=self.gallery_update_thresh,
            prototype_sim_thresh=self.prototype_sim_thresh,
            prototype_momentum=self.prototype_momentum,
        )

    def _appearance_distance(self, tracks, detections, return_components=False):
        """
        计算轨迹与检测框之间的外观距离矩阵。

        返回矩阵形状是 [num_tracks, num_detections]。
        数值越小表示越相似，越适合匹配。

        当 with_gallery=False 时：
        - 使用 smooth_feat 做普通 embedding distance。

        当 with_gallery=True 时：
        - 额外融合 gallery/prototype 距离，让外观匹配不只依赖一个平滑向量。
        """
        # appearance_cost 默认来自 smooth_feat；
        # 如果启用了 gallery，则再把多原型记忆融合进去。
        if not self.with_reid or len(tracks) == 0 or len(detections) == 0:
            return (None, None) if return_components else None

        valid_track_feats = all(track.smooth_feat is not None for track in tracks)
        valid_det_feats = all(det.curr_feat is not None for det in detections)
        if not (valid_track_feats and valid_det_feats):
            return (None, None) if return_components else None

        smooth_cost = matching.embedding_distance(tracks, detections) / 2.0
        smooth_cost = np.clip(smooth_cost, 0.0, 1.0)
        appearance_cost = smooth_cost.copy()
        gallery_cost = None

        if self.with_gallery:
            gallery_cost = matching.gallery_embedding_distance(
                tracks,
                detections,
                topk=self.gallery_topk,
            ) / 2.0
            valid_rows = np.all(np.isfinite(gallery_cost), axis=1)
            if np.any(valid_rows):
                gallery_cost = np.clip(gallery_cost, 0.0, 1.0)
                appearance_cost[valid_rows] = (
                    appearance_cost[valid_rows] * (1.0 - self.gallery_match_weight)
                    + gallery_cost[valid_rows] * self.gallery_match_weight
                )
        if return_components:
            return appearance_cost, {
                "smooth_cost": smooth_cost,
                "gallery_cost": gallery_cost,
                "appearance_cost": appearance_cost.copy(),
            }
        return appearance_cost

    def _fuse_first_stage_reid(self, cost_matrix, tracks, detections, return_debug=False):
        """
        将 ReID 外观代价融合进第一阶段关联矩阵。

        参数 cost_matrix 通常已经包含 IoU cost 和 detection score 融合结果。
        这里会做：
        1. 计算 appearance_cost。
        2. 用 Kalman gate 过滤运动上明显不合理的候选。
        3. 用 appearance_thresh 把外观不可信的候选置为高代价。
        4. 按 reid_weight 与原始代价融合。

        注意：这里使用 np.minimum，表示外观信息只在它能提供更可信匹配时降低代价，
        尽量避免 ReID 错误强行覆盖运动几何信息。
        """
        # 第一阶段关联中，把 IoU/score cost 和外观 cost 融合成最终代价矩阵。
        if not self.with_reid or len(tracks) == 0 or len(detections) == 0:
            return (cost_matrix, None) if return_debug else cost_matrix

        appearance_result = self._appearance_distance(
            tracks,
            detections,
            return_components=return_debug,
        )
        if return_debug:
            appearance_cost, appearance_components = appearance_result
        else:
            appearance_cost, appearance_components = appearance_result, None
        if appearance_cost is None:
            return (cost_matrix, None) if return_debug else cost_matrix

        appearance_cost = matching.gate_cost_matrix(self.kalman_filter, appearance_cost, tracks, detections)
        appearance_cost[appearance_cost > self.appearance_thresh] = 1.0
        fused_cost = np.minimum(
            cost_matrix,
            appearance_cost * self.reid_weight + cost_matrix * (1.0 - self.reid_weight),
        )
        if return_debug:
            appearance_debug = appearance_components or {}
            appearance_debug["appearance_cost"] = appearance_cost.copy()
            appearance_debug["fused_cost"] = fused_cost.copy()
            return fused_cost, appearance_debug
        return fused_cost

    def _collect_ambiguous_candidates(self, cost_matrix, tracks, detections):
        """
        收集第一阶段中的歧义匹配候选。

        触发条件包括：
        - top-2 候选代价非常接近，说明匈牙利匹配可能不稳定。
        - top-2 候选检测框中心距离很近，说明可能处于拥挤/遮挡区域。
        - top-2 候选检测框有重叠，说明存在近距离交互。

        返回值 ambiguous 是一个 dict：
        - key：轨迹在 track pool 中的行索引。
        - value：该轨迹需要 temporal rerank 的候选检测框列索引和触发原因。
        """
        # 这一步专门收集“top-2 很接近”或“两个候选框挤得很近”的歧义场景，
        # 供 temporal matcher 做局部 rerank。
        ambiguous = {}
        if cost_matrix.size == 0:
            return ambiguous

        for row in range(cost_matrix.shape[0]):
            track = tracks[row]
            if self.temporal_ambiguity_tracked_only and track.state != TrackState.Tracked:
                continue
            row_cost = cost_matrix[row]
            valid_cols = np.flatnonzero(np.isfinite(row_cost))
            if valid_cols.size < 2:
                continue

            valid_costs = row_cost[valid_cols]
            order = np.argsort(valid_costs)
            sorted_cols = valid_cols[order]
            sorted_costs = valid_costs[order]
            best_cost = float(sorted_costs[0])
            second_cost = float(sorted_costs[1])
            if best_cost > self.temporal_ambiguity_cost_thresh:
                continue

            det_a = detections[int(sorted_cols[0])]
            det_b = detections[int(sorted_cols[1])]
            cost_gap = second_cost - best_cost
            center_gap = self._candidate_center_distance_norm(det_a, det_b)
            det_iou = self._candidate_iou(det_a, det_b)

            reason = None
            if cost_gap <= self.temporal_ambiguity_gap:
                reason = "cost_gap"
            elif center_gap <= self.temporal_proximity_center_thresh:
                reason = "crowd_center"
            elif det_iou >= self.temporal_proximity_iou_thresh:
                reason = "crowd_iou"
            else:
                continue

            candidate_limit = min(best_cost + self.temporal_ambiguity_gap, self.temporal_ambiguity_cost_thresh)
            candidate_cols = sorted_cols[sorted_costs <= candidate_limit][: self.temporal_ambiguity_topk]
            if candidate_cols.size < 2:
                candidate_cols = sorted_cols[: self.temporal_ambiguity_topk]
            if candidate_cols.size < 2:
                continue
            ambiguous[row] = {
                "det_indices": candidate_cols.tolist(),
                "reason": reason,
                "cost_gap": float(cost_gap),
                "center_gap": float(center_gap),
                "det_iou": float(det_iou),
            }
        return ambiguous

    def _fuse_first_stage_temporal_ambiguity(self, cost_matrix, tracks, detections, image_size, return_debug=False):
        """
        使用 temporal matcher 对第一阶段歧义候选做局部重打分。

        它不会对整张代价矩阵全量运行 Transformer，而是只处理 _collect_ambiguous_candidates()
        找出的少量歧义 pair。这样做的好处是：
        - 计算量更小。
        - 避免时序模型过度干预本来已经很确定的匹配。
        """
        # 注意：temporal matcher 不是替代匈牙利匹配，而是只改写歧义候选的局部代价。
        if (
            not self.with_temporal_ambiguity
            or self.temporal_matcher is None
            or len(tracks) == 0
            or len(detections) == 0
        ):
            return (cost_matrix, None) if return_debug else cost_matrix

        ambiguous = self._collect_ambiguous_candidates(cost_matrix, tracks, detections)
        if len(ambiguous) == 0:
            debug_info = {
                "ambiguous_map": {},
                "temporal_scores": None,
                "fused_cost": cost_matrix.copy(),
            }
            return (cost_matrix, debug_info) if return_debug else cost_matrix

        track_indices = sorted(ambiguous.keys())
        det_indices = sorted(
            {
                det_idx
                for entry in ambiguous.values()
                for det_idx in entry.get("det_indices", [])
            }
        )
        selected_tracks = [tracks[idx] for idx in track_indices]
        selected_dets = [detections[idx] for idx in det_indices]
        temporal_scores = self.temporal_matcher.pairwise_score(
            selected_tracks,
            selected_dets,
            frame_id=self.frame_id,
            image_size=image_size,
        )
        if temporal_scores.size == 0:
            debug_info = {
                "ambiguous_map": ambiguous,
                "temporal_scores": None,
                "fused_cost": cost_matrix.copy(),
            }
            return (cost_matrix, debug_info) if return_debug else cost_matrix

        det_index_map = {det_idx: local_idx for local_idx, det_idx in enumerate(det_indices)}
        fused_cost = cost_matrix.copy()
        full_temporal_scores = np.zeros_like(cost_matrix, dtype=np.float32)
        for local_track_idx, global_track_idx in enumerate(track_indices):
            candidate_cols = ambiguous[global_track_idx]["det_indices"]
            local_det_indices = [det_index_map[det_idx] for det_idx in candidate_cols]
            row_scores = temporal_scores[local_track_idx, local_det_indices]
            for det_idx, local_det_idx in zip(candidate_cols, local_det_indices):
                full_temporal_scores[global_track_idx, det_idx] = temporal_scores[local_track_idx, local_det_idx]
            if row_scores.size == 0:
                continue
            best_temporal_pos = int(np.argmax(row_scores))
            best_temporal_score = float(row_scores[best_temporal_pos])
            if best_temporal_score < self.temporal_ambiguity_score_thresh:
                continue

            second_temporal_score = 0.0
            if row_scores.size >= 2:
                second_temporal_score = float(np.partition(row_scores, -2)[-2])
            score_margin = best_temporal_score - second_temporal_score
            if score_margin < self.temporal_ambiguity_score_margin:
                continue

            base_row_costs = cost_matrix[global_track_idx, candidate_cols]
            base_best_pos = int(np.argmin(base_row_costs))
            base_best_cost = float(base_row_costs[base_best_pos])
            temporal_best_det_idx = int(candidate_cols[best_temporal_pos])
            temporal_cost = 1.0 - best_temporal_score
            temporal_fused_cost = (
                float(base_row_costs[best_temporal_pos]) * (1.0 - self.temporal_ambiguity_weight)
                + temporal_cost * self.temporal_ambiguity_weight
            )

            # Transformer 只有在“明显更自信”且“确实能带来足够代价改进”时，
            # 才允许推翻当前第一阶段的最佳候选，避免近距离双人场景下轻微分差导致误切换。
            if best_temporal_pos != base_best_pos:
                improvement = base_best_cost - temporal_fused_cost
                if improvement < self.temporal_ambiguity_override_margin:
                    continue

            fused_cost[global_track_idx, temporal_best_det_idx] = min(
                fused_cost[global_track_idx, temporal_best_det_idx],
                temporal_fused_cost,
            )
        debug_info = {
            "ambiguous_map": ambiguous,
            "temporal_scores": full_temporal_scores,
            "fused_cost": fused_cost.copy(),
        }
        return (fused_cost, debug_info) if return_debug else fused_cost

    def _match_long_lost_tracks(self, lost_tracks, detections, image_size):
        """
        长时丢失轨迹恢复。

        这一阶段只处理已经 lost 较久的轨迹，目的不是普通帧间匹配，
        而是判断“这个重新出现的检测框是否属于之前消失的某条轨迹”。

        关联依据更偏外观和时序：
        - appearance distance
        - Kalman gate
        - optional temporal matcher score
        """
        # 这一支负责“长时间遮挡后重新出现”的恢复。
        # 它比常规第一阶段更依赖外观和时序，而不是单纯 IoU。
        if not self.with_long_recovery or len(lost_tracks) == 0 or len(detections) == 0:
            return np.empty((0, 2), dtype=int), tuple(range(len(lost_tracks))), tuple(range(len(detections)))

        valid_track_indices = [
            idx for idx, track in enumerate(lost_tracks)
            if track.mean is not None and track.covariance is not None and track.smooth_feat is not None
        ]
        valid_det_indices = [idx for idx, det in enumerate(detections) if det.curr_feat is not None]
        if len(valid_track_indices) == 0 or len(valid_det_indices) == 0:
            return np.empty((0, 2), dtype=int), tuple(range(len(lost_tracks))), tuple(range(len(detections)))

        valid_tracks = [lost_tracks[idx] for idx in valid_track_indices]
        valid_dets = [detections[idx] for idx in valid_det_indices]

        recovery_cost = self._appearance_distance(valid_tracks, valid_dets)
        if recovery_cost is None:
            return np.empty((0, 2), dtype=int), tuple(range(len(lost_tracks))), tuple(range(len(detections)))
        recovery_cost = matching.gate_cost_matrix(self.kalman_filter, recovery_cost, valid_tracks, valid_dets)
        recovery_cost[recovery_cost > self.lost_appearance_thresh] = np.inf
        recovery_cost = matching.fuse_motion(
            self.kalman_filter,
            recovery_cost,
            valid_tracks,
            valid_dets,
            lambda_=self.long_motion_lambda,
        )
        if self.with_temporal_matcher and self.temporal_matcher is not None:
            temporal_scores = self.temporal_matcher.pairwise_score(
                valid_tracks,
                valid_dets,
                frame_id=self.frame_id,
                image_size=image_size,
            )
            temporal_cost = 1.0 - temporal_scores
            temporal_cost[temporal_scores < self.temporal_score_thresh] = np.inf
            fused_temporal_cost = recovery_cost * (1.0 - self.temporal_weight) + temporal_cost * self.temporal_weight
            recovery_cost = np.minimum(recovery_cost, fused_temporal_cost)

        matches, unmatched_tracks, unmatched_dets = matching.linear_assignment(
            recovery_cost, thresh=self.long_match_thresh
        )
        if len(matches) == 0:
            return np.empty((0, 2), dtype=int), tuple(range(len(lost_tracks))), tuple(range(len(detections)))

        remapped_matches = np.asarray(
            [[valid_track_indices[i], valid_det_indices[j]] for i, j in matches],
            dtype=int,
        )
        matched_track_set = {m[0] for m in remapped_matches}
        matched_det_set = {m[1] for m in remapped_matches}
        all_tracks = tuple(idx for idx in range(len(lost_tracks)) if idx not in matched_track_set)
        all_dets = tuple(idx for idx in range(len(detections)) if idx not in matched_det_set)
        return remapped_matches, all_tracks, all_dets

    def _apply_cmc(self, image, track_groups):
        # 如果机位在动，先对轨迹做一次全局运动补偿，再进入关联。
        if self.cmc is None or image is None:
            return
        warp = self.cmc.estimate(image)
        for stracks in track_groups:
            STrack.multi_gmc(stracks, warp)

    def update(self, output_results, img_info, img_size, img=None):
        """
        处理单帧检测结果并输出当前帧激活轨迹。

        参数说明：
        - output_results：检测器输出，格式通常是 [x1,y1,x2,y2,score,...]。
        - img_info：原图高宽信息，用来把检测框映射回原图坐标。
        - img_size：检测器输入尺寸，例如 YOLOX/YOLOv8 的 test_size。
        - img：原始 BGR 图像，ReID 裁剪和 GMC 需要它。

        返回：
        - output_stracks：当前帧处于 Tracked 且已激活的轨迹列表。

        这是整套跟踪算法的主函数，也是论文“算法流程”部分最应该展开描述的函数。
        """
        # 这是整条推理链最关键的入口。
        # 每来一帧，都按下面顺序走：
        # 1. 整理 detector 输出
        # 2. 高分框第一阶段关联
        # 3. 低分框第二阶段补关联
        # 4. long-lost recovery
        # 5. 新轨迹创建 / 丢失轨迹清理
        self.frame_id += 1
        self.last_debug_frame = None
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        output_results = self._prepare_output(output_results)
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]

        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes = bboxes / scale

        # ByteTrack 的经典做法：把检测框拆成高分与低分两组。
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        det_features = self._extract_features(img, dets)
        detections = [
            self._build_detection(tlbr, score, feat=feat)
            for tlbr, score, feat in zip(dets, scores_keep, det_features)
        ]

        detections_second = [
            self._build_detection(tlbr, score)
            for tlbr, score in zip(dets_second, scores_second)
        ]

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # 第一阶段的候选轨迹池 = 当前 tracked + 最近 lost。
        recent_lost_stracks, long_lost_stracks = self._split_lost_stracks()
        strack_pool = joint_stracks(tracked_stracks, recent_lost_stracks)
        STrack.multi_predict(strack_pool)
        STrack.multi_predict(unconfirmed)
        STrack.multi_predict(long_lost_stracks)
        self._apply_cmc(img, [strack_pool, unconfirmed, long_lost_stracks])

        # 第一阶段：高分检测框关联。
        iou_dists = matching.iou_distance(strack_pool, detections)
        dists = iou_dists.copy()
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        score_fused_dists = dists.copy()
        reid_result = self._fuse_first_stage_reid(
            dists,
            strack_pool,
            detections,
            return_debug=self.debug_association,
        )
        if self.debug_association:
            dists, appearance_debug = reid_result
        else:
            dists, appearance_debug = reid_result, None

        temporal_result = self._fuse_first_stage_temporal_ambiguity(
            dists,
            strack_pool,
            detections,
            image_size=(img_h, img_w),
            return_debug=self.debug_association,
        )
        if self.debug_association:
            dists, temporal_debug = temporal_result
        else:
            dists, temporal_debug = temporal_result, None

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        matches, anti_swap_debug = self._refine_close_pair_matches(
            matches,
            strack_pool,
            detections,
            dists,
            temporal_debug=temporal_debug,
        )
        self.last_debug_frame = self._build_first_stage_debug(
            strack_pool,
            detections,
            iou_dists,
            score_fused_dists,
            appearance_debug,
            temporal_debug,
            dists,
            matches,
            u_track,
            u_detection,
            anti_swap_debug=anti_swap_debug,
        )
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 第二阶段：只对第一阶段未匹配的 tracked 轨迹，用低分框再做一次补关联。
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # 未确认轨迹单独处理，避免一帧假阳性过快成长为稳定轨迹。
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # 长时丢失轨迹恢复分支。
        detections = [detections[i] for i in u_detection]
        recovery_matches, _, u_detection = self._match_long_lost_tracks(
            long_lost_stracks,
            detections,
            image_size=(img_h, img_w),
        )
        for ilost, idet in recovery_matches:
            track = long_lost_stracks[ilost]
            det = detections[idet]
            track.re_activate(det, self.frame_id, new_id=False)
            refind_stracks.append(track)

        # 仍然剩下的高分检测框会被视为新目标，创建新轨迹。
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # 最后统一维护 tracked / lost / removed 三个集合，作为下一帧的输入状态。
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
