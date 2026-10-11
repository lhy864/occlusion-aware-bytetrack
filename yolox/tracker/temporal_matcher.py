#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
时序关联模块。

它不是像 MOTR/TransTrack 那样端到端替代 ByteTrack，
而是作为一个“关联打分头”插在现有 ByteTrack 上：
1. 输入轨迹最近若干帧的外观特征和框几何信息。
2. 输入当前候选检测框的外观特征和几何信息。
3. 输出这条轨迹与该候选框是否属于同一人的匹配概率。
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn


_TEMPORAL_MATCHER_CACHE = {}


def _normalize_device(device):
    # 兼容 torch.device / "cuda:0" / "cpu" / int 等多种传法。
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        if device == "cpu":
            return torch.device("cpu")
        if device.startswith("cuda"):
            return torch.device(device)
    if isinstance(device, int):
        return torch.device("cuda:{}".format(device))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tlwh_to_xyah(tlwh):
    # ByteTrack/KalmanFilter 内部大量使用 xyah 表示，
    # 这里把 tlwh 转成更适合时序建模的几何表示。
    tlwh = np.asarray(tlwh, dtype=np.float32)
    x = tlwh[0] + tlwh[2] / 2.0
    y = tlwh[1] + tlwh[3] / 2.0
    a = tlwh[2] / max(tlwh[3], 1e-6)
    h = tlwh[3]
    return np.asarray([x, y, a, h], dtype=np.float32)


def encode_geometry(tlwh, image_size, delta_t, max_dt):
    # 几何编码包含位置、尺度和时间间隔，用来告诉 Transformer：
    # “这个候选框离历史轨迹有多远、隔了多久、尺度是否合理”。
    image_h, image_w = image_size
    x, y, a, h = tlwh_to_xyah(tlwh)
    return np.asarray(
        [
            x / max(float(image_w), 1.0),
            y / max(float(image_h), 1.0),
            np.clip(a / 4.0, 0.0, 1.0),
            h / max(float(image_h), 1.0),
            np.clip(float(delta_t) / max(float(max_dt), 1.0), 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def build_temporal_pair_inputs(
    history_features,
    history_boxes,
    history_frames,
    candidate_feature,
    candidate_tlwh,
    candidate_frame,
    image_size,
    max_history,
    max_dt,
    min_history=3,
):
    """
    构造单个“轨迹历史 - 候选检测框”样本。

    输入包含：
    - history_features：轨迹过去若干帧的 ReID 特征。
    - history_boxes：轨迹过去若干帧的位置框。
    - history_frames：这些历史观测来自哪些帧。
    - candidate_feature/candidate_tlwh：当前候选检测框的信息。

    输出是 Transformer 需要的固定长度张量：
    - history_feat：历史外观序列。
    - history_geom：历史几何和时间间隔编码。
    - history_mask：哪些位置是 padding。
    - candidate_feat/candidate_geom：候选框 token。
    """
    if candidate_feature is None:
        return None

    history_features = list(history_features)
    history_boxes = list(history_boxes)
    history_frames = list(history_frames)
    history_count = min(len(history_features), len(history_boxes), len(history_frames))
    if history_count < min_history:
        return None

    feat_dim = int(np.asarray(candidate_feature, dtype=np.float32).shape[-1])
    # 这里把可变长度轨迹历史整理成固定长度张量，方便后续批量训练与推理。
    history_feat = np.zeros((max_history, feat_dim), dtype=np.float32)
    history_geom = np.zeros((max_history, 5), dtype=np.float32)
    history_mask = np.ones((max_history,), dtype=bool)

    history_features = history_features[-max_history:]
    history_boxes = history_boxes[-max_history:]
    history_frames = history_frames[-max_history:]
    offset = max_history - len(history_features)
    for idx, (feat, tlwh, frame_id) in enumerate(zip(history_features, history_boxes, history_frames)):
        history_feat[offset + idx] = np.asarray(feat, dtype=np.float32)
        history_geom[offset + idx] = encode_geometry(
            tlwh,
            image_size=image_size,
            delta_t=max(candidate_frame - frame_id, 0),
            max_dt=max_dt,
        )
        history_mask[offset + idx] = False

    candidate_feature = np.asarray(candidate_feature, dtype=np.float32)
    candidate_geom = encode_geometry(
        candidate_tlwh,
        image_size=image_size,
        delta_t=0,
        max_dt=max_dt,
    )
    return {
        "history_feat": history_feat,
        "history_geom": history_geom,
        "history_mask": history_mask,
        "candidate_feat": candidate_feature,
        "candidate_geom": candidate_geom,
    }


class TemporalAssociationTransformer(nn.Module):
    """
    轻量时序关联 Transformer。

    它的任务是二分类：
    - 输入：一条轨迹的历史序列 + 一个当前候选检测框。
    - 输出：二者是否属于同一身份的 logit。

    它不是检测器，也不是完整跟踪器，只是 ByteTrack 关联阶段的一个辅助评分器。
    """

    def __init__(
        self,
        feat_dim=512,
        geom_dim=5,
        d_model=256,
        nhead=8,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        max_history=8,
        max_dt=120,
        min_history=3,
    ):
        super().__init__()
        # 这个模型本质上是一个轻量 Transformer encoder：
        # - feature_proj 处理外观 embedding
        # - geometry_proj 处理框与时间间隔
        # - 最终 head 输出二分类 logit
        self.feat_dim = feat_dim
        self.geom_dim = geom_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_history = max_history
        self.max_dt = max_dt
        self.min_history = min_history

        self.feature_proj = nn.Linear(feat_dim, d_model)
        self.geometry_proj = nn.Sequential(
            nn.Linear(geom_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.type_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Embedding(max_history + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def get_config(self):
        return {
            "feat_dim": self.feat_dim,
            "geom_dim": self.geom_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "max_history": self.max_history,
            "max_dt": self.max_dt,
            "min_history": self.min_history,
        }

    def forward(self, history_feat, history_geom, history_mask, candidate_feat, candidate_geom):
        """
        前向传播。

        history_feat/history_geom 会被编码成历史 token；
        candidate_feat/candidate_geom 会被编码成候选 token；
        然后通过 TransformerEncoder 建模历史与候选之间的关系。
        """
        batch_size, history_len, _ = history_feat.shape
        device = history_feat.device

        # 历史 token 与候选 token 共用一套编码空间，
        # 再用 type_embed 区分“历史轨迹片段”和“当前候选框”。
        hist_tokens = self.feature_proj(history_feat) + self.geometry_proj(history_geom)
        hist_tokens = hist_tokens + self.type_embed(
            torch.zeros((batch_size, history_len), dtype=torch.long, device=device)
        )
        hist_tokens = hist_tokens + self.pos_embed(
            torch.arange(history_len, device=device, dtype=torch.long)
        ).unsqueeze(0)

        candidate_token = self.feature_proj(candidate_feat).unsqueeze(1) + self.geometry_proj(
            candidate_geom
        ).unsqueeze(1)
        candidate_token = candidate_token + self.type_embed(
            torch.ones((batch_size, 1), dtype=torch.long, device=device)
        )
        candidate_token = candidate_token + self.pos_embed.weight[history_len].view(1, 1, -1)

        tokens = torch.cat([hist_tokens, candidate_token], dim=1)
        padding_mask = torch.cat(
            [history_mask, torch.zeros((batch_size, 1), dtype=torch.bool, device=device)], dim=1
        )
        # padding_mask 会让补零位置不参与注意力，避免短轨迹被无意义的零特征污染。
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)

        # 对历史 token 做 masked average pooling，再与候选 token 拼接进行最终判别。
        history_valid = (~history_mask).float().unsqueeze(-1)
        history_repr = (encoded[:, :history_len, :] * history_valid).sum(dim=1)
        history_repr = history_repr / history_valid.sum(dim=1).clamp_min(1.0)
        candidate_repr = encoded[:, -1, :]
        fused = torch.cat([self.norm(history_repr), self.norm(candidate_repr)], dim=1)
        logits = self.head(fused).squeeze(-1)
        return logits


class TemporalAssociationScorer:
    """
    推理阶段使用的 temporal matcher 包装器。

    训练脚本保存的是 PyTorch checkpoint；
    tracker 推理时不直接操作模型细节，而是通过这个类的 pairwise_score()
    得到一个和代价矩阵同形状的分数矩阵。
    """

    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = _normalize_device(device)

        checkpoint = torch.load(model_path, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise KeyError("Temporal matcher checkpoint must contain `state_dict`: {}".format(model_path))
        config = checkpoint.get("config", {})
        self.model = TemporalAssociationTransformer(**config)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.feat_dim = self.model.feat_dim
        self.max_history = self.model.max_history
        self.max_dt = self.model.max_dt
        self.min_history = self.model.min_history

    def _track_inputs(self, track, frame_id, image_size):
        temporal_features = getattr(track, "temporal_features", deque())
        temporal_boxes = getattr(track, "temporal_boxes", deque())
        temporal_frames = getattr(track, "temporal_frames", deque())
        # 历史观测太短时，时序模型给出的判断不稳定，直接跳过。
        if len(temporal_features) < self.min_history:
            return None

        return {
            "history_features": list(temporal_features),
            "history_boxes": list(temporal_boxes),
            "history_frames": list(temporal_frames),
            "frame_id": frame_id,
            "image_size": image_size,
        }

    def pairwise_score(self, tracks, detections, frame_id, image_size):
        """
        对多条轨迹和多个检测框批量计算时序匹配分数。

        返回 scores，形状为 [len(tracks), len(detections)]。
        分数越接近 1，表示 temporal matcher 越认为二者属于同一身份。
        分数为 0 通常表示该 pair 没有足够历史或候选特征，不参与时序增强。
        """
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)

        pair_indices = []
        history_feat = []
        history_geom = []
        history_mask = []
        candidate_feat = []
        candidate_geom = []

        # 这里不是一次只算一个 pair，而是把所有合法的 track-det 组合批量打分，
        # 再还原成与代价矩阵同形状的 score matrix。
        prepared_tracks = [self._track_inputs(track, frame_id=frame_id, image_size=image_size) for track in tracks]
        for track_idx, prepared_track in enumerate(prepared_tracks):
            if prepared_track is None:
                continue
            for det_idx, detection in enumerate(detections):
                sample = build_temporal_pair_inputs(
                    prepared_track["history_features"],
                    prepared_track["history_boxes"],
                    prepared_track["history_frames"],
                    getattr(detection, "curr_feat", None),
                    detection.tlwh,
                    candidate_frame=frame_id,
                    image_size=image_size,
                    max_history=self.max_history,
                    max_dt=self.max_dt,
                    min_history=self.min_history,
                )
                if sample is None:
                    continue
                pair_indices.append((track_idx, det_idx))
                history_feat.append(sample["history_feat"])
                history_geom.append(sample["history_geom"])
                history_mask.append(sample["history_mask"])
                candidate_feat.append(sample["candidate_feat"])
                candidate_geom.append(sample["candidate_geom"])

        # 默认分数全 0，表示“这对轨迹和检测框没有足够时序证据支撑匹配”。
        scores = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        if len(pair_indices) == 0:
            return scores

        with torch.no_grad():
            history_feat = torch.as_tensor(np.asarray(history_feat), dtype=torch.float32, device=self.device)
            history_geom = torch.as_tensor(np.asarray(history_geom), dtype=torch.float32, device=self.device)
            history_mask = torch.as_tensor(np.asarray(history_mask), dtype=torch.bool, device=self.device)
            candidate_feat = torch.as_tensor(np.asarray(candidate_feat), dtype=torch.float32, device=self.device)
            candidate_geom = torch.as_tensor(np.asarray(candidate_geom), dtype=torch.float32, device=self.device)
            logits = self.model(history_feat, history_geom, history_mask, candidate_feat, candidate_geom)
            probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

        for (track_idx, det_idx), probability in zip(pair_indices, probabilities):
            scores[track_idx, det_idx] = probability
        return scores


def get_temporal_matcher(args, device=None):
    if not getattr(args, "with_temporal_matcher", False):
        return None

    # 和 ReID 提特征器一样，这里也做缓存，避免 demo/eval 里重复读权重。
    model_path = getattr(args, "temporal_model", None)
    if model_path is None:
        raise ValueError("`--with-temporal-matcher` requires `--temporal-model`.")
    key = (model_path, str(_normalize_device(device if device is not None else getattr(args, "device", "cpu"))))
    matcher = _TEMPORAL_MATCHER_CACHE.get(key)
    if matcher is None:
        matcher = TemporalAssociationScorer(model_path=model_path, device=key[1])
        _TEMPORAL_MATCHER_CACHE[key] = matcher
    return matcher
