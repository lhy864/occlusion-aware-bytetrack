# -*- coding:utf-8 -*-
"""
Matching - 数据关联和匹配算法

本模块实现了多目标跟踪中的数据关联算法，包括：
1. IoU 距离计算
2. 外观特征距离计算（用于 ReID）
3. 匈牙利算法求解最优匹配
4. 马氏距离门限过滤
5. 多策略融合（IoU + 置信度、IoU + 外观特征）
"""

import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from yolox.tracker import kalman_filter
import time


def merge_matches(m1, m2, shape):
    """
    合并两阶段的匹配结果
    
    当使用两阶段匹配策略时（如 ByteTrack），需要合并两次匹配结果。
    例如：第一阶段匹配轨迹A到检测框1，第二阶段匹配检测框1到检测框2，
    最终结果就是轨迹A关联到检测框2。
    
    参数：
        m1: 第一阶段匹配结果，shape=(N,2)，每行是[行索引, 列索引]
        m2: 第二阶段匹配结果，shape=(M,2)
        shape: 矩阵形状 (O, P, Q)
              O: 第一次匹配的原始行数
              P: 第一次匹配的原始列数（也是第二次的行数）
              Q: 第二次匹配的列数
    
    返回：
        tuple: (merged_matches, unmatched_O, unmatched_Q)
               - merged_matches: 合并后的匹配对列表
               - unmatched_O: 未匹配的O索引集合
               - unmatched_Q: 未匹配的Q索引集合
    """
    # 转换为 numpy 数组
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)
    
    # 构建稀疏矩阵
    # M1: O x P 矩阵，表示第一阶段匹配
    M1 = scipy.sparse.coo_matrix(
        (np.ones(len(m1)), (m1[:, 0], m1[:, 1])),
        shape=(shape[0], shape[1])
    )
    
    # M2: P x Q 矩阵，表示第二阶段匹配
    M2 = scipy.sparse.coo_matrix(
        (np.ones(len(m2)), (m2[:, 0], m2[:, 1])),
        shape=(shape[1], shape[2])
    )
    
    # 矩阵乘法合并两阶段匹配
    # 如果 M1[a,b]=1 且 M2[b,c]=1，则 M[a,c]=1
    mask = M1 * M2
    
    # 提取合并后的匹配对
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    
    # 找出未匹配的索引
    unmatched_O = tuple(set(range(shape[0])) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(shape[2])) - set([j for i, j in match]))
    
    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    """
    将索引转换为匹配对，并过滤掉代价超过阈值的匹配
    
    用于从匈牙利算法的输出中筛选有效匹配。
    
    参数：
        cost_matrix: 代价矩阵
        indices: 匈牙利算法返回的匹配索引
        thresh: 代价阈值，超过此值的匹配将被过滤
    
    返回：
        tuple: (matches, unmatched_a, unmatched_b)
               - matches: 有效匹配对列表
               - unmatched_a: 未匹配的A索引
               - unmatched_b: 未匹配的B索引
    """
    # 获取匹配对对应的代价
    matched_cost = cost_matrix[tuple(zip(*indices))]
    
    # 创建有效匹配掩码
    matched_mask = (matched_cost <= thresh)
    
    # 筛选有效匹配
    matches = indices[matched_mask]
    
    # 找出未匹配的索引
    unmatched_a = tuple(
        set(range(cost_matrix.shape[0])) - set(matches[:, 0])
    )
    unmatched_b = tuple(
        set(range(cost_matrix.shape[1])) - set(matches[:, 1])
    )
    
    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    """
    线性分配（匈牙利算法）
    
    求解代价矩阵的最优二分匹配问题。
    使用 LAP（线性分配问题）求解器。
    
    参数：
        cost_matrix: 代价矩阵，shape=(M, N)
                    matrix[i][j] 表示将第i个轨迹匹配到第j个检测的代价
        thresh: 代价上限阈值，超过此值的匹配不会被接受
    
    返回：
        tuple: (matches, unmatched_a, unmatched_b)
               - matches: 匹配对列表，每项是[track_idx, detection_idx]
               - unmatched_a: 未匹配的行索引（元组）
               - unmatched_b: 未匹配的列索引（元组）
    
    算法复杂度：O(n^3)
    
    使用说明：
        代价矩阵中的值越小表示越匹配。
        当 matrix[i][j] = 0 表示完全匹配。
        匈牙利算法会找到一个最小代价的完美匹配（或最大代价的完美匹配，取决于实现）。
    """
    # 处理空矩阵情况
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1]))
        )
    
    matches, unmatched_a, unmatched_b = [], [], []
    
    # 使用 LAPJV 算法求解
    # cost: 最小总代价
    # x: 每行的分配列索引（-1表示未分配）
    # y: 每列的分配行索引（-1表示未分配）
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    
    # 解析匹配结果
    for ix, mx in enumerate(x):
        if mx >= 0:
            # 该行被成功分配
            matches.append([ix, mx])
    
    # 找出未匹配的行
    unmatched_a = np.where(x < 0)[0]
    
    # 找出未匹配的列
    unmatched_b = np.where(y < 0)[0]
    
    # 转换为 numpy 数组
    matches = np.asarray(matches)
    
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    计算两组边界框之间的 IoU (Intersection over Union)
    
    参数：
        atlbrs: 边界框列表A，格式为 [x1, y1, x2, y2]（tlbr格式）
                可以是 list 或 numpy.ndarray
        btlbrs: 边界框列表B，格式同上
    
    返回：
        numpy.ndarray: IoU 矩阵，shape=(len(A), len(B))
                       matrix[i][j] = IoU(A[i], B[j])
    
    IoU 计算公式：
        IoU(A, B) = |A ∩ B| / |A ∪ B|
                  = |A ∩ B| / (|A| + |B| - |A ∩ B|)
    
    底层使用 Cython 加速的 bbox_ious 函数计算。
    """
    # 初始化 IoU 矩阵
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    
    if ious.size == 0:
        return ious
    
    # 使用 Cython 加速的 IoU 计算
    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )
    
    return ious


def iou_distance(atracks, btracks):
    """
    计算轨迹与检测框之间的 IoU 距离
    
    IoU 距离定义：dist = 1 - IoU
    距离越小表示越匹配（IoU 越大）
    
    参数：
        atracks: 轨迹列表或边界框列表
        btracks: 轨迹列表或边界框列表
    
    返回：
        numpy.ndarray: 距离矩阵，shape=(len(atracks), len(btracks))
                      matrix[i][j] = 1 - IoU(track_i, detection_j)
    
    特点：
        - IoU = 1 时，距离 = 0（完全重合）
        - IoU = 0 时，距离 = 1（完全不重合）
        - 适用于跟踪场景，当目标运动平滑时效果较好
    """
    # 判断输入类型
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
       (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        # 直接是边界框数组
        atlbrs = atracks
        btlbrs = btracks
    else:
        # 是轨迹对象，提取边界框
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    
    # 计算 IoU
    _ious = ious(atlbrs, btlbrs)
    
    # 转换为距离：1 - IoU
    cost_matrix = 1 - _ious
    
    return cost_matrix


def v_iou_distance(atracks, btracks):
    """
    计算轨迹的预测边界框与检测框之间的 v-IoU 距离
    
    v-IoU (Video IoU) 考虑了目标运动，用于视频跟踪。
    使用预测的边界框（而非当前估计）进行匹配。
    
    参数：
        atracks: 轨迹列表
        btracks: 轨迹列表
    
    返回：
        numpy.ndarray: v-IoU 距离矩阵
    """
    # 判断输入类型
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
       (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        # 从轨迹获取预测边界框
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    
    # 计算 IoU
    _ious = ious(atlbrs, btlbrs)
    
    # 转换为距离
    cost_matrix = 1 - _ious
    
    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    计算轨迹与检测框之间的外观特征距离
    
    使用 ReID 外观特征进行匹配，适用于目标长时间遮挡后重新出现的场景。
    
    参数：
        tracks: 轨迹列表，每个轨迹应有 smooth_feat 属性
        detections: 检测列表，每个检测应有 curr_feat 属性
        metric: 距离度量方式
               - 'cosine': 余弦距离（默认）
               - 'euclidean': 欧氏距离
    
    返回：
        numpy.ndarray: 距离矩阵，shape=(len(tracks), len(detections))
    
    特点：
        - 余弦距离：衡量方向相似性，取值 [0, 2]
        - 适合光照变化、视角变化等场景
        - 需要额外的 ReID 特征提取网络
    """
    # 初始化距离矩阵
    cost_matrix = np.zeros(
        (len(tracks), len(detections)),
        dtype=np.float64
    )
    
    if cost_matrix.size == 0:
        return cost_matrix
    
    # 提取检测特征矩阵 (N x D)
    det_features = np.asarray(
        [track.curr_feat for track in detections],
        dtype=np.float64
    )
    
    # 提取轨迹特征矩阵 (M x D)
    track_features = np.asarray(
        [track.smooth_feat for track in tracks],
        dtype=np.float64
    )
    
    # 计算特征距离矩阵
    # 使用 cdist 计算批量距离
    # 余弦距离 = 1 - cosine_similarity
    # 取 max(0, ...) 确保距离非负
    cost_matrix = np.maximum(
        0.0,
        cdist(track_features, det_features, metric)
    )
    
    return cost_matrix


def gallery_embedding_distance(tracks, detections, metric='cosine', topk=3):
    cost_matrix = np.full((len(tracks), len(detections)), np.nan, dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    topk = max(int(topk), 1)

    for row, track in enumerate(tracks):
        if not hasattr(track, "get_gallery_bank"):
            continue
        track_bank = track.get_gallery_bank(
            include_recent=True,
            include_gallery=True,
            include_prototypes=True,
        )
        if track_bank.size == 0:
            continue

        bank_costs = np.maximum(
            0.0,
            cdist(np.asarray(track_bank, dtype=np.float64), det_features, metric),
        )
        k = min(topk, bank_costs.shape[0])
        if k == 1:
            cost_matrix[row] = np.min(bank_costs, axis=0)
        else:
            topk_costs = np.partition(bank_costs, kth=k - 1, axis=0)[:k]
            cost_matrix[row] = np.mean(topk_costs, axis=0)

    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """
    使用马氏距离门限过滤代价矩阵
    
    基于卡尔曼滤波的马氏距离，过滤掉物理上不可能的匹配。
    马氏距离考虑了状态估计的不确定性。
    
    参数：
        kf: 卡尔曼滤波器实例
        cost_matrix: 原始代价矩阵
        tracks: 轨迹列表
        detections: 检测列表
        only_position: 是否只考虑位置（x, y）
    
    返回：
        numpy.ndarray: 过滤后的代价矩阵
                      不可能的匹配被设为无穷大 (inf)
    
    门限原理：
        马氏距离服从卡方分布，95%分位点作为门限。
        4维情况下，门限约为 9.49（来自 chi2inv95[4]）
        超过门限的匹配被认为不太可能，代价设为无穷大。
    """
    if cost_matrix.size == 0:
        return cost_matrix
    
    # 确定门限维度
    gating_dim = 2 if only_position else 4
    
    # 获取卡方分布 95% 分位点作为门限
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    
    # 提取检测框的 xyah 格式
    measurements = np.asarray([det.to_xyah() for det in detections])
    
    # 对每一行（每个轨迹）计算马氏距离
    for row, track in enumerate(tracks):
        # 计算该轨迹状态到所有检测的马氏距离
        gating_dist = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        
        # 超过门限的匹配设为无穷大（不可能的匹配）
        cost_matrix[row, gating_dist > gating_threshold] = np.inf
    
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections,
                only_position=False, lambda_=0.98):
    """
    融合运动信息到代价矩阵
    
    将卡尔曼滤波的马氏距离与原始代价（IoU距离）加权融合。
    这种方法比单纯使用门限更柔和。
    
    参数：
        kf: 卡尔曼滤波器实例
        cost_matrix: 原始代价矩阵（如 IoU 距离）
        tracks: 轨迹列表
        detections: 检测列表
        only_position: 是否只考虑位置
        lambda_: 加权系数，融合公式：
                 cost_fused = λ * cost_matrix + (1-λ) * gating_distance
                 
                 λ=0.98 表示主要信任原始代价
                 λ=0.5 表示平等对待两种信息
    
    返回：
        numpy.ndarray: 融合后的代价矩阵
    """
    if cost_matrix.size == 0:
        return cost_matrix
    
    # 确定门限维度
    gating_dim = 2 if only_position else 4
    
    # 获取门限
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    
    # 提取检测框
    measurements = np.asarray([det.to_xyah() for det in detections])
    
    # 融合代价
    for row, track in enumerate(tracks):
        # 计算马氏距离
        gating_dist = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha'
        )
        
        # 超过门限设为无穷大
        cost_matrix[row, gating_dist > gating_threshold] = np.inf
        
        # 加权融合
        # 原始代价权重为 lambda，马氏距离权重为 (1-lambda)
        cost_matrix[row] = lambda_ * cost_matrix[row] + \
                          (1 - lambda_) * gating_dist
    
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    """
    融合外观特征相似度到 IoU 代价矩阵
    
    用于 DeepSORT 等使用外观特征的跟踪器。
    
    参数：
        cost_matrix: 原始代价矩阵（外观特征距离）
        tracks: 轨迹列表
        detections: 检测列表
    
    返回：
        numpy.ndarray: 融合后的代价矩阵
    
    融合公式：
        iou_sim = 1 - iou_dist
        fuse_sim = reid_sim * (1 + iou_sim) / 2
        fuse_cost = 1 - fuse_sim
    """
    if cost_matrix.size == 0:
        return cost_matrix
    
    # ReID 相似度
    reid_sim = 1 - cost_matrix
    
    # IoU 相似度
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    
    # 融合相似度
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    
    # 考虑检测置信度
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(
        cost_matrix.shape[0], axis=0
    )
    # fuse_sim = fuse_sim * (1 + det_scores) / 2
    
    # 转换回代价
    fuse_cost = 1 - fuse_sim
    
    return fuse_cost


def fuse_score(cost_matrix, detections):
    """
    将检测置信度融合到代价矩阵
    
    这是 ByteTrack 中使用的融合策略。
    高置信度检测在匹配时具有更高的优先级。
    
    参数：
        cost_matrix: 原始代价矩阵（如 IoU 距离 = 1 - IoU）
        detections: 检测列表，每个检测有 score 属性
    
    返回：
        numpy.ndarray: 融合后的代价矩阵
    
    融合公式：
        iou_sim = 1 - cost_matrix = IoU
        fuse_sim = iou_sim × det_score
        fuse_cost = 1 - fuse_sim
                      = 1 - IoU × det_score
    
    示例：
        - 高置信度(0.9) + 高IoU(0.9): fuse_sim = 0.81, cost = 0.19
        - 低置信度(0.2) + 高IoU(0.9): fuse_sim = 0.18, cost = 0.82
        - 高置信度(0.9) + 低IoU(0.3): fuse_sim = 0.27, cost = 0.73
    
    效果：
        高置信度检测更容易被匹配上
        低置信度检测的匹配代价更高
    """
    if cost_matrix.size == 0:
        return cost_matrix
    
    # IoU 相似度：1 - 原始代价 = IoU
    iou_sim = 1 - cost_matrix
    
    # 提取检测置信度
    det_scores = np.array([det.score for det in detections])
    
    # 扩展维度以匹配代价矩阵
    det_scores = np.expand_dims(det_scores, axis=0).repeat(
        cost_matrix.shape[0], axis=0
    )
    
    # 融合：相似度 × 置信度
    fuse_sim = iou_sim * det_scores
    
    # 转换回代价
    fuse_cost = 1 - fuse_sim
    
    return fuse_cost
