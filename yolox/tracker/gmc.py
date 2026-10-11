#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
轻量级相机运动补偿（Global Motion Compensation, GMC）。

当视频存在抖动或机位移动时，单纯依赖上一帧轨迹位置做预测会偏移。
这里用稀疏光流估计整幅图像的仿射变换，再把这份全局运动补偿到每条轨迹上。
"""

import cv2
import numpy as np


class SparseOptFlowGMC:
    """
    稀疏光流相机运动补偿器。

    输入连续两帧图像，估计背景整体运动。
    输出 2x3 仿射矩阵，用于把上一帧轨迹预测位置补偿到当前帧坐标系。

    这个模块解决的是“相机在动”带来的整体位移问题，
    不是解决行人本身的外观重识别问题。
    """

    def __init__(self, downscale=2, max_corners=1000, quality_level=0.01, min_distance=8, min_points=6):
        # 这几个参数都围绕“从上一帧选角点 -> LK 光流跟踪 -> 估计全局仿射”展开。
        self.downscale = max(int(downscale), 1)
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.min_points = min_points
        self.prev_gray = None

    @staticmethod
    def _identity():
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    def reset(self):
        self.prev_gray = None

    def _prepare_gray(self, image):
        # GMC 只需要亮度信息，先转灰度；必要时再下采样提速。
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            width = max(gray.shape[1] // self.downscale, 1)
            height = max(gray.shape[0] // self.downscale, 1)
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        return gray

    def estimate(self, image):
        """
        估计当前帧相对于上一帧的全局仿射变换。

        算法步骤：
        1. 当前帧转灰度，并按 downscale 下采样。
        2. 在上一帧灰度图上提取角点。
        3. 用 LK 光流追踪这些角点到当前帧。
        4. 用 RANSAC 估计仿射变换，过滤不可靠点。
        5. 如果估计失败，返回单位变换，保证 tracker 不崩。
        """
        if image is None:
            return self._identity()

        curr_gray = self._prepare_gray(image)
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return self._identity()

        # 先在上一帧上找可跟踪角点，再通过 LK 光流找到当前帧对应位置。
        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=3,
        )
        if prev_pts is None or len(prev_pts) < self.min_points:
            self.prev_gray = curr_gray
            return self._identity()

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, prev_pts, None)
        if curr_pts is None or status is None:
            self.prev_gray = curr_gray
            return self._identity()

        status = status.reshape(-1).astype(bool)
        prev_pts = prev_pts.reshape(-1, 2)[status]
        curr_pts = curr_pts.reshape(-1, 2)[status]
        if len(prev_pts) < self.min_points or len(curr_pts) < self.min_points:
            self.prev_gray = curr_gray
            return self._identity()

        # 用 RANSAC 拟合“上一帧 -> 当前帧”的全局仿射变换。
        warp, _ = cv2.estimateAffinePartial2D(
            prev_pts,
            curr_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
        self.prev_gray = curr_gray
        if warp is None:
            return self._identity()

        warp = warp.astype(np.float32)
        # 如果前面做过下采样，这里要把平移量映射回原图尺度。
        if self.downscale > 1:
            warp[:, 2] *= float(self.downscale)
        return warp
