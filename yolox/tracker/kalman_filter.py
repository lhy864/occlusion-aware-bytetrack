# vim: expandtab:ts=4:sw=4
"""
KalmanFilter - 卡尔曼滤波器实现

本模块实现了用于目标运动跟踪的卡尔曼滤波器。
卡尔曼滤波是一种递归滤波器，用于估计动态系统的状态。

在 ByteTrack 中，卡尔曼滤波器用于：
1. 预测目标在下一帧的位置
2. 根据观测（检测框）校正预测
3. 处理测量噪声，提供平滑的目标轨迹
"""

import numpy as np
import scipy.linalg


# ==========================================
# 卡方分布表：用于马氏距离门限
# ==========================================
# 0.95 分位点的卡方分布值
# 用于马氏距离门限（Mahalanobis Gating）
# key: 自由度(维度数), value: 0.95分位点的卡方值
chi2inv95 = {
    1: 3.8415,   # 1维：95%置信度门限
    2: 5.9915,   # 2维：仅位置(x,y)
    3: 7.8147,   # 3维
    4: 9.4877,   # 4维：位置+宽高(x,y,a,h)
    5: 11.070,   # 5维
    6: 12.592,   # 6维
    7: 14.067,   # 7维
    8: 15.507,   # 8维：完整状态向量
    9: 16.919    # 9维
}


class KalmanFilter:
    """
    卡尔曼滤波器 - 用于跟踪图像空间中的边界框
    
    本实现使用8维状态空间：
        [x, y, a, h, vx, vy, va, vh]
    
    其中：
        x, y: 边界框中心点坐标
        a: 宽高比 (width / height)
        h: 边界框高度
        vx, vy, va, vh: 对应的速度分量
    
    运动模型假设目标做恒速运动，边界框位置为状态空间
    的直接观测（线性观测模型）。
    
    卡尔曼滤波的五个核心公式：
    
    1. 状态预测：
       X_pred = F * X
       P_pred = F * P * F^T + Q
    
    2. 观测预测：
       Z_pred = H * X_pred
    
    3. 卡尔曼增益：
       K = P_pred * H^T * (H * P_pred * H^T + R)^-1
    
    4. 状态更新：
       X_updated = X_pred + K * (Z - Z_pred)
    
    5. 协方差更新：
       P_updated = (I - K * H) * P_pred
    
    其中：
        X: 状态向量
        P: 状态协方差矩阵
        F: 状态转移矩阵
        H: 观测矩阵
        Q: 过程噪声协方差矩阵
        R: 观测噪声协方差矩阵
        K: 卡尔曼增益
        Z: 实际观测值
    """
    
    def __init__(self):
        """
        初始化卡尔曼滤波器
        
        创建状态转移矩阵和观测矩阵，并设置噪声权重。
        """
        ndim, dt = 4, 1.  # ndim=4个位置/观测维度, dt=时间步长=1帧
        
        # ==========================================
        # 状态转移矩阵 F (8x8)
        # 用于预测下一时刻的状态
        # 
        # 形式：
        # | I   dt*I |
        # | 0    I   |
        # 
        # 其中I是4x4单位矩阵，表示恒速模型
        # ==========================================
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            # 对角线上设置时间步长，连接位置和速度
            self._motion_mat[i, ndim + i] = dt
        
        # ==========================================
        # 观测矩阵 H (4x8)
        # 从状态空间中提取可观测的部分
        # 即 [x, y, a, h]
        # ==========================================
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # ==========================================
        # 噪声权重
        # 控制模型不确定性
        # ==========================================
        # 位置分量的标准差权重（相对于高度h）
        self._std_weight_position = 1. / 20
        
        # 速度分量的标准差权重（相对于高度h）
        self._std_weight_velocity = 1. / 160
    
    def initiate(self, measurement):
        """
        为新检测创建跟踪轨迹
        
        根据初始观测（边界框）初始化状态向量和协方差矩阵。
        
        参数：
            measurement: 初始观测值 [x, y, a, h]
                       - x, y: 边界框中心点
                       - a: 宽高比 (width/height)
                       - h: 边界框高度
        
        返回：
            tuple: (mean, covariance)
                   - mean: 8维状态均值向量 [x, y, a, h, vx, vy, va, vh]
                   - covariance: 8x8 状态协方差矩阵
        """
        # 初始化位置分量为观测值
        mean_pos = measurement
        
        # 初始化速度分量为零
        mean_vel = np.zeros_like(mean_pos)
        
        # 拼接为完整状态向量
        mean = np.r_[mean_pos, mean_vel]
        
        # 根据观测不确定性初始化协方差
        # 高度h较大的目标，不确定性也较大
        std = [
            # 位置分量的不确定性（2倍权重×高度）
            2 * self._std_weight_position * measurement[3],  # x
            2 * self._std_weight_position * measurement[3],  # y
            1e-2,                                            # a (宽高比变化小)
            2 * self._std_weight_position * measurement[3],  # h
            
            # 速度分量的不确定性（10倍权重×高度）
            10 * self._std_weight_velocity * measurement[3], # vx
            10 * self._std_weight_velocity * measurement[3], # vy
            1e-5,                                            # va (宽高比变化率极小)
            10 * self._std_weight_velocity * measurement[3]  # vh
        ]
        
        # 创建对角协方差矩阵
        covariance = np.diag(np.square(std))
        
        return mean, covariance
    
    def predict(self, mean, covariance):
        """
        卡尔曼滤波预测步骤
        
        根据当前状态估计预测下一时刻的状态。
        
        参数：
            mean: 8维状态均值向量
            covariance: 8x8 状态协方差矩阵
        
        返回：
            tuple: (predicted_mean, predicted_covariance)
                   预测后的状态均值和协方差
        """
        # ==========================================
        # 计算过程噪声协方差 Q
        # 噪声与目标高度成正比
        # 速度分量噪声比位置分量小
        # ==========================================
        std_pos = [
            self._std_weight_position * mean[3],  # x方向位置
            self._std_weight_position * mean[3],  # y方向位置
            1e-2,                                  # 宽高比
            self._std_weight_position * mean[3]    # 高度
        ]
        
        std_vel = [
            self._std_weight_velocity * mean[3],  # x方向速度
            self._std_weight_velocity * mean[3],  # y方向速度
            1e-5,                                 # 速度变化率（极小）
            self._std_weight_velocity * mean[3]   # 高度变化率
        ]
        
        # 构建噪声协方差矩阵
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        # ==========================================
        # 状态预测
        # X_pred = F * X
        # ==========================================
        # 使用状态转移矩阵预测下一状态
        mean = np.dot(mean, self._motion_mat.T)
        
        # ==========================================
        # 协方差预测
        # P_pred = F * P * F^T + Q
        # ==========================================
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T
        )) + motion_cov
        
        return mean, covariance
    
    def project(self, mean, covariance):
        """
        将状态分布投影到观测空间
        
        计算预测状态对应的观测分布（均值和协方差）。
        用于计算卡尔曼增益和更新步骤。
        
        参数：
            mean: 8维状态均值向量
            covariance: 8x8 状态协方差矩阵
        
        返回：
            tuple: (projected_mean, projected_covariance)
                   投影到观测空间后的均值和协方差
        """
        # 观测噪声（注入一些不确定性）
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,  # 宽高比观测噪声较大
            self._std_weight_position * mean[3]
        ]
        
        innovation_cov = np.diag(np.square(std))
        
        # 投影到观测空间
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        ))
        
        # 加上观测噪声
        return mean, covariance + innovation_cov
    
    def multi_predict(self, mean, covariance):
        """
        批量预测多目标的状态（向量化版本）
        
        一次性预测N个目标的状态，比逐个调用 predict() 快很多。
        用于 STrack.multi_predict() 批量预测。
        
        参数：
            mean: Nx8 矩阵，N个目标的状态均值
            covariance: Nx8x8 张量，N个目标的协方差矩阵
        
        返回：
            tuple: (predicted_mean, predicted_covariance)
                   批量预测后的状态均值和协方差
        """
        # 计算位置分量的过程噪声
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]
        ]
        
        # 计算速度分量的过程噪声
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]
        ]
        
        # 构建批量的噪声协方差矩阵
        sqr = np.square(np.r_[std_pos, std_vel]).T
        
        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)
        
        # 批量状态预测
        mean = np.dot(mean, self._motion_mat.T)
        
        # 批量协方差预测
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        
        return mean, covariance
    
    def update(self, mean, covariance, measurement):
        """
        卡尔曼滤波更新步骤
        
        根据新的观测（检测框）校正状态估计。
        
        参数：
            mean: 8维预测状态均值
            covariance: 8x8 预测状态协方差矩阵
            measurement: 4维观测向量 [x, y, a, h]
                       从检测框提取的实际观测值
        
        返回：
            tuple: (updated_mean, updated_covariance)
                   校正后的状态均值和协方差
        """
        # ==========================================
        # Step 1: 投影预测状态到观测空间
        # ==========================================
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # ==========================================
        # Step 2: 计算卡尔曼增益
        # K = P_pred * H^T * (H * P_pred * H^T + R)^-1
        # 
        # 使用 Cholesky 分解提高数值稳定性
        # ==========================================
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T
        
        # ==========================================
        # Step 3: 计算观测残差（ innovation ）
        # innovation = Z_actual - Z_predicted
        # ==========================================
        innovation = measurement - projected_mean
        
        # ==========================================
        # Step 4: 更新状态
        # X_updated = X_pred + K * innovation
        # ==========================================
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        
        # ==========================================
        # Step 5: 更新协方差
        # P_updated = (I - K * H) * P_pred
        # ==========================================
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T
        ))
        
        return new_mean, new_covariance
    
    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """
        计算状态分布与观测之间的门限距离
        
        用于判断观测是否可能来自该状态（数据关联前的过滤）。
        常用的门限是卡方分布的0.95分位点。
        
        参数：
            mean: 8维状态均值向量
            covariance: 8x8 状态协方差矩阵
            measurements: Nx4 矩阵，N个候选观测
            only_position: 是否只考虑位置（x, y）
                          True时门限用2维卡方分布
            metric: 距离度量方式
                   - 'maha': 马氏距离（Mahalanobis）
                   - 'gaussian': 欧氏距离
        
        返回：
            ndarray: N个观测与状态的马氏距离平方
                   距离越小，观测越可能来自该状态
        """
        # 投影状态到观测空间
        mean, covariance = self.project(mean, covariance)
        
        # 如果只考虑位置，提取位置分量
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        
        # 计算残差
        d = measurements - mean
        
        if metric == 'gaussian':
            # 高斯距离（欧氏距离平方）
            return np.sum(d * d, axis=1)
        
        elif metric == 'maha':
            # 马氏距离 - 考虑协方差（方向性）
            # d^T * Σ^(-1) * d
            
            # Cholesky 分解
            cholesky_factor = np.linalg.cholesky(covariance)
            
            # 求解线性系统
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True,
                check_finite=False, overwrite_b=True
            )
            
            # 计算马氏距离平方
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        
        else:
            raise ValueError('invalid distance metric')
