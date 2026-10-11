# vim: expandtab:ts=4:sw=4
"""
BaseTrack - 多目标跟踪的轨迹基类

本模块定义了所有跟踪轨迹的基类和数据结构，
包括轨迹状态枚举和基本的状态管理方法。
"""

import numpy as np
from collections import OrderedDict


class TrackState:
    """
    轨迹状态枚举类
    
    定义了跟踪轨迹的四种可能状态：
    - New: 新创建的轨迹，尚未确认
    - Tracked: 正在被跟踪的活跃轨迹
    - Lost: 暂时丢失的轨迹（可能在后续帧中重新匹配）
    - Removed: 已永久移除的轨迹
    """
    New = 0       # 新轨迹，等待激活
    Tracked = 1   # 正在跟踪
    Lost = 2      # 丢失（暂时未匹配到）
    Removed = 3   # 已移除（永久删除）


class BaseTrack:
    """
    轨迹基类
    
    所有目标轨迹的抽象基类，定义了轨迹的基本属性和接口。
    子类（如 STrack）需要实现 activate(), predict(), update() 等方法。
    
    类属性（所有实例共享）：
        _count: 全局轨迹计数器，用于分配唯一ID
    
    实例属性：
        track_id: 轨迹的唯一标识符
        is_activated: 轨迹是否已激活（首次成功匹配）
        state: 当前轨迹状态（TrackState枚举）
        history: 历史位置的有序字典
        features: 外观特征向量列表（用于ReID）
        curr_feature: 当前帧的外观特征
        score: 检测置信度分数
        start_frame: 轨迹起始帧号
        frame_id: 最近更新帧号
        time_since_update: 自上次更新以来的帧数
        location: 多相机场景中的位置坐标
    """
    
    # 类变量：全局轨迹计数，从1开始递增
    _count = 0
    
    # 轨迹唯一ID，初始化为0（未分配）
    track_id = 0
    
    # 轨迹是否已激活（新轨迹首次匹配成功后激活）
    is_activated = False
    
    # 当前轨迹状态，初始化为NEW状态
    state = TrackState.New
    
    # 历史轨迹记录，按帧号存储边界框等信息
    history = OrderedDict()
    
    # 外观特征向量列表，用于计算相似度
    features = []
    
    # 当前帧的外观特征向量
    curr_feature = None
    
    # 检测置信度分数
    score = 0
    
    # 轨迹起始帧号
    start_frame = 0
    
    # 最近更新的帧号
    frame_id = 0
    
    # 自上次成功更新以来经过的帧数
    time_since_update = 0
    
    # 多相机场景中的位置坐标（x, y），默认为无穷远
    location = (np.inf, np.inf)
    
    @property
    def end_frame(self):
        """
        获取轨迹结束帧号
        
        返回值:
            当前帧号（frame_id）
        """
        return self.frame_id
    
    @staticmethod
    def next_id():
        """
        生成下一个唯一的轨迹ID
        
        这是一个静态方法，可以直接通过类名调用。
        每次调用时全局计数器加1，确保每个轨迹都有唯一ID。
        
        返回值:
            int: 下一个可用的轨迹ID
        """
        BaseTrack._count += 1
        return BaseTrack._count
    
    def activate(self, *args):
        """
        激活新轨迹的抽象方法
        
        当检测框首次匹配到新轨迹时调用此方法。
        子类需要实现具体的激活逻辑，如初始化卡尔曼滤波器等。
        
        参数:
            *args: 可变参数，子类可自定义参数列表
        """
        raise NotImplementedError
    
    def predict(self):
        """
        预测轨迹下一位置的抽象方法
        
        使用运动模型（如卡尔曼滤波）预测目标在下一帧的位置。
        子类需要实现具体的预测逻辑。
        """
        raise NotImplementedError
    
    def update(self, *args, **kwargs):
        """
        更新轨迹状态的抽象方法
        
        当轨迹成功匹配到检测框时调用此方法。
        子类需要实现具体的更新逻辑，如更新卡尔曼滤波器状态。
        
        参数:
            *args: 可变位置参数
            **kwargs: 可变关键字参数
        """
        raise NotImplementedError
    
    def mark_lost(self):
        """
        将轨迹标记为丢失状态
        
        当轨迹在某帧中未能匹配到任何检测框时调用。
        丢失的轨迹会在一段时间内保留，以便在目标重新出现时恢复跟踪。
        """
        self.state = TrackState.Lost
    
    def mark_removed(self):
        """
        将轨迹标记为已移除状态
        
        当轨迹丢失超过允许的最大帧数时调用。
        已移除的轨迹将从跟踪器中彻底删除。
        """
        self.state = TrackState.Removed
