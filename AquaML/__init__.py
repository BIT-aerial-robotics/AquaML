from .core import AquaMLCoordinator, ComponentRegistry, LifecycleManager

from AquaML.enum import *

# 创建协调器实例
coordinator = AquaMLCoordinator()

# 导出主要组件
__all__ = [
    'AquaMLCoordinator',
    'ComponentRegistry', 
    'LifecycleManager',
    'coordinator'
]
