'''
AquaML环境模块

包含环境基类、具体环境实现和包装器适配器
'''
from .base_env import BaseEnv
from .gymnasium_envs import GymnasiumWrapper
from .pettingzoo_wrapper import PettingZooWrapper

# 导入环境包装器适配器
try:
    from .wrappers import (
        BaseWrapperAdapter, 
        MultiAgentWrapperAdapter,
        GymnasiumWrapperAdapter,
        auto_wrap_env
    )
    from .wrappers.isaaclab_envs import IsaacLabWrapper, IsaacLabMultiAgentWrapper
    WRAPPERS_AVAILABLE = True
except ImportError:
    WRAPPERS_AVAILABLE = False

__all__ = ["BaseEnv", "GymnasiumWrapper", "PettingZooWrapper"]

if WRAPPERS_AVAILABLE:
    __all__.extend([
        "BaseWrapperAdapter",
        "MultiAgentWrapperAdapter", 
        "GymnasiumWrapperAdapter",
        "IsaacLabWrapper",
        "IsaacLabMultiAgentWrapper",
        "auto_wrap_env"
    ])
