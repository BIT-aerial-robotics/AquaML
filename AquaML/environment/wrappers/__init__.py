"""
AquaML环境包装器模块

移植自skrl环境包装器，保持AquaML的字典数据格式和风格
支持多种环境类型的自动适配
"""

import re
from typing import Any, Union
from loguru import logger

from .base import Wrapper, MultiAgentEnvWrapper
from .gymnasium_envs import GymnasiumWrapper
from .brax_envs import BraxWrapper
from .isaaclab_envs import IsaacLabWrapper, IsaacLabMultiAgentWrapper

__all__ = ["wrap_env", "Wrapper", "MultiAgentEnvWrapper", "GymnasiumWrapper", "BraxWrapper", "IsaacLabWrapper", "IsaacLabMultiAgentWrapper"]


def wrap_env(env: Any, wrapper: str = "auto", verbose: bool = True) -> Union[Wrapper, MultiAgentEnvWrapper]:
    """包装环境以使用统一接口
    
    移植自skrl的wrap_env函数，保持AquaML的字典数据格式
    
    Args:
        env: 要包装的环境
        wrapper: 包装器类型，默认为"auto"自动检测
                支持的包装器类型：
                - "auto": 自动检测
                - "gymnasium": Gymnasium环境
                - "brax": Brax环境  
                - "isaaclab": Isaac Lab单智能体环境
                - "isaaclab-multi-agent": Isaac Lab多智能体环境
        verbose: 是否打印详细信息
        
    Returns:
        包装后的环境
        
    Raises:
        ValueError: 未知的包装器类型
    """
    
    def _get_wrapper_name(env, verbose):
        """自动检测环境类型"""
        def _in(values, container):
            if isinstance(values, str):
                values = [values]
            for item in container:
                for value in values:
                    if value in item or re.match(value, item):
                        return True
            return False
        
        # 获取类层次结构
        base_classes = [str(base).replace("<class '", "").replace("'>", "") for base in env.__class__.__bases__]
        try:
            base_classes += [
                str(base).replace("<class '", "").replace("'>", "") for base in env.unwrapped.__class__.__bases__
            ]
        except:
            pass
        base_classes = sorted(list(set(base_classes)))
        
        if verbose:
            logger.info(f"Environment wrapper: 'auto' (class: {', '.join(base_classes)})")
        
        # 检测环境类型
        if _in(["omni.isaac.lab.*", "isaaclab.*"], base_classes):
            return "isaaclab-*"
        elif _in("brax.envs.*", base_classes):
            return "brax"
        elif _in("gymnasium.*", base_classes):
            return "gymnasium"
        elif _in("gym.*", base_classes):
            return "gymnasium"  # 使用gymnasium包装器兼容gym
        else:
            return "gymnasium"  # 默认使用gymnasium包装器
    
    # 自动检测包装器类型
    if wrapper == "auto":
        wrapper = _get_wrapper_name(env, verbose)
    
    # 根据类型选择包装器
    if wrapper == "gymnasium":
        if verbose:
            logger.info("Environment wrapper: Gymnasium")
        return GymnasiumWrapper(env)
    elif wrapper == "brax":
        if verbose:
            logger.info("Environment wrapper: Brax")
        return BraxWrapper(env)
    elif isinstance(wrapper, str) and wrapper.startswith("isaaclab"):
        # 检测单智能体还是多智能体
        if wrapper == "isaaclab-multi-agent":
            env_type = "multi-agent"
            env_wrapper = IsaacLabMultiAgentWrapper
        elif wrapper == "isaaclab-single-agent":
            env_type = "single-agent"
            env_wrapper = IsaacLabWrapper
        else:
            # 自动检测
            env_type = "single-agent"
            env_wrapper = IsaacLabWrapper
            if hasattr(env, 'possible_agents') or hasattr(env.unwrapped, 'possible_agents'):
                env_type = "multi-agent"
                env_wrapper = IsaacLabMultiAgentWrapper
        
        if verbose:
            logger.info(f"Environment wrapper: Isaac Lab ({env_type})")
        return env_wrapper(env)
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper}")


# 便捷函数
def create_gymnasium_env(env_id: str, **kwargs) -> GymnasiumWrapper:
    """创建Gymnasium环境"""
    import gymnasium as gym
    env = gym.make(env_id, **kwargs)
    return GymnasiumWrapper(env)


def create_brax_env(env_name: str, **kwargs) -> BraxWrapper:
    """创建Brax环境"""
    import brax.envs
    env = brax.envs.get_environment(env_name, **kwargs)
    return BraxWrapper(env)


def create_isaaclab_env(env_cfg, multi_agent: bool = False, **kwargs) -> Union[IsaacLabWrapper, IsaacLabMultiAgentWrapper]:
    """创建Isaac Lab环境"""
    # 这里需要根据Isaac Lab的实际API来实现
    # 暂时返回一个占位符
    if multi_agent:
        return IsaacLabMultiAgentWrapper(env_cfg)
    else:
        return IsaacLabWrapper(env_cfg)