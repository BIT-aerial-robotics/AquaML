"""
环境包装器工具函数

提供各种环境包装器需要的工具函数
"""

import numpy as np
import gymnasium
from typing import Any


def convert_gym_space(space: Any, squeeze_batch_dimension: bool = False) -> gymnasium.Space:
    """将gym空间转换为gymnasium空间
    
    Args:
        space: gym空间
        squeeze_batch_dimension: 是否压缩批次维度
        
    Returns:
        gymnasium空间
    """
    # 如果已经是gymnasium空间，直接返回
    if isinstance(space, gymnasium.Space):
        return space
    
    # 处理Box空间
    if hasattr(space, 'low') and hasattr(space, 'high'):
        low = np.array(space.low)
        high = np.array(space.high)
        
        if squeeze_batch_dimension and low.ndim > 1:
            low = low.squeeze(0)
            high = high.squeeze(0)
        
        return gymnasium.spaces.Box(
            low=low,
            high=high,
            dtype=getattr(space, 'dtype', np.float32)
        )
    
    # 处理Discrete空间
    elif hasattr(space, 'n'):
        return gymnasium.spaces.Discrete(space.n)
    
    # 处理MultiDiscrete空间
    elif hasattr(space, 'nvec'):
        return gymnasium.spaces.MultiDiscrete(space.nvec)
    
    # 处理MultiBinary空间
    elif hasattr(space, 'n') and hasattr(space, 'shape'):
        return gymnasium.spaces.MultiBinary(space.n)
    
    # 默认情况，尝试创建Box空间
    else:
        if hasattr(space, 'shape'):
            shape = space.shape
            if squeeze_batch_dimension and len(shape) > 1:
                shape = shape[1:]
            return gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=shape,
                dtype=np.float32
            )
        else:
            return gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float32
            )


def flatten_tensorized_space(tensorized_space: Any) -> np.ndarray:
    """扁平化张量化空间
    
    Args:
        tensorized_space: 张量化空间
        
    Returns:
        扁平化的numpy数组
    """
    if isinstance(tensorized_space, (list, tuple)):
        return np.concatenate([flatten_tensorized_space(item) for item in tensorized_space])
    elif hasattr(tensorized_space, 'numpy'):
        return tensorized_space.numpy().flatten()
    elif hasattr(tensorized_space, 'detach'):
        return tensorized_space.detach().cpu().numpy().flatten()
    else:
        return np.array(tensorized_space).flatten()


def tensorize_space(space: gymnasium.Space, data: Any, device: Any = None) -> Any:
    """将数据张量化到指定设备
    
    Args:
        space: gymnasium空间
        data: 要张量化的数据
        device: 目标设备
        
    Returns:
        张量化的数据
    """
    import torch
    
    if device is None:
        device = torch.device("cpu")
    
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return torch.tensor(data, device=device)


def unflatten_tensorized_space(space: gymnasium.Space, data: Any) -> Any:
    """反扁平化张量化空间
    
    Args:
        space: gymnasium空间
        data: 要反扁平化的数据
        
    Returns:
        反扁平化的数据
    """
    if isinstance(space, gymnasium.spaces.Box):
        return data.reshape(space.shape)
    elif isinstance(space, gymnasium.spaces.Discrete):
        return data
    elif isinstance(space, gymnasium.spaces.MultiDiscrete):
        return data
    elif isinstance(space, gymnasium.spaces.MultiBinary):
        return data
    else:
        return data