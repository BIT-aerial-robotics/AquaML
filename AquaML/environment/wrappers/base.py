"""
基础环境包装器

移植自skrl环境包装器，保持AquaML的字典数据格式和风格
"""

import abc
from typing import Any, Dict, Tuple, Union, Mapping, Sequence
import numpy as np
import gymnasium
import torch

from AquaML.environment.base_env import BaseEnv
from AquaML.data import unitCfg


class Wrapper(BaseEnv):
    """单智能体环境包装器基类
    
    移植自skrl.envs.wrappers.torch.base.Wrapper
    保持AquaML的字典数据格式: (num_machines, num_envs, feature_dim)
    """
    
    def __init__(self, env: Any) -> None:
        """初始化环境包装器
        
        Args:
            env: 要包装的环境
        """
        super().__init__()
        
        self._env = env
        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env
        
        # 设备信息
        if hasattr(self._unwrapped, "device"):
            self._device = self._parse_device(self._unwrapped.device)
        else:
            self._device = self._parse_device(None)
        
        # 环境数量
        self.num_envs = self._unwrapped.num_envs if hasattr(self._unwrapped, "num_envs") else 1
        
        # 设置AquaML数据配置
        self._setup_aquaml_configs()
    
    def _parse_device(self, device) -> torch.device:
        """解析设备信息"""
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            return torch.device(device)
        elif isinstance(device, torch.device):
            return device
        else:
            return torch.device("cpu")
    
    def _setup_aquaml_configs(self):
        """设置AquaML数据配置"""
        # 观察空间配置
        obs_shape = self._get_space_shape(self.observation_space)
        self.observation_cfg_ = {
            'state': unitCfg(
                name='state',
                dtype=np.float32,
                single_shape=obs_shape,
            ),
        }
        
        # 动作空间配置
        action_shape = self._get_space_shape(self.action_space)
        self.action_cfg_ = {
            'action': unitCfg(
                name='action',
                dtype=np.float32,
                single_shape=action_shape,
            ),
        }
        
        # 奖励配置
        self.reward_cfg_ = {
            'reward': unitCfg(
                name='reward',
                dtype=np.float32,
                single_shape=(1,),
            ),
        }
    
    def _get_space_shape(self, space) -> Tuple[int, ...]:
        """获取空间形状"""
        if isinstance(space, gymnasium.spaces.Box):
            return space.shape
        elif isinstance(space, gymnasium.spaces.Discrete):
            return (1,)
        elif isinstance(space, gymnasium.spaces.MultiDiscrete):
            return (len(space.nvec),)
        elif isinstance(space, gymnasium.spaces.MultiBinary):
            return (space.n,)
        elif isinstance(space, gymnasium.spaces.Dict):
            # 对于字典空间，返回所有子空间维度的总和
            total_dim = 0
            for subspace in space.spaces.values():
                subshape = self._get_space_shape(subspace)
                total_dim += np.prod(subshape)
            return (total_dim,)
        else:
            return (1,)
    
    def __getattr__(self, key: str) -> Any:
        """代理属性访问到包装的环境"""
        if hasattr(self._env, key):
            return getattr(self._env, key)
        if hasattr(self._unwrapped, key):
            return getattr(self._unwrapped, key)
        raise AttributeError(
            f"Wrapped environment ({self._unwrapped.__class__.__name__}) does not have attribute '{key}'"
        )
    
    @abc.abstractmethod
    def reset(self) -> Tuple[Dict[str, np.ndarray], Any]:
        """重置环境
        
        Returns:
            (observation_dict, info): 观察字典和信息
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def step(self, action: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, Any
    ]:
        """执行一步
        
        Args:
            action: 动作字典
            
        Returns:
            (next_observation_dict, reward_dict, terminated, truncated, info)
        """
        raise NotImplementedError
    
    def state(self) -> Union[torch.Tensor, None]:
        """获取环境状态"""
        if hasattr(self._env, 'state'):
            try:
                return self._env.state()
            except:
                pass
        return None
    
    def render(self, *args, **kwargs) -> Any:
        """渲染环境"""
        if hasattr(self._env, 'render'):
            return self._env.render(*args, **kwargs)
        return None
    
    def close(self) -> None:
        """关闭环境"""
        if hasattr(self._env, 'close'):
            self._env.close()
    
    @property
    def device(self) -> torch.device:
        """环境使用的设备"""
        return self._device
    
    @property
    def num_agents(self) -> int:
        """智能体数量"""
        return self._unwrapped.num_agents if hasattr(self._unwrapped, "num_agents") else 1
    
    @property
    def state_space(self) -> Union[gymnasium.Space, None]:
        """状态空间"""
        return self._unwrapped.state_space if hasattr(self._unwrapped, "state_space") else None
    
    @property
    def observation_space(self) -> gymnasium.Space:
        """观察空间"""
        return self._unwrapped.observation_space
    
    @property
    def action_space(self) -> gymnasium.Space:
        """动作空间"""
        return self._unwrapped.action_space


class MultiAgentEnvWrapper(BaseEnv):
    """多智能体环境包装器基类
    
    移植自skrl.envs.wrappers.torch.base.MultiAgentEnvWrapper
    保持AquaML的字典数据格式
    """
    
    def __init__(self, env: Any) -> None:
        """初始化多智能体环境包装器"""
        super().__init__()
        
        self._env = env
        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env
        
        # 设备信息
        if hasattr(self._unwrapped, "device"):
            self._device = self._parse_device(self._unwrapped.device)
        else:
            self._device = self._parse_device(None)
        
        # 环境数量
        self.num_envs = self._unwrapped.num_envs if hasattr(self._unwrapped, "num_envs") else 1
        
        # 设置多智能体配置
        self._setup_multi_agent_configs()
    
    def _parse_device(self, device) -> torch.device:
        """解析设备信息"""
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            return torch.device(device)
        elif isinstance(device, torch.device):
            return device
        else:
            return torch.device("cpu")
    
    def _setup_multi_agent_configs(self):
        """设置多智能体数据配置"""
        self.observation_cfg_ = {}
        self.action_cfg_ = {}
        self.reward_cfg_ = {}
        
        # 为每个智能体设置配置
        for agent in self.possible_agents:
            # 观察配置
            obs_space = self.observation_space(agent)
            obs_shape = self._get_space_shape(obs_space)
            self.observation_cfg_[f"{agent}_state"] = unitCfg(
                name=f"{agent}_state",
                dtype=np.float32,
                single_shape=obs_shape,
            )
            
            # 动作配置
            action_space = self.action_space(agent)
            action_shape = self._get_space_shape(action_space)
            self.action_cfg_[f"{agent}_action"] = unitCfg(
                name=f"{agent}_action",
                dtype=np.float32,
                single_shape=action_shape,
            )
            
            # 奖励配置
            self.reward_cfg_[f"{agent}_reward"] = unitCfg(
                name=f"{agent}_reward",
                dtype=np.float32,
                single_shape=(1,),
            )
    
    def _get_space_shape(self, space) -> Tuple[int, ...]:
        """获取空间形状"""
        if isinstance(space, gymnasium.spaces.Box):
            return space.shape
        elif isinstance(space, gymnasium.spaces.Discrete):
            return (1,)
        elif isinstance(space, gymnasium.spaces.MultiDiscrete):
            return (len(space.nvec),)
        elif isinstance(space, gymnasium.spaces.MultiBinary):
            return (space.n,)
        elif isinstance(space, gymnasium.spaces.Dict):
            total_dim = 0
            for subspace in space.spaces.values():
                subshape = self._get_space_shape(subspace)
                total_dim += np.prod(subshape)
            return (total_dim,)
        else:
            return (1,)
    
    def __getattr__(self, key: str) -> Any:
        """代理属性访问"""
        if hasattr(self._env, key):
            return getattr(self._env, key)
        if hasattr(self._unwrapped, key):
            return getattr(self._unwrapped, key)
        raise AttributeError(
            f"Wrapped environment ({self._unwrapped.__class__.__name__}) does not have attribute '{key}'"
        )
    
    @abc.abstractmethod
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重置环境"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray], 
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, Any]
    ]:
        """执行一步"""
        raise NotImplementedError
    
    def state(self) -> Union[torch.Tensor, None]:
        """获取环境状态"""
        if hasattr(self._env, 'state'):
            try:
                return self._env.state()
            except:
                pass
        return None
    
    def render(self, *args, **kwargs) -> Any:
        """渲染环境"""
        if hasattr(self._env, 'render'):
            return self._env.render(*args, **kwargs)
        return None
    
    def close(self) -> None:
        """关闭环境"""
        if hasattr(self._env, 'close'):
            self._env.close()
    
    @property
    def device(self) -> torch.device:
        """环境使用的设备"""
        return self._device
    
    @property
    def num_agents(self) -> int:
        """当前智能体数量"""
        try:
            return self._unwrapped.num_agents
        except:
            return len(self.agents)
    
    @property
    def max_num_agents(self) -> int:
        """最大智能体数量"""
        try:
            return self._unwrapped.max_num_agents
        except:
            return len(self.possible_agents)
    
    @property
    def agents(self) -> Sequence[str]:
        """当前智能体名称列表"""
        return self._unwrapped.agents
    
    @property
    def possible_agents(self) -> Sequence[str]:
        """可能的智能体名称列表"""
        return self._unwrapped.possible_agents
    
    @property
    def state_spaces(self) -> Mapping[str, gymnasium.Space]:
        """状态空间字典"""
        space = self._unwrapped.state_space if hasattr(self._unwrapped, "state_space") else None
        return {agent: space for agent in self.possible_agents}
    
    @property
    def observation_spaces(self) -> Mapping[str, gymnasium.Space]:
        """观察空间字典"""
        return self._unwrapped.observation_spaces
    
    @property
    def action_spaces(self) -> Mapping[str, gymnasium.Space]:
        """动作空间字典"""
        return self._unwrapped.action_spaces
    
    def state_space(self, agent: str) -> gymnasium.Space:
        """获取指定智能体的状态空间"""
        return self.state_spaces[agent]
    
    def observation_space(self, agent: str) -> gymnasium.Space:
        """获取指定智能体的观察空间"""
        return self.observation_spaces[agent]
    
    def action_space(self, agent: str) -> gymnasium.Space:
        """获取指定智能体的动作空间"""
        return self.action_spaces[agent]