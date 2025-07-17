"""
Gymnasium环境包装器

移植自skrl的GymnasiumWrapper，保持AquaML的字典数据格式和风格
"""

from typing import Any, Dict, Tuple
import numpy as np
import gymnasium
import torch

from .base import Wrapper
from AquaML.data import unitCfg


class GymnasiumWrapper(Wrapper):
    """Gymnasium环境包装器
    
    移植自skrl的GymnasiumWrapper，保持AquaML的字典数据格式
    """
    
    def __init__(self, env: Any) -> None:
        """初始化Gymnasium包装器
        
        Args:
            env: Gymnasium环境实例或环境ID字符串
        """
        # 如果是字符串，创建环境
        if isinstance(env, str):
            self.env = gymnasium.make(env)
        else:
            self.env = env
        
        super().__init__(self.env)
        
        # 检查是否为向量化环境
        self._vectorized = False
        try:
            self._vectorized = self._vectorized or isinstance(self.env, gymnasium.vector.VectorEnv)
        except Exception:
            pass
        try:
            self._vectorized = self._vectorized or isinstance(self.env, gymnasium.experimental.vector.VectorEnv)
        except Exception:
            pass
        
        if self._vectorized:
            self._reset_once = True
            self._observation = None
            self._info = None
        
        # 更新环境数量
        if self._vectorized:
            self.num_envs = self.env.num_envs
        else:
            self.num_envs = 1
        
        # 重新设置配置以反映正确的环境数量
        self._setup_aquaml_configs()
    
    @property
    def observation_space(self) -> gymnasium.Space:
        """观察空间"""
        if self._vectorized:
            return self.env.single_observation_space
        return self.env.observation_space
    
    @property
    def action_space(self) -> gymnasium.Space:
        """动作空间"""
        if self._vectorized:
            return self.env.single_action_space
        return self.env.action_space
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], Any]:
        """重置环境
        
        Returns:
            (observation_dict, info): 观察字典和信息
        """
        # 处理向量化环境（向量化环境是自动重置的）
        if self._vectorized:
            if self._reset_once:
                observation, self._info = self.env.reset()
                # 转换为AquaML格式 (num_machines, num_envs, feature_dim)
                observation = np.array(observation)
                if observation.ndim == 2:  # (num_envs, feature_dim)
                    observation = np.expand_dims(observation, axis=0)  # (1, num_envs, feature_dim)
                else:
                    observation = observation.reshape(1, self.num_envs, -1)
                self._observation = {'state': observation.astype(np.float32)}
                self._reset_once = False
            return self._observation, self._info
        
        # 单一环境
        observation, info = self.env.reset()
        
        # 转换为AquaML格式 (num_machines, num_envs, feature_dim)
        observation = np.expand_dims(observation, axis=[0, 1])  # (1, 1, feature_dim)
        observation_dict = {'state': observation.astype(np.float32)}
        
        return observation_dict, info
    
    def step(self, action: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, Any
    ]:
        """执行一步
        
        Args:
            action: 动作字典，格式为 {'action': np.ndarray}
            
        Returns:
            (next_observation_dict, reward_dict, terminated, truncated, info)
        """
        # 从AquaML格式提取动作
        action_data = action['action']
        
        # 处理动作维度
        if isinstance(action_data, np.ndarray):
            if action_data.ndim > 1:
                action_data = action_data.squeeze()
            # 对于离散环境，转换为int
            if hasattr(self.action_space, 'n'):  # 离散动作空间
                action_data = int(action_data)
            # 对于需要1D数组的环境，确保不是标量
            elif action_data.ndim == 0:
                action_data = np.array([action_data])
        
        # 执行步骤
        next_observation, reward, terminated, truncated, info = self.env.step(action_data)
        
        # 处理自动重置
        if terminated or truncated:
            if not self._vectorized:
                next_observation, info = self.env.reset()
        
        # 转换观察到AquaML格式
        if self._vectorized:
            next_observation = np.array(next_observation)
            if next_observation.ndim == 2:  # (num_envs, feature_dim)
                next_observation = np.expand_dims(next_observation, axis=0)
            else:
                next_observation = next_observation.reshape(1, self.num_envs, -1)
        else:
            next_observation = np.expand_dims(next_observation, axis=[0, 1])
        
        next_observation_dict = {'state': next_observation.astype(np.float32)}
        
        # 转换奖励到AquaML格式
        if self._vectorized:
            reward = np.array(reward)
            reward = np.expand_dims(reward, axis=[0, 2])  # (1, num_envs, 1)
        else:
            reward = np.array([[[reward]]])  # (1, 1, 1)
        
        reward_dict = {'reward': reward.astype(np.float32)}
        
        # 转换终止标志到AquaML格式
        if self._vectorized:
            terminated = np.expand_dims(terminated, axis=0)  # (1, num_envs)
            truncated = np.expand_dims(truncated, axis=0)    # (1, num_envs)
        else:
            terminated = np.expand_dims(terminated, axis=[0, 1])  # (1, 1)
            truncated = np.expand_dims(truncated, axis=[0, 1])    # (1, 1)
        
        # 保存向量化环境的观察和信息
        if self._vectorized:
            self._observation = next_observation_dict
            self._info = info
        
        return next_observation_dict, reward_dict, terminated, truncated, info
    
    def render(self, *args, **kwargs) -> Any:
        """渲染环境"""
        if self._vectorized:
            return self.env.call("render", *args, **kwargs)
        return self.env.render(*args, **kwargs)
    
    def close(self) -> None:
        """关闭环境"""
        self.env.close()