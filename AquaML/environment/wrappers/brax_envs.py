"""
Brax环境包装器

移植自skrl的BraxWrapper，保持AquaML的字典数据格式和风格
"""

from typing import Any, Dict, Tuple
import numpy as np
import gymnasium
import torch

from .base import Wrapper
from AquaML.data import unitCfg


class BraxWrapper(Wrapper):
    """Brax环境包装器
    
    移植自skrl的BraxWrapper，保持AquaML的字典数据格式
    """
    
    def __init__(self, env: Any) -> None:
        """初始化Brax包装器
        
        Args:
            env: Brax环境实例
        """
        # 包装Brax环境
        import brax.envs.wrappers.gym
        import brax.envs.wrappers.torch
        
        # 首先用Brax的VectorGymWrapper包装
        env = brax.envs.wrappers.gym.VectorGymWrapper(env)
        # 然后用TorchWrapper包装
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = brax.envs.wrappers.torch.TorchWrapper(env, device=device)
        
        self.env = env
        super().__init__(env)
        
        # Brax环境总是向量化的
        self.num_envs = getattr(env, 'num_envs', 1)
        
        # 重新设置配置
        self._setup_aquaml_configs()
    
    def _get_space_shape(self, space) -> Tuple[int, ...]:
        """获取空间形状，处理Brax的特殊情况"""
        # 对于Brax，需要从gym空间转换
        if hasattr(space, 'shape'):
            return space.shape
        elif hasattr(space, 'n'):
            return (1,)
        else:
            return (1,)
    
    @property
    def observation_space(self) -> gymnasium.Space:
        """观察空间"""
        # 转换gym空间到gymnasium空间
        from AquaML.environment.wrappers.utils import convert_gym_space
        return convert_gym_space(self._unwrapped.observation_space, squeeze_batch_dimension=True)
    
    @property
    def action_space(self) -> gymnasium.Space:
        """动作空间"""
        # 转换gym空间到gymnasium空间
        from AquaML.environment.wrappers.utils import convert_gym_space
        return convert_gym_space(self._unwrapped.action_space, squeeze_batch_dimension=True)
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], Any]:
        """重置环境
        
        Returns:
            (observation_dict, info): 观察字典和信息
        """
        observation = self.env.reset()
        
        # 转换为AquaML格式
        if isinstance(observation, torch.Tensor):
            observation = observation.detach().cpu().numpy()
        else:
            observation = np.array(observation)
        
        # 确保是正确的维度格式 (num_machines, num_envs, feature_dim)
        if observation.ndim == 2:  # (num_envs, feature_dim)
            observation = np.expand_dims(observation, axis=0)  # (1, num_envs, feature_dim)
        else:
            observation = observation.reshape(1, self.num_envs, -1)
        
        observation_dict = {'state': observation.astype(np.float32)}
        
        return observation_dict, {}
    
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
        
        # 转换动作维度 (num_machines, num_envs, feature_dim) -> (num_envs, feature_dim)
        if action_data.ndim == 3:
            action_data = action_data[0]  # 去掉num_machines维度
        
        # 转换为torch tensor
        if not isinstance(action_data, torch.Tensor):
            action_tensor = torch.from_numpy(action_data).to(self.device)
        else:
            action_tensor = action_data.to(self.device)
        
        # 执行步骤
        observation, reward, terminated, info = self.env.step(action_tensor)
        
        # 转换观察到AquaML格式
        if isinstance(observation, torch.Tensor):
            observation = observation.detach().cpu().numpy()
        else:
            observation = np.array(observation)
        
        if observation.ndim == 2:  # (num_envs, feature_dim)
            observation = np.expand_dims(observation, axis=0)  # (1, num_envs, feature_dim)
        else:
            observation = observation.reshape(1, self.num_envs, -1)
        
        next_observation_dict = {'state': observation.astype(np.float32)}
        
        # 转换奖励到AquaML格式
        if isinstance(reward, torch.Tensor):
            reward = reward.detach().cpu().numpy()
        else:
            reward = np.array(reward)
        
        reward = reward.reshape(1, self.num_envs, 1)  # (1, num_envs, 1)
        reward_dict = {'reward': reward.astype(np.float32)}
        
        # 转换终止标志到AquaML格式
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.detach().cpu().numpy()
        else:
            terminated = np.array(terminated)
        
        terminated = terminated.reshape(1, self.num_envs)  # (1, num_envs)
        
        # Brax没有truncated，创建零数组
        truncated = np.zeros_like(terminated)
        
        return next_observation_dict, reward_dict, terminated, truncated, info
    
    def render(self, *args, **kwargs) -> Any:
        """渲染环境"""
        try:
            frame = self.env.render(mode="rgb_array")
            
            # 使用OpenCV渲染帧
            try:
                import cv2
                cv2.imshow("env", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)
            except ImportError:
                # OpenCV不可用，只返回帧
                pass
            
            return frame
        except Exception:
            return None
    
    def close(self) -> None:
        """关闭环境"""
        # Brax环境通常不需要显式关闭
        pass