'''
用于统一Gym环境接口。
'''
try:
    import gymnasium as gym
except:
    import gym
from AquaML.worker import RLEnvBase
import numpy as np
# from AquaML.core import DataInfo

class GymMaker(RLEnvBase):
    
    def __init__(self, env_name:str, env_args:dict):
        """
        GymnasiumMaker的构造函数。
        
        Args:
            env_name (str): 环境的名称。
            env_args (dict): 环境的参数。
        """
        
        super(GymMaker, self).__init__()
        
        self._env = gym.make(env_name, **env_args)
        
        self.action_bound = self._env.action_space.high
        
        # 获取该环境的观察值空间
        self._obs_info.add_info(
            name='obs',
            shape=self._env.observation_space.shape,
            dtype=self._env.observation_space.dtype
        )
        
        self._rewards = ('reward',)
        
    def reset(self):
        """
        重置环境。
        
        Returns:
            observation: 环境的观察值。
            info：环境的信息。
        """
        
        obs, info = self._env.reset()
        
        obs_dict = {
            'obs': obs
        }
        
        return obs_dict, info
    
    def step(self, action):
        """
        执行动作。
        
        Args:
            action (dict): 动作。
        
        Returns:
            tuple: observation, reward, terminated, truncated, info
        """
        
        obs, reward, terminated, info = self._env.step(action*self.action_bound)
        
        truncated = np.zeros_like(terminated)
        
        obs_dict = {
            'obs': obs
        }
        
        reward_dict = {
            'reward': reward,
            # 'indicator': reward
        }
        
        return obs_dict, reward_dict, terminated, truncated, info
    
    def close(self):
        """
        关闭环境。
        """
        
        self._env.close()
        
        
       