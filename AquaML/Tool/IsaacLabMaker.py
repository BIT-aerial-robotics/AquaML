from AquaML.worker import RLEnvBase
import torch
from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv
import numpy as np

class IsaacLabMaker(RLEnvBase):
    """
    Wraps around Isaac Lab environment for AquaML library
    
    """
    
    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv):
        """
        Initializes the wrapper.
        
        不同于其他的maker，这个maker直接传递env进来，不需要再次初始化
        """
        
        super(IsaacLabMaker,self).__init__()
        
        self._env = env
        
        self._env_num = env.num_envs
        
        observation_shape_dict = env.observation_manager.group_obs_dim
        
        if not isinstance(observation_shape_dict, dict):
            observation_shape_dict = {"obs": observation_shape_dict}
            
        for key, value in observation_shape_dict.items():
            self._obs_info.add_info(
                name=key,
                shape=value,
                dtype=torch.float32,
            )
            
        # reward
        self._rewards = ('reward',)
        
    
    def reset(self)-> tuple[dict, None]:
        """
        Reset the environment
        
        Returns:
            obs: the initial observation dict.
        """
        
        obs_dict, _ = self._env.reset()
        
        return obs_dict, None
    
    def step(self, action_dict: dict)-> tuple[dict, dict, torch.Tensor, torch.Tensor, dict]:
        """
        Take a step in the environment
        
        Args:
            action_dict: the action dict
        
        Returns:
            obs: the observation dict.
            reward: the reward dict
            done: the done flag
            info: the info dict
        """
        
        action = action_dict['action']
        
        obs_dict, reward, terminated, truncated, info = self._env.step(action)
        
        reward_dict = {'reward': reward}
        
        return obs_dict, reward_dict, terminated, truncated, info
    
    def close(self):
        """
        Close the environment
        """
        
        self._env.close()
        
    def auto_reset(self):
        return self.reset()
    
    def auto_step(self, action_dict: dict,  max_step = np.inf):
        return self.step(action_dict)
    
    
        
        
        