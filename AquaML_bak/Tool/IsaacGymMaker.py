try:
    import isaacgym
    import isaacgymenvs
except:
    print('IsaacGym is not installed!')
from numpy import inf
from AquaML.worker import RLEnvBase
import numpy as np

try:
    import torch
except:
    print('PyTorch is not installed!')
    print('current env is not support IsaacGym!')

class IsaacGymMaker(RLEnvBase):
    
    def __init__(self, env_name:str, env_num:int ,env_args:dict):
        """
        IsaacGymMaker的构造函数。
        
        Args:
            env_name (str): 环境的名称。
            env_args (dict): 环境的参数。
        """
        
        super(IsaacGymMaker, self).__init__()
        
        default_args = {
            'seed':1,
            'task':env_name,
            'num_envs':env_num,
            'sim_device':"cuda:0" if torch.cuda.is_available() else "cpu",
            'rl_device':"cuda:0" if torch.cuda.is_available() else "cpu",
            'graphics_device_id':0 if torch.cuda.is_available() else -1,
            'headless':False if torch.cuda.is_available() else True,
            'multi_gpu':False,
            'virtual_screen_capture':False,
            'force_render':False,
        }
        
        default_args.update(env_args)
        
        
        self._env = isaacgymenvs.make(**default_args)
        
        self._obs_info.add_info(
            name='obs',
            shape=self._env.observation_space.shape,
            dtype=self._env.observation_space.dtype
        )
        
        self._rewards = ('reward',)
        
    def reset(self):
        """
        重置环境。
        
        returns:
            observation(dict): 环境的观察值。
            info：环境的信息。
        """
        
        obs = self._env.reset()
        
        return obs, None
    
    def step(self, action_dict: dict):
        """
        执行动作。
        
        Args:
            action_dict (dict): 动作。
        
        Returns:
            tuple: observation, reward, terminated, truncated, info
        """
        
        action = action_dict['action']
        
        obs, reward, done, info = self._env.step(action)
        
        reward = torch.unsqueeze(torch.tensor(reward), axis=1)
        
        terminated = torch.unsqueeze(torch.tensor(done), axis=1)
        truncated = torch.zeros_like(done)
        
        reward_dict = {
            'reward': reward
        }
        
        return obs, obs ,reward_dict, terminated, truncated, info
    
    def close(self):
        """
        关闭环境。
        """
        
        self._env.close()
        
    def auto_step(self, action_dict: dict, max_step: int = np.inf):
        return self.step(action_dict)
    
    def auto_reset(self):
        return self.reset()