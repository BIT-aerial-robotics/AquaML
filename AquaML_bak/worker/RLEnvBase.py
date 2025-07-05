import abc
from AquaML.core.DataInfo import DataInfo
import numpy as np

class RLEnvBase(abc.ABC):
    """
    为不同的环境提供统一的接口，如vector_env, gym_env等。
    """
    
    def __init__(self, pre_constructed:bool = False):
        """
        
        创建环境部分初始化。

        Args:
            pre_constructed (bool, optional): 如果是为了获取环境部分信息，可以设置为True。默认为False。
        """
        self._env_num = 1 # 默认使用为1.VecEnv会覆盖这个值。
        
        self._step_count = 0
        
        self._obs_info:DataInfo = DataInfo() # 观察值的信息。
        self._rewards:tuple = None # 奖励的名称。
    
    @abc.abstractmethod
    def reset(self)->tuple:
        """
        重置环境。
        
        Returns:
            observation: 环境的观察值。
            info：环境的信息。
        """
        pass
    
    @abc.abstractmethod
    def step(self, action)->tuple:
        """
        执行动作。
        
        Args:
            action (dict): 动作。
        
        Returns:
            tuple: observation, reward, terminated, truncated, info
        """
        pass
    
    @abc.abstractmethod
    def close(self):
        """
        关闭环境。
        """
        pass
    
    ##############################
    # 通用调用接口
    ##############################
    def auto_step(self, action_dict:dict, max_step:int = np.inf):
        """
        自动执行动作，满足要求时对环境重制。

        Args:
            action_dict (dict): 动作的字典。
            max_step (int, optional): 最大步数。 默认为np.inf。
        """
        
        self._step_count += 1
        action = action_dict['action']
        
        next_observation, reward, terminated, truncated, info = self.step(action)
        
        # if self._step_count >= max_step-1:
        #     terminated = True
        
        if terminated or truncated:
            computing_obs, info = self.reset()
            self._step_count = 0
        else:
            computing_obs = next_observation
        
        return next_observation, computing_obs, reward, terminated, truncated, info
    
    def auto_reset(self):
        """
        自动重置环境。
        """
        computing_obs, info = self.reset()
        self._step_count = 0
        
        return computing_obs, info
        
    
    ##############################
    # 通用接口  
    ##############################
    @property
    def env_num(self):
        """
        返回环境的数量。
        """
        return self._env_num
    
    @property
    def obs_info(self)->DataInfo:
        """
        返回观察信息。
        """
        return self._obs_info
    
    @property
    def reward_info(self)->DataInfo:
        """
        返回奖励信息。
        """
        
        # 转换为DataInfo
        shapes = len(self._rewards)*((1,),)
        return DataInfo(self._rewards, shapes, (np.float32,))
        