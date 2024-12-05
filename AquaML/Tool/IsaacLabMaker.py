from AquaML.worker import RLEnvBase
import torch
from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv

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
        
        
        