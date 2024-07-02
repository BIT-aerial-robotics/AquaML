from AquaML.worker import RLEnvBase as EnvBase
from AquaML import logger, aqua_tool
# import AquaML
import numpy as np

class VecCollector:
    """
    Collect data from sub envs.
    """

    def __init__(self):
        self.next_obs_dict = {}
        self.compute_obs_dict = {}
        self.reward_dict = {}
        self.terminals = []
        self.truncateds = []
        # self.masks = []
    
    def append(self, 
               next_obs, 
               reward, 
               computing_obs, 
               terminal, 
               truncated):
        """
        
        将数据添加到collector中，并将数据格式变换为（1，dims）。

        Args:
            next_obs (dict): 下一个观察值，其数据格式为：（dims, ）。
            reward (dict): 奖励，其数据格式为：（dims, ）。
            computing_obs (dict): 计算观察值，其数据格式为：（dims, ）。
            terminal (bool): 是否终止。
            truncated (bool): 是否截断。
        """
        
        for name in next_obs.keys():
            if name not in self.next_obs_dict.keys():
                self.next_obs_dict[name] = []
            self.next_obs_dict[name].append(np.expand_dims(next_obs[name], axis=0))
        
        for name in reward.keys():
            if name not in self.reward_dict.keys():
                self.reward_dict[name] = []
            self.reward_dict[name].append(reward[name])
            
        for name in computing_obs.keys():
            if name not in self.compute_obs_dict.keys():
                self.compute_obs_dict[name] = []
            self.compute_obs_dict[name].append(np.expand_dims(computing_obs[name], axis=0))
        
        self.terminals.append(terminal)
        self.truncateds.append(truncated)
    
    def reset_append(self, observation):
        """
        Append data to collector.

        Args:
            observation (dict): observation.
        """
        for name in observation.keys():
            if name not in self.next_obs_dict.keys():
                self.next_obs_dict[name] = []
            self.next_obs_dict[name].append(np.expand_dims(observation[name], axis=0))
        

    def get_reset_data(self):
        """
        获取reset的数据。
        返回的数据shape为（num_envs, dims）。
        
        Returns:
            dict: 数据。
        """
        
        obs = {}
        
        for name in self.next_obs_dict.keys():
            obs[name] = np.concatenate(self.next_obs_dict[name], axis=0)
        
        return obs
    
    def get_data(self):
        """
        获取数据。
        返回的数据shape为（num_envs, step_num, dims）。
        
        Returns:
            dict: 数据。
        """
        
        next_obs = {}
        computing_obs = {}
        rewards = {}
        # rewards = {}
        
        # TODO:检查数据顺序是否正确
        
        for name in self.next_obs_dict.keys():
            next_obs[name] = np.vstack(self.next_obs_dict[name])
        
        for name in self.compute_obs_dict.keys():
            computing_obs[name] = np.vstack(self.compute_obs_dict[name])
        
        for name in self.reward_dict.keys():
            rewards[name] = np.vstack(self.reward_dict[name])
        
        terminated = np.vstack(self.terminals)
        # terminated = np.expand_dims(terminated, axis=1)
        truncated = np.vstack(self.truncateds)
        # terminated = np.expand_dims(truncated, axis=1)
        
        return next_obs, computing_obs, rewards, terminated, truncated
        

class RLVectorEnv(EnvBase):
    """

    This is the base class of vector environment.
    """

    def __init__(self, 
                env_class:EnvBase, 
                env_num:int,
                envs_args=None):
        """
        创建一个vector环境。
        
        Args:
            env_class (EnvBase): 环境的类。
            num_envs (int): 环境数量。
            envs_args (dict or list): 环境参数，AquaML允许环境使用不同的参数。
            
        """
        super(RLVectorEnv, self).__init__()

        ########################################
        # 1. 初始化环境
        ########################################

        self._envs = []

        # 为每个环境创建相应的参数
        if envs_args is None:
            envs_args_tuple = [{} for _ in range(env_num)]
        else:
            if isinstance(envs_args, list):
                if len(envs_args) != env_num:
                    logger.error('envs_args length must be equal to env_num')
                    raise ValueError("envs_args length must be equal to env_num")
                else:
                    envs_args_tuple = envs_args
            elif isinstance(envs_args, dict):
                envs_args_tuple = [envs_args for _ in range(env_num)]
            else:
                logger.error('envs_args must be list or dict')
                raise TypeError("envs_args must be list or dict")

        for i in range(env_num):
            env_ = env_class(**envs_args_tuple[i])
            # env_.set_id(i)
            self._envs.append(env_)

        ########################################
        # 2. 生成env_info
        ########################################

        # _reward_info = {}

        env_1:EnvBase = self._envs[0]

        
        self._obs_info = env_1.obs_info
        self._rewards = env_1.reward_info.names

        # self._reward_info = env_1.reward_info
        # TODO：明确这个地方接口
        # self._reward_info = ('indicate', *env_1.reward_info) 
        ########################################
        # 3.重要的API接口
        ########################################

        self._num_envs = env_num

        self.last_obs = None
        
    
    def auto_reset(self):
        """
        重置环境。
        
        Returns:
            observation: 环境的观察值。
            info：环境的信息。
        """
        
        collector = VecCollector()
        
        for env in self._envs:
            obs, info = env.auto_reset()
            collector.reset_append(obs)
        
        obs = collector.get_reset_data()
        
        return obs, None
    
    def auto_step(self, actions: dict, max_step:int = np.inf):
        """
        自动执行一步。
        
        Args:
            actions (dict): 动作。
            max_step (int, optional): 最大步数。 默认为np.inf。
        
        Returns:
            next_obs (dict): 下一个观察值。
            rewards (dict): 奖励。
            dones (list): 是否终止。
            infos (dict): 信息。
        """
        
        collector = VecCollector()
        
        for key, value in actions.items():
            actions[key] = aqua_tool.convert_numpy_fn(value)
        
        for i in range(self._num_envs):
            # TODO:这个地方可以提速
            sub_action_dict = {}
            for key, value in actions.items():
                sub_action_dict[key] = value[i, :]
                
            next_obs,computing_obs, reward,  terminal, truncated, info = self._envs[i].auto_step(sub_action_dict, max_step=max_step)
            collector.append(next_obs=next_obs,
                             reward=reward,
                             computing_obs=computing_obs,
                             terminal=terminal,
                             truncated=truncated)
            
        next_obs, computing_obs, rewards, terminated, truncated = collector.get_data()
        
        return next_obs, computing_obs, rewards, terminated, truncated, None
    
    def close(self):
        
        for env in self._envs:
            env.close()
            
    def step(self, action):
        logger.error("This method is not implemented.")
        
    def reset(self):
        logger.error("This method is not implemented.")
        
      