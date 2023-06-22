from abc import ABC, abstractmethod
import numpy as np

class RLBaseEnv(ABC):
    """
    The base class of environment.
    
    All the environment should inherit this class.
    
    reward_info should be a specified.

    obs_info should be a specified.
    """

    def __init__(self):
        self._reward_info = ('total_reward',)  # reward info is a tuple
        self._obs_info = None  # DataInfo
        self.action_state_info = {}  # default is empty dict
        self.adjust_parameters = []  # default is empty list, this is used for high level algorithm
        self.meta_parameters = {}  # meta参数接口
        self.reward_fn_input = []  # 计算reward需要哪些参数, 使用meta时候声明

    @abstractmethod
    def reset(self):
        """
        Reset the environment.
        
        return: 
        observation (dict): observation of environment.
        """

    def update_meta_parameter_by_args_pool(self, args_pool):
        """
        update meta parameters.
        """
        for key, value in self.meta_parameters.items():
            value = args_pool.get_param(key)
            setattr(self, key, value)

    def display(self):
        """
        Display the environment.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Step the environment.
        
        Args:
            action (optional): action of environment.
        Return: 
        observation (dict): observation of environment.
        reward(dict): reward of environment.
        done (bool): done flag of environment.
        info (dict or None): info of environment.
        """

    @abstractmethod
    def close(self):
        """
        Close the environment.
        """

    @property
    def reward_info(self):
        return self._reward_info

    @property
    def obs_info(self):
        return self._obs_info

    def initial_obs(self, obs):
        for key, shape in self.action_state_info.items():
            obs[key] = np.zeros(shape=shape, dtype=np.float32).reshape(1, -1)
        return obs

    def check_obs(self, obs, action_dict):
        for key in self.action_state_info.keys():
            obs[key] = action_dict[key]
        return obs

    def set_action_state_info(self, actor_out_info: dict, actor_input_name: tuple):
        """
        set action state info.
        Judge the input is as well as the output of actor network.

        """
        for key, shape in actor_out_info.items():
            if key in actor_input_name:
                self.action_state_info[key] = shape
                self._obs_info.add_info(key, shape, np.float32)

    @property
    def get_env_info(self):
        """
        get env info.
        """
        info = {
            "reward_info": self.reward_info,
            "obs_info": self.obs_info,

        }

        return info

    def get_reward(self):
        """
        该函数用于计算reward，用于meta中得到reward，如果不需要使用，不用理会。
        当然，此函数实现需要注意，计算时候需要考虑矩阵运算，此为meta唯一接口，因此在core中可以不一样，
        内部计算必须使用tf内部函数，防止梯度消失，具体参照TensorFlow官网中关于Tf.GradientTape介绍。

        我们建议support env和core env使用不同reward函数，但是拥有一套meta参数。
        """



