from abc import ABC, abstractmethod
import numpy as np
from AquaML.core.DataParser import DataInfo


class VecMDPCollector:
    """
    vectorize MDP collcetor
    """

    def __init__(self,
                 obs_names,
                 action_names,
                 reward_names,
                 ):
        self.obs_names = obs_names
        self.action_names = action_names
        self.reward_names = reward_names
        self.next_obs_names = obs_names

        self.obs_dict = {}
        self.action_dict = {}
        self.reward_dict = {}
        self.next_obs_dict = {}

        self.masks = []

    def reset(self):
        """
        reset
        """
        del self.obs_dict
        del self.action_dict
        del self.reward_dict
        del self.next_obs_dict
        del self.masks

        self.obs_dict = {}
        self.action_dict = {}
        self.reward_dict = {}
        self.next_obs_dict = {}

        self.masks = []

        for name in self.obs_names:
            self.obs_dict[name] = []

        for name in self.action_names:
            self.action_dict[name] = []

        for name in self.reward_names:
            self.reward_dict[name] = []

        for name in self.next_obs_names:
            self.next_obs_dict[name] = []

    def store_data(self, obs: dict, action: dict, reward: dict, next_obs: dict, mask: int):
        """
        store data
        """

        for name in self.obs_names:
            self.obs_dict[name].append(obs[name])

        for name in self.action_names:
            self.action_dict[name].append(action[name])

        for name in self.reward_names:
            self.reward_dict[name].append(reward[name])

        for name in self.next_obs_names:
            self.next_obs_dict[name].append(next_obs[name])

        self.masks.append(mask)

    def get_data(self):
        """
        get data

        返回的数据格式为
        """

        obs_dict = {}

        for name in self.obs_names:
            obs_dict[name] = np.hstack(self.obs_dict[name])

        action_dict = {}

        for name in self.action_names:
            action_dict[name] = np.hstack(self.action_dict[name])

        reward_dict = {}

        for name in self.reward_names:
            reward_dict[name] = np.hstack(self.reward_dict[name])

        next_obs_dict = {}

        for name in self.next_obs_names:
            next_obs_dict['next_' + name] = np.hstack(self.next_obs_dict[name])

        mask = np.hstack(self.masks)

        return obs_dict, action_dict, reward_dict, next_obs_dict, mask


class VecCollector:
    """
    Collect data from sub envs.
    """

    def __init__(self):
        self.obs_dict = {}
        self.reward_dict = {}
        self.masks = []

    def append(self, obs, reward, mask):
        """
        Append data to collector.

        Every element in obs, action, reward should be like:
        shape = (1, *)

        Args:
            obs (dict): observation.
            action (dict): action.
            reward (dict): reward.
            mask (int): mask.
        """
        for name in obs.keys():
            if name not in self.obs_dict.keys():
                self.obs_dict[name] = []
            self.obs_dict[name].append(obs[name])

        for name in reward.keys():
            if name not in self.reward_dict.keys():
                self.reward_dict[name] = []
            self.reward_dict[name].append(reward[name])

        self.masks.append(mask)

    def get_data(self, expand_dim=False):
        """
        Get data from collector.


        Args:
            expand_dim (bool): expand dim of data.

            if expand_dim is True, the shape of data will be (1, *)
            else the shape of data will be (*, )
        Return:
            obs (dict): observation.
            action (dict): action.
            reward (dict): reward.
            mask (nd.array): mask.
        """
        obs = {}
        reward = {}

        for name in self.obs_dict.keys():
            obs[name] = np.vstack(self.obs_dict[name])
            if expand_dim:
                obs[name] = np.expand_dims(obs[name], axis=0)

        for name in self.reward_dict.keys():
            reward[name] = np.vstack(self.reward_dict[name])
            if expand_dim:
                reward[name] = np.expand_dims(reward[name], axis=0)

        mask = np.vstack(self.masks)

        return obs, reward, mask

    def get_data_h(self, expand_dim=False):
        """
        Get data from collector.


        Args:
            expand_dim (bool): expand dim of data.

            if expand_dim is True, the shape of data will be (1, *)
            else the shape of data will be (*, )

        Return:
            obs (dict): observation.
            action (dict): action.
            reward (dict): reward.
            mask (nd.array): mask.
        """

        obs = {}
        reward = {}

        for name in self.obs_dict.keys():
            obs[name] = np.hstack(self.obs_dict[name])
            if expand_dim:
                obs[name] = np.expand_dims(obs[name], axis=0)

        for name in self.reward_dict.keys():
            reward[name] = np.hstack(self.reward_dict[name])
            if expand_dim:
                reward[name] = np.expand_dims(reward[name], axis=0)

        mask = np.vstack(self.masks)

        return obs, reward, mask

    def inital_appand(self, obs):
        """
        Inital append data to collector.

        Every element in obs, action, reward should be like:
        shape = (1, *)

        Args:
            obs (dict): observation.
        """
        for name in obs.keys():
            if name not in self.obs_dict.keys():
                self.obs_dict[name] = []
            self.obs_dict[name].append(obs[name])

    def get_initial_data(self, expand_dim=False):
        """
        Get data from collector.


        Args:
            expand_dim (bool): expand dim of data.

            if expand_dim is True, the shape of data will be (1, *)
        """
        obs = {}

        for name in self.obs_dict.keys():
            obs[name] = np.vstack(self.obs_dict[name])
            if expand_dim:
                obs[name] = np.expand_dims(obs[name], axis=0)

        return obs


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

        self.last_done = True

    @abstractmethod
    def reset(self) -> tuple:
        """
        Reset the environment.
        
        return: 
        observation (dict): observation of environment.
        flag (bool): undefined.
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
    def step(self, action) -> tuple:
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

    def step_vector(self, action_dict: dict):
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

        obs, reward, done, info = self.step(action_dict)

        if done:
            obs, flag = self.reset()

        return obs, reward, done, info

    @abstractmethod
    def close(self):
        """
        Close the environment.
        """

    @property
    def reward_info(self):
        return self._reward_info

    @property
    def obs_info(self) -> DataInfo:
        if self._obs_info is None:
            raise ValueError("obs_info is not specified.")
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


class RLVerctorEnv:
    """

    This is the base class of vector environment.
    """

    def __init__(self, env, num_envs, envs_args=None):
        """
        initialize the vector env.

        Args:
            env (RLBaseEnv): the env class.

        """

        ########################################
        # instanclize the env
        ########################################

        self._envs = []

        if envs_args is None:
            envs_args_tuple = [{} for _ in range(num_envs)]
        else:
            if isinstance(envs_args, list):
                if len(envs_args) != num_envs:
                    raise ValueError("envs_args length must be equal to num_envs")
                else:
                    envs_args_tuple = envs_args
            elif isinstance(envs_args, dict):
                envs_args_tuple = [envs_args for _ in range(num_envs)]
            else:
                raise TypeError("envs_args must be list or dict")

        for i in range(num_envs):
            self._envs.append(env(**envs_args_tuple[i]))

        if not isinstance(env, RLBaseEnv):
            raise TypeError("can not recognize env type")

        ########################################
        # vectorized env info
        ########################################

        _reward_info = {}

        env_1 = self._envs[0]

        # get vectorized reward info
        # for key in env_1.reward_info:
        #     for key in env_1.reward_info:
        #         _reward_info[key] = (num_envs,)
        #
        # # get vectorized obs info
        # obs_data_info = env_1.obs_info
        #
        # obs_data_shape_ = obs_data_info.shape_dict
        # new_obs_data_shape = []
        #
        # for _, value in obs_data_shape_.item():
        #     new_obs_data_shape.append((num_envs, *value))
        #
        # self._obs_info = DataInfo(
        #     names=obs_data_info.names,
        #     shapes=new_obs_data_shape,  # type: ignore
        #     dtypes=obs_data_info.dtypes,
        # )
        #
        # self._reward_info = DataInfo(
        #     names=tuple(_reward_info.keys()),
        #     shapes=tuple(_reward_info.values()),
        #     dtypes=np.float32,
        # )

        self._obs_info = env_1.obs_info
        self._reward_info = env_1.reward_info

        ########################################
        # key info API
        ########################################

        self._num_envs = num_envs

        self.action_state_info = {}

        self.last_obs = None

    def step(self, actions: dict):
        """
        Step the environment.

        Args:
            actions (dict): actions of environment. The key is the name of action, {"action": (num_envs, action_dim),
                                                                                    "hidden_state": (num_envs, hidden_state_dim),}
                                                    

        """
        vec_collector = VecCollector()

        for i in range(self._num_envs):
            sub_action_dict = {}
            for key, value in actions.items():
                sub_action_dict[key] = value[i, :]
            sub_obs, sub_rew, sub_done, sub_info = self._envs[i].step(sub_action_dict)
            vec_collector.append(sub_obs, sub_rew, 1 - sub_done)

        obs, rew, done = vec_collector.get_data(expand_dim=True)

        return obs, rew, done

    def reset(self):
        """
        reset the environment.

        This function is only used in the beginning of the episode.
        """
        vec_collector = VecCollector()

        for i in range(self._num_envs):
            sub_obs, flag = self._envs[i].reset()
            vec_collector.inital_appand(sub_obs)

        obs = vec_collector.get_data(expand_dim=False)


        return obs

    def set_action_state_info(self, actor_out_info: dict, actor_input_name: tuple):

        for key, shape in actor_out_info.items():
            if key in actor_input_name:
                for env in self._envs:
                    env.set_action_state_info(actor_out_info, actor_input_name)

                self.action_state_info[key] = (self._num_envs, *shape)
                self._obs_info.add_info(key, (self._num_envs, *shape), np.float32)

    @property
    def reward_info(self):
        return self._reward_info

    @property
    def obs_info(self):
        return self._obs_info

    @property
    def num_envs(self):
        return self._num_envs

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
