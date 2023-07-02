from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from AquaML.core.DataParser import DataInfo
import copy


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    # def update(self, x):
    #     x = np.asarray(x)
    #     size = x.shape[0]
    #
    #     for i in range(size):
    #         self.__update(x[i])

    def update(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = copy.deepcopy(self.mean)
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, obs_shape_dict: dict):
        self.running_ms = {}
        for key, value in obs_shape_dict.items():
            self.running_ms[key] = RunningMeanStd(value)

    def __call__(self, x: dict, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        new_x = {}
        if update:
            for key, value in x.items():
                self.running_ms[key].update(value)

        for key, value in x.items():
            new_x[key] = (value - self.running_ms[key].mean) / (self.running_ms[key].std + 1e-8)

        return new_x


class RLStandardDataSet:
    """
    RLStandardDataSet. 

    The data set for RL algorithm. All the RL buffer plugin 
    should return this type of data set.
    """

    def __init__(self,
                 rollout_steps,
                 num_envs,
                 ) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs

        self._buffer_dict = {}

    def __call__(self, obs: dict, action: dict, reward: dict, next_obs: dict, mask: np.ndarray):

        # check the data type
        self.check_dict(obs, 'obs')
        self.check_dict(action, 'action')
        self.check_dict(reward, 'reward')
        self.check_dict(next_obs, 'next_obs')
        self.check_data(mask, 'mask')

        self.obs_names = obs.keys()
        self.action_names = action.keys()
        self.reward_names = reward.keys()
        self.next_obs_names = next_obs.keys()

        for name in self.obs_names:
            setattr(self, name, obs[name])
            self._buffer_dict[name] = getattr(self, name)

        for name in self.action_names:
            setattr(self, name, action[name])
            self._buffer_dict[name] = getattr(self, name)

        for name in self.reward_names:
            setattr(self, name, reward[name])
            self._buffer_dict[name] = getattr(self, name)

        for name in self.next_obs_names:
            setattr(self, name, next_obs[name])
            self._buffer_dict[name] = getattr(self, name)

        self.mask = mask
        self._buffer_dict['mask'] = self.mask

    def get_env_data(self):
        """
        get the data of each env.

        return a generator, the element of the generator is a dict, 
        the key is the name of the data, the value is the data.

        Return:
            env_data: a dict, the key is the name of the data, the value is the data. 
            And the shape of the data is (rollout_steps, ...)

        """

        for i in range(self.num_envs):
            env_data = {}
            for name in self._buffer_dict.keys():
                env_data[name] = self._buffer_dict[name][i]
            yield env_data

    def add_data(self, data: Any, name: str):
        """
        add data to the data set.

        check and add the data to the data set.
        """

        self.check_data(data, name)
        setattr(self, name, data)

    def check_dict(self, dic: dict, name: str):
        """
        check the dict.

        the element in the dict should be np.ndarray. And the shape should be:
        (num_envs, rollout_steps, ...)
        """

        for key in dic.keys():
            if not isinstance(dic[key], np.ndarray):
                raise TypeError(f'The type of {name} should be np.ndarray, but got {type(dic[key])}.')

            if dic[key].shape != (self.num_envs, self.rollout_steps, *dic[key].shape[2:]):
                raise ValueError(
                    f'The shape of {name} should be (num_envs, rollout_steps, ...), but got {dic[key].shape}.')

    def check_data(self, data, name: str):
        """
        check the data.

        the data should be np.ndarray. And the shape should be:
        (num_envs, rollout_steps, ...)
        """

        if isinstance(data, np.ndarray):
            if data.shape != (self.num_envs, self.rollout_steps, *data.shape[2:]):
                raise ValueError(f'The shape of {name} should be (num_envs, rollout_steps, ...), but got {data.shape}.')
        elif isinstance(data, dict):
            self.check_dict(data, name)
        else:
            raise TypeError(f'The type of {name} should be np.ndarray or dict, but got {type(data)}.')


class MDPCollector:
    """
    ThreadCollector
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
            obs_dict[name] = np.vstack(self.obs_dict[name])

        action_dict = {}

        for name in self.action_names:
            action_dict[name] = np.vstack(self.action_dict[name])

        reward_dict = {}

        for name in self.reward_names:
            reward_dict[name] = np.vstack(self.reward_dict[name])

        next_obs_dict = {}

        for name in self.next_obs_names:
            next_obs_dict['next_' + name] = np.vstack(self.next_obs_dict[name])

        mask = np.vstack(self.masks)

        return obs_dict, action_dict, reward_dict, next_obs_dict, mask


class VecMDPCollector:
    """
    vectorize MDP collcetor
    """

    def __init__(self,
                 obs_names,
                 next_obs_names,
                 action_names,
                 reward_names,
                 ):
        self.obs_names = obs_names
        self.action_names = action_names
        self.reward_names = reward_names
        self.next_obs_names = next_obs_names

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
            self.obs_dict[name].append(np.expand_dims(obs[name], axis=1))

        for name in self.action_names:
            self.action_dict[name].append(np.expand_dims(action[name], axis=1))

        for name in self.reward_names:
            self.reward_dict[name].append(np.expand_dims(reward[name], axis=1))

        for name in self.next_obs_names:
            self.next_obs_dict[name].append(np.expand_dims(next_obs[name], axis=1))

        self.masks.append(np.expand_dims(mask, axis=1))

    def get_data(self):
        """
        get data

        返回的数据格式为
        """

        obs_dict = {}

        for name in self.obs_names:
            obs_dict[name] = np.concatenate(self.obs_dict[name], axis=1)

        action_dict = {}

        for name in self.action_names:
            action_dict[name] = np.concatenate(self.action_dict[name], axis=1)

        reward_dict = {}

        for name in self.reward_names:
            reward_dict[name] = np.concatenate(self.reward_dict[name], axis=1)

        next_obs_dict = {}

        for name in self.next_obs_names:
            next_obs_dict[name] = np.concatenate(self.next_obs_dict[name], axis=1)

        mask = np.concatenate(self.masks, axis=1)

        return obs_dict, action_dict, reward_dict, next_obs_dict, mask


class VecCollector:
    """
    Collect data from sub envs.
    """

    def __init__(self):
        self.obs_dict = {}
        self.compute_obs_dict = {}
        self.reward_dict = {}
        self.masks = []

    def append(self, next_obs, reward, mask, computing_obs):
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
        for name in next_obs.keys():
            all_name = name
            if all_name not in self.obs_dict.keys():
                self.obs_dict[all_name] = []
            self.obs_dict[all_name].append(next_obs[name])

        for name in computing_obs.keys():
            if name not in self.compute_obs_dict.keys():
                self.compute_obs_dict[name] = []
            self.compute_obs_dict[name].append(computing_obs[name])

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
        compute_obs = {}

        for name in self.obs_dict.keys():
            obs[name] = np.vstack(self.obs_dict[name])
            if expand_dim:
                obs[name] = np.expand_dims(obs[name], axis=0)

        for name in self.reward_dict.keys():
            reward[name] = np.vstack(self.reward_dict[name])
            if expand_dim:
                reward[name] = np.expand_dims(reward[name], axis=0)

        for name in self.compute_obs_dict.keys():
            compute_obs[name] = np.vstack(self.compute_obs_dict[name])
            if expand_dim:
                compute_obs[name] = np.expand_dims(compute_obs[name], axis=0)

        mask = np.vstack(self.masks)

        return obs, reward, mask, compute_obs

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
        self.steps = 0
        self._reward_info = ('total_reward',)  # reward info is a tuple
        self._obs_info = None  # DataInfo
        self.action_state_info = {}  # default is empty dict
        self.adjust_parameters = []  # default is empty list, this is used for high level algorithm
        self.meta_parameters = {}  # meta参数接口
        self.reward_fn_input = []  # 计算reward需要哪些参数, 使用meta时候声明

        self.done = True

        self.id = None

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

    def step_vector(self, action_dict: dict, max_step: int = 1000000):
        """
        Step the environment.
        
        Args:
            action_dict (dict): action of environment.
            max_step (int): max step of environment.
        Return: 
        observation (dict): observation of environment.
        reward(dict): reward of environment.
        done (bool): done flag of environment.
        info (dict or None): info of environment.

        """
        self.steps += 1
        next_obs, reward, done, info = self.step(action_dict)

        reward['indicate'] = copy.deepcopy(reward['total_reward'])

        if self.steps >= max_step:
            done = True

        if done:
            self.steps = 0
            obs, flag = self.reset()
            computing_obs = obs
        else:
            computing_obs = next_obs
            # normalize_flag = False

        # next_obs['step'] = self.steps

        return next_obs, reward, done, computing_obs

    @abstractmethod
    def close(self):
        """
        Close the environment.
        """

    @property
    def reward_info(self) -> tuple or list:
        return self._reward_info

    @property
    def obs_info(self) -> DataInfo:
        if self._obs_info is None:
            raise ValueError("obs_info is not specified.")
        return self._obs_info

    @property
    def num_envs(self):
        return 1

    def initial_obs(self, obs):
        for key, shape in self.action_state_info.items():
            obs[key] = np.zeros(shape=shape, dtype=np.float32).reshape(1, -1)
        return obs

    def check_obs(self, obs, action_dict):
        for key in self.action_state_info.keys():
            obs[key] = action_dict[key]
        return obs

    def set_action_state_info(self, actor_out_info: dict, actor_input_name: tuple) -> None:
        """
        set action state info.
        Judge the input is as well as the output of actor network.

        """
        for key, shape in actor_out_info.items():
            if key in actor_input_name:
                self.action_state_info[key] = shape
                self._obs_info.add_info(key, shape, np.float32)

    @property
    def get_env_info(self) -> dict:
        """
        get env info.
        """
        info = {
            "reward_info": self.reward_info,
            "obs_info": self.obs_info,

        }

        return info

    def set_id(self, id):
        self.id = id

    def get_reward(self):
        """
        该函数用于计算reward，用于meta中得到reward，如果不需要使用，不用理会。
        当然，此函数实现需要注意，计算时候需要考虑矩阵运算，此为meta唯一接口，因此在core中可以不一样，
        内部计算必须使用tf内部函数，防止梯度消失，具体参照TensorFlow官网中关于Tf.GradientTape介绍。

        我们建议support env和core env使用不同reward函数，但是拥有一套meta参数。
        """

    def seed(self, seed):
        """
        设置随机种子
        """
        raise NotImplementedError


class RLVectorEnv:
    """

    This is the base class of vector environment.
    """

    def __init__(self, env, num_envs, envs_args=None, normalize_obs=False):
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
            env_ = env(**envs_args_tuple[i])
            env_.set_id(i)
            self._envs.append(env_)

        ########################################
        # vectorized env info
        ########################################

        # _reward_info = {}

        env_1 = self._envs[0]

        if not isinstance(env_1, RLBaseEnv):
            raise TypeError("can not recognize env type")

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

        # self._reward_info = env_1.reward_info
        self._reward_info = ('indicate', *env_1.reward_info)
        ########################################
        # key info API
        ########################################

        self._num_envs = num_envs

        self.action_state_info = {}

        self.last_obs = None

        self._max_steps = 100000

        self.normalize_obs = normalize_obs

        self._normalize_obs = Normalization(
            self._obs_info.shape_dict,
        )

    def set_max_steps(self, max_steps):
        self._max_steps = max_steps

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

            sub_next_obs, sub_rew, sub_done, computing_obs = self._envs[i].step_vector(sub_action_dict, self._max_steps)

            if self.normalize_obs:
                sub_next_obs_ = self._normalize_obs(copy.deepcopy(sub_next_obs))
                sub_next_obs.update(sub_next_obs_)

                computing_obs_ = self._normalize_obs(copy.deepcopy(computing_obs), update=False)

                computing_obs.update(computing_obs_)

            vec_collector.append(sub_next_obs, sub_rew, 1 - sub_done, computing_obs)

        obs, rew, done, compute_obs = vec_collector.get_data(expand_dim=False)

        return obs, rew, done, compute_obs

    def reset(self):
        """
        reset the environment.

        This function is only used in the beginning of the episode.
        """
        vec_collector = VecCollector()

        for i in range(self._num_envs):
            sub_obs, flag = self._envs[i].reset()
            vec_collector.inital_appand(sub_obs)

        obs = vec_collector.get_initial_data(expand_dim=False)

        return obs

    def set_action_state_info(self, actor_out_info: dict, actor_input_name: tuple):

        for key, shape in actor_out_info.items():
            if key in actor_input_name:
                for env in self._envs:
                    env.set_action_state_info(actor_out_info, actor_input_name)

                self.action_state_info[key] = shape
                self._obs_info.add_info(key, shape, np.float32)

    def close(self):
        """
        close the environment.
        """
        for env in self._envs:
            env.close()

    @property
    def reward_info(self) -> tuple or list:
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
