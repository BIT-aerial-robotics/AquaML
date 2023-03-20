# provide base class for all classes in AquaML

from abc import ABC, abstractmethod
import tensorflow as tf
import abc
import os
import numpy as np
from AquaML.DataType import DataInfo
from AquaML.data.DataUnit import DataUnit


# TODO: implement the base class when implementing the other classes
class BaseAlgo(ABC):
    def __init__(self):

        self.data_pool = None # 数据池子
        self.sample_thread_num = None # 能够使用的子线程数目
        self.each_summary_episodes = None # 验证数目
        self.algo_info = None

    @property
    def algo_name(self):
        return self.name

    def create_data_pool(self):
        """
        由于在2.1版本中，所有的data pool将逐渐统一这个函数，这个函数也会逐渐改进扩展。
        """

        # 添加summary信息到meta_info中
        reward_info_dict = {}
        for name in self.algo_info.reward_info:
            reward_info_dict['summary_' + name] = (
                self.each_summary_episodes * self.sample_thread_num, 1)

        for name, shape in reward_info_dict.items():
            buffer = DataUnit(
                name=name + '_' + name,
                shape=shape,
                dtype=np.float32,
                level=self.level,
                computer_type=self._computer_type,
            )
            self.algo_info.add_info(
                name=name,
                shape=shape,
                dtype=np.float32,
            )
            self.data_pool.add_unit(
                name=name,
                data_unit=buffer
            )

        # 无论是否使用多线程，我们都会开启共享内存池子，方便管理
        self.data_pool.multi_init(
            self.algo_info.data_info,
            type='buffer'
        )

    def get_corresponding_data(self, data_dict: dict, names: tuple, prefix: str = '', tf_tensor: bool = True):
        """

        Get corresponding data from data dict.

        Args:
            data_dict (dict): data dict.
            names (tuple): name of data.
            prefix (str): prefix of data name.
            tf_tensor (bool): if return tf tensor.
        Returns:
            corresponding data. list or tuple.
        """

        data = []

        for name in names:
            name = prefix + name
            buffer = data_dict[name]
            if tf_tensor:
                buffer = tf.cast(buffer, dtype=tf.float32)
            data.append(buffer)

        return data

    def get_all_data(self):
        """
        TODO: v2.1版本中将此函数扩展为标准接口，逐步改进， data_pool作为输入之类的

        该函数会自动搜寻所有pool中的数据，并且匹配格式，返回一个dict
        """
        return_dict = {}

        for key, unit in self.data_pool.data_pool.items():
            return_dict[key] = unit.buffer

        if self.meta_parameter_names is not None:
            for key in self.meta_parameter_names:
                value = self.args_pool.get_param(key)
                return_dict[key] = np.ones_like(return_dict['total_reward']) * value
                return_dict['next_' + key] = np.ones_like(return_dict['total_reward']) * value

        return return_dict

    def load_weights_from_file(self):
        """
        load weights from file.
        """
        for key, model in self._all_model_dict.items():
            path = getattr(model, 'weight_path', None)
            if path is not None:
                model.load_weights(path)
                print("load {} weight from {}".format(key, path))


class BaseStarter(ABC):

    def __init__(self):

        self._work_folder = None
        self._log_folder = None
        self._cache_folder = None

        self.level = None

    def initial_dir(self, work_folder: str):
        """
        Initial the directory for working.

        Args:
            work_folder (_type_:str): name of work folder.

        Returns:
            _type_: None
        """
        # create a folder for working
        if self.level == 0:
            self._work_folder = work_folder
            self.mkdir(self.work_folder)

            # create a folder for storing the log
            self._log_folder = self.work_folder + '/log'
            self.mkdir(self.log_folder)

            # create cache folder
            self._cache_folder = self.work_folder + '/cache'
            self.mkdir(self.cache_folder)
        else:
            self._work_folder = work_folder
            self._log_folder = self.work_folder + '/log'
            self._cache_folder = self.work_folder + '/cache'

    @staticmethod
    def mkdir(path: str):
        """
        create a directory in current path.

        Args:
            path (_type_:str): name of directory.

        Returns:
            _type_: str or None: path of directory.
        """
        current_path = os.getcwd()
        # print(current_path)
        path = os.path.join(current_path, path)
        if not os.path.exists(path):
            os.makedirs(path)
            return path
        else:
            None

    @property
    def work_folder(self):
        return self._work_folder

    @property
    def log_folder(self):
        return self._log_folder

    @property
    def cache_folder(self):
        return self._cache_folder


# Base class for reinforcement learning algorithm

# TODO: 重新规定环境的接口
class RLBaseEnv(abc.ABC):
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

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    def get_reward(self):
        """
        该函数用于计算reward，用于meta中得到reward，如果不需要使用，不用理会。
        当然，此函数实现需要注意，计算时候需要考虑矩阵运算，此为meta唯一接口，因此在core中可以不一样，
        内部计算必须使用tf内部函数，防止梯度消失，具体参照TensorFlow官网中关于Tf.GradientTape介绍。

        我们建议support env和core env使用不同reward函数，但是拥有一套meta参数。
        """


class RLBaseModel(abc.ABC, tf.keras.Model):
    """All the neral network models should inherit this class.
    
    The default optimizer is Adam. If you want to change it, you can set _optimizer.
    such as, self._optimizer = 'SGD'
    
    The learning rate is 0.001. If you want to change it, you can set _lr.
    
    If the model is Q network, _input_name should not contain 'action'
    
    You should specify the input name by set self._input_name.
    """

    def __init__(self):
        super(RLBaseModel, self).__init__()
        self.rnn_flag = False
        self._optimizer = 'Adam'
        self._learning_rate = 0.001

        self._input_name = None

        # if the model is an actor, please specify the output info
        # eg: {'action':(2,), 'log_std':(2,)}
        self._output_info = None

    @abc.abstractmethod
    def reset(self):
        """
        Reset the model.
        Such as reset the rnn state.
        """

    @abc.abstractmethod
    def call(self, *args, **kwargs):
        """
        The call function of keras model.
        
        Return is tuple or tf.Tensor.
        
        When the model is actor, the return is tuple like (action, (h, c)).
        When the model is critic, the return is q_value or value.
        """

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def input_name(self):
        return self._input_name

    @property
    def output_info(self):
        return self._output_info


# TODO: 多线程参数设置太反人类，逐渐替换
class BaseParameter:
    """
    The base class of parameter.
    """

    def __init__(self):
        self.meta_parameters = {}

    def add_meta_parameters(self, meta_parameters: dict):
        """
        Set the meta parameters.
        """
        for key, value in meta_parameters.items():
            self.meta_parameters[key] = value
            setattr(self, key, value)

    def add_meta_parameter_by_names(self, meta_parameter_names: tuple or list):
        """
        Set the meta parameters by names.
        """
        dic = {}
        for name in meta_parameter_names:
            value = getattr(self, name)
            dic[name] = value

        self.add_meta_parameters(dic)
        self.meta_parameters = dic

    def update_meta_parameters(self, meta_parameters: dict):
        """
        Update the meta parameters.
        """
        for key, value in meta_parameters.items():
            if key in self.meta_parameters.keys():
                self.meta_parameters[key] = value
                setattr(self, key, value)

    def update_meta_parameter_by_args_pool(self, args_pool):
        """
        Update the meta parameters by args pool.
        """
        for key, value in self.meta_parameters.items():
            value = args_pool.get_param(key)
            setattr(self, key, value)

    @property
    def meta_parameter_names(self):
        return tuple(self.meta_parameters.keys())


if __name__ == '__main__':
    parameter = BaseParameter()

    parameter.add_meta_parameters({'a': 1, 'b': 2})

    parameter.add_meta_parameter_by_names(['a', 'b'])

    print(parameter.meta_parameters)
