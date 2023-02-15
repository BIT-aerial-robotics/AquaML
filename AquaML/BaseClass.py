# provide base class for all classes in AquaML

from abc import ABC, abstractmethod
import tensorflow as tf
import abc
import os
import numpy as np


# TODO: implement the base class when implementing the other classes
class BaseAlgo(ABC):
    @property
    def algo_name(self):
        return self.name


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

    @abc.abstractmethod
    def reset(self):
        """
        Reset the environment.
        
        return: 
        observation (dict): observation of environment.
        """

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


if __name__ == '__main__':
    BaseStarter.mkdir('test')
