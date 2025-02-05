from . import BaseEnv
from typing import Any
import gymnasium as gym
from typing import TYPE_CHECKING, Union
import numpy as np
from loguru import logger
if TYPE_CHECKING:
    from AquaML.data import unitCfg


class VirtualEnv(BaseEnv):
    '''
    Virtual environment. This env does not have any real environment. 
    But it can load env information from the configuration file.
    '''

    def __init__(self, env: Any):
        """Virtual environment

        This class is a virtual environment. It does not have any real environment.
        But it can load env information from the configuration file.

        :param env: The environment to be wrapped.
        :type env: str or Any supported Gymnasium environment
        """

        super().__init__()

    def reset(self):
        '''
        Reset the environment to the initial state.
        '''
        logger.error("Virtual environment dose not have function reset.")

        raise NotImplementedError

    def step(self, action):
        '''
        Take an action and return the next state, reward, done, and info.
        '''
        logger.error("Virtual environment dose not have function step.")

        raise NotImplementedError

    def loadEnvInfo(self, env_info: dict):
        '''
        Load the environment information from the configuration file.

        :param env_info: The environment information.
        :type env_info: dict
        '''
        # Load the observation configuration.
        observation_cfg_dict = env_info['observation_cfg']

        for key, value in observation_cfg_dict.items():
            self.observation_cfg_[key] = unitCfg(**value)

        # Load the action configuration.
        action_cfg_dict = env_info['action_cfg']

        for key, value in action_cfg_dict.items():
            self.action_cfg_[key] = unitCfg(**value)

        # Load the number of environments.
        self.num_envs = env_info['num_envs']
