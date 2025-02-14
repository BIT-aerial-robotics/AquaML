from . import BaseEnv
from typing import Any
import gymnasium as gym
from typing import TYPE_CHECKING, Union
import numpy as np
from .gymnasium_envs import GymnasiumWrapper
from AquaML.collector import DictConcat, ScalarConcat
import torch


class GymnasiumVectorWrapper(BaseEnv):
    '''
    The wrapper for Gymnasium environment.

    This class makes the Gymnasium environment support vectorized environments.
    In the one thread, it can run multiple environments.
    '''

    def __init__(self, env: Any, num_envs: int):
        """Gymnasium environment wrapper

        This class is a wrapper for vectorized Gymnasium environment.
        One instance run in one thread.

        :param env: The environment to be wrapped.
        :type env: str or Any supported Gymnasium environment
        :param num_envs: The number of environments.
        :type num_envs: int
        """

        super().__init__()

        self.num_envs = num_envs  # The number of environments.

        self.envs_ = [GymnasiumWrapper(env) for _ in range(num_envs)]

        self.observation_cfg_ = self.envs_[0].getObservationCfg()
        self.action_cfg_ = self.envs_[0].getActionCfg()
        self.reward_cfg_ = self.envs_[0].getRewardCfg()

        # concat the observation and action configuration
        # collect the observation data, once the data is collected, it will be concatenated.
        self.observation_concat_ = DictConcat()
        self.reward_concat_ = DictConcat()
        self.done_concat_ = ScalarConcat()
        self.truncated_concat_ = ScalarConcat()

    def reset(self) -> tuple[dict[str, np.ndarray], Union[dict[str, np.ndarray], None]]:
        '''
        Reset the environment to the initial state.
        # TODO: Info is not supported yet.

        :return: initial state and info.
        :rtype: tuple[dict[str, np.ndarray], Union[dict[str, np.ndarray], None]]
        '''
        # reset concat
        self.observation_concat_.reset()
        self.reward_concat_.reset()
        self.done_concat_.reset()
        self.truncated_concat_.reset()

        for env in self.envs_:  # Reset all the environments.
            next_observation, info = env.reset()
            self.observation_concat_.append(next_observation)

        # concatenate the data alone the num_envs axis.
        observation = self.observation_concat_.getConcatData(concat_axis=1)

        return observation, None

    def step(self, action: dict) -> tuple:
        '''
        Take an action and return the next state, reward, done, and info.

        # TODO: Currently, the info is not supported yet.

        :param action: The action to take. 
        :type action: dict{str: tensor or numpy.array} or tensor or numpy.array.

        :return: The next state, reward, done, truncated, and info.
        :rtype: dict{str: tensor or numpy.array}, dict{str: tensor or numpy.array}, tensor or numpy.array, dict{str: tensor or numpy.array}.
        '''

        # reset concat
        self.observation_concat_.reset()
        self.reward_concat_.reset()
        self.done_concat_.reset()
        self.truncated_concat_.reset()

        action: torch.Tensor = action['action']

        for env, act in zip(self.envs_, action):
            next_observation, reward, done, truncated, info = env.step(act)
            self.observation_concat_.append(next_observation)
            self.reward_concat_.append(reward)
            self.done_concat_.append(done)
            self.truncated_concat_.append(truncated)

        # concatenate the data alone the num_envs axis.
        next_observation = self.observation_concat_.getConcatData(
            concat_axis=1)
        reward_dict = self.reward_concat_.getConcatData()
        done = self.done_concat_.getConcatData()
        truncated = self.truncated_concat_.getConcatData()

        return next_observation, reward_dict, done, truncated, None
