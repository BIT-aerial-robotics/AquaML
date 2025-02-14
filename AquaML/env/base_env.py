import abc
from typing import TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from AquaML.data import unitCfg


class BaseEnv(abc.ABC):
    '''
    Base class for all environments.
    The step function must contain the reset function.
    It means that the step function can automatically reset the environment when the episode ends.

    Some attributes are regulated as follows,

    >>> observation_cfg_  : The observation configuration.
    >>> action_cfg_       : The action configuration.
    >>> num_envs          : The number of environments.

    All data shape is regulated as follows,
    >>> (num_machines, num_envs, feature_dim) 
    -> num_machines: The number of machines used to sample the data.
    -> num_envs: The number of environments used to sample the data in one thread.
    -> feature_dim: The dimension of the feature.

    '''

    def __init__(self):
        '''
        Initialize the environment.
        '''

        self.observation_cfg_: dict = None  # The observation configuration.
        self.action_cfg_: dict = None  # The action configuration.
        self.reward_cfg_: dict = None  # The reward configuration.

        self.num_envs: int = 1  # The number of environments.

    @abc.abstractmethod
    def reset(self) -> tuple:
        '''
        Reset the environment to the initial state.

        :return: The initial state.
        :rtype: dict{str: tensor or numpy.array} or tensor or numpy.array.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action) -> tuple:
        '''
        Take an action and return the next state, reward, done, and info.

        :param action: The action to take. 
        :type action: tensor or dict{str: tensor}.

        :return: The next state, reward, done, truncated, and info.
        :rtype: dict{str: tensor or numpy.array}, dict{str: tensor or numpy.array}, tensor or numpy.array, dict{str: tensor or numpy.array}.
        '''
        raise NotImplementedError

    def getObservationCfg(self) -> dict:
        '''
        Get the observation configuration.

        :return: The observation configuration.
        :rtype: dict.
        '''
        if self.observation_cfg_ is None:
            logger.error('The observation configuration is not set yet.')
            raise ValueError('The observation configuration is not set yet.')
        return self.observation_cfg_

    def getActionCfg(self) -> dict:
        '''
        Get the action configuration.

        :return: The action configuration.
        :rtype: dict.
        '''
        if self.action_cfg_ is None:
            logger.error('The action configuration is not set yet.')
            raise ValueError('The action configuration is not set yet.')
        return self.action_cfg_

    def getEnvInfo(self) -> dict:
        '''
        Get the environment information.

        :return: The environment information.
        :rtype: dict.
        '''

        if self.observation_cfg_ is None:
            logger.error('The observation configuration is not set yet.')
            raise ValueError('The observation configuration is not set yet.')

        if self.action_cfg_ is None:
            logger.error('The action configuration is not set yet.')
            raise ValueError('The action configuration is not set yet.')

        observation_cfg_dict = {}

        for key, value in self.observation_cfg_.items():
            obs_cfg: unitCfg = value
            obs_dict = {}

            obs_dict['name'] = obs_cfg.name
            obs_dict['dtype'] = obs_cfg.dtype
            obs_dict['single_shape'] = obs_cfg.single_shape

            observation_cfg_dict[key] = obs_dict

        action_cfg_dict = {}

        for key, value in self.action_cfg_.items():
            act_cfg: unitCfg = value
            act_dict = {}

            act_dict['name'] = act_cfg.name
            act_dict['dtype'] = act_cfg.dtype
            act_dict['single_shape'] = act_cfg.single_shape

            action_cfg_dict[key] = act_dict

        env_info = {
            'observation_cfg': observation_cfg_dict,
            'action_cfg': action_cfg_dict,
            'num_envs': self.num_envs,
        }

        return env_info
