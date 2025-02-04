import abc
from typing import TYPE_CHECKING

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
        :rtype: dict{str: tensor or numpy.array}, tensor or numpy.array, tensor or numpy.array, dict{str: tensor or numpy.array}.
        '''
        raise NotImplementedError

    def getObservationCfg(self) -> dict:
        '''
        Get the observation configuration.

        :return: The observation configuration.
        :rtype: dict.
        '''
        return self.observation_cfg_

    def getActionCfg(self) -> dict:
        '''
        Get the action configuration.

        :return: The action configuration.
        :rtype: dict.
        '''
        return self.action_cfg_
