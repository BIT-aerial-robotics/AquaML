from . import BaseEnv
from typing import Any
import gymnasium as gym
from typing import TYPE_CHECKING, Union
import numpy as np
from AquaML.data import unitCfg
from AquaML import coordinator


@coordinator.registerEnv
class GymnasiumWrapper(BaseEnv):
    '''
    The wrapper for Gymnasium environment.
    '''

    def __init__(self, env: Any):
        """Gymnasium environment wrapper

        This class is a wrapper for single Gymnasium environment.
        One instance run in one thread.

        :param env: The environment to be wrapped.
        :type env: str or Any supported Gymnasium environment
        """

        super().__init__()

        if isinstance(env, str):
            self.env = gym.make(env)
        else:
            self.env: gym.Env = env

        self.observation_cfg_ = {
            'state': unitCfg(
                name='state',
                dtype=np.float32,
                single_shape=self.env.observation_space.shape,
                size=1,
            ),
        }

        self.action_cfg_ = {
            'action': unitCfg(
                name='action',
                dtype=np.float32,
                single_shape=self.env.action_space.shape,
                size=1,
            ),
        }

        self.reward_cfg_ = {
            'reward': unitCfg(
                name='reward',
                dtype=np.float32,
                single_shape=(1,),
                size=1,
            ),
        }

        self.num_envs = 1  # The number of environments.

    def reset(self) -> tuple[dict[str, Any], Union[dict[str, np.ndarray], None]]:
        '''
        Reset the environment to the initial state.

        :return: initial state and info.
        :rtype: tuple[dict[str, Any], Union[dict[str, np.ndarray], None]]
        '''
        observation, info = self.env.reset()

        # 扩充observation的维度
        # (num_machines, num_envs, feature_dim)
        observation = np.expand_dims(observation, axis=[0, 1])

        observation_dict = {'state': observation}

        # Create AquaML data units for observations
        observation_units = self._create_data_units(observation_dict, "tensor")

        return observation_units, info

    def step(self, action: dict[str, np.ndarray]) -> tuple[dict[str, Any], Any, bool, bool, Any]:
        '''
        Take an action and return the next state, reward, done, and info.

        :param action: The action to take. 
        :type action: dict{str: np.ndarray}.

        :return: The next state, reward, done, truncated, and info.
        :rtype: tuple[dict{str: Any}, Any, bool, bool, Any]
        '''

        action = action['action']
        
        # Ensure action is properly shaped for the environment
        if isinstance(action, np.ndarray):
            if action.ndim > 1:
                action = action.squeeze()
            # For discrete environments, convert to int
            if hasattr(self.env.action_space, 'n'):  # Discrete action space
                action = int(action)
            # For environments expecting 1D arrays, ensure it's not a scalar
            elif action.ndim == 0:
                action = np.array([action])
        
        next_observation, reward, done, truncated, info = self.env.step(action)

        if done or truncated:
            next_observation, info = self.env.reset()

        next_observation = np.expand_dims(next_observation, axis=[0, 1])
        reward = np.array([[[reward]]])  # (1, 1, 1)
        done = np.expand_dims(done, axis=[0, 1])
        truncated = np.expand_dims(truncated, axis=[0, 1])

        reward_dict = {'reward': reward}

        next_observation_dict = {'state': next_observation}

        return next_observation_dict, reward_dict, done, truncated, info
    
    def _create_data_units(self, data_dict: dict[str, Any], mode: str = "tensor") -> dict[str, Any]:
        """Create data units from observation dictionary
        
        Args:
            data_dict: Dictionary containing observation data
            mode: Mode for data units ("tensor" or "numpy")
            
        Returns:
            Dictionary with processed data units
        """
        # For compatibility with existing code, just return the data_dict as-is
        # The data is already properly formatted in reset() and step() methods
        return data_dict
