import numpy as np

import gym
from AquaML.DataType import DataInfo
from AquaML.BaseClass import RLBaseEnv


class GymEnvWrapper(RLBaseEnv):
    def __init__(self, env_name: str):
        super().__init__()
        # TODO: update in the future
        self.env = gym.make(env_name)
        self.env_name = env_name
        self._obs_info = DataInfo(
            names=('obs',),
            shapes=((3,)),
            dtypes=np.float32
        )
        # self._obs_info = {'obs': (3,)}
        # self.episode_length = 200

    def reset(self):
        observation = self.env.reset()
        observation = observation.reshape(1, -1)

        # obs = {'obs': observation}
        obs = {'obs': observation}

        return obs

    def step(self, action):
        action = action * 2
        observation, reward, done, info = self.env.step(action)
        observation = observation.reshape(1, -1)

        obs = {'obs': observation, }
        # obs = {'obs': observation}
        reward = {'total_reward': reward}

        return obs, reward, done, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed):
        return self.env.seed(seed)

    def get_obs_info(self):
        return self.env.observation_space

    def get_action_info(self):
        return self.env.action_space

    def get_reward_info(self):
        return self.env.reward_range

    def get_info(self):
        return self.env.spec

    def get_name(self):
        return self.env_name

    def get_env(self):
        return self.env

    def get_env_info(self):
        return self.env.observation_space, self.env.action_space, \
            self.env.reward_range, self.env.spec, self.env_name

    def __del__(self):
        self.close()

    def __str__(self):
        return self.env_name

    def __repr__(self):
        return self.env_name
