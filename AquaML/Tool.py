import numpy as np

import gym
from AquaML.DataType import DataInfo
from AquaML.BaseClass import RLBaseEnv
import tensorflow as tf
import os


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

        # for rnn policy, we assume the hidden state of rnn is also the observation
        self.action_state_info = {}  # default is empty dict
        # self._obs_info = {'obs': (3,)}
        # self.episode_length = 200

    def set_action_state_info(self, actor_input_dict: dict):
        """
        set action state info.

        Example:
            >> action_state_dict = {'hidden_0':(256,), 'hidden_1':(256,),'action':(2,)}
            >> env.set_action_state_info(action_state_dict)
        """
        for key, shape in actor_input_dict.items():
            if 'hidden' in key:
                self.action_state_info[key] = shape

    def reset(self):
        observation = self.env.reset()
        observation = observation.reshape(1, -1)

        # observation = tf.convert_to_tensor(observation, dtype=tf.float32)

        # obs = {'obs': observation}
        obs = {'obs': observation}

        for key, shape in self.action_state_info.items():
            obs[key] = np.zeros(shape=shape, dtype=np.float32)

        return obs

    def step(self, action_dict):
        action = action_dict['action']
        action *= 2
        observation, reward, done, info = self.env.step(action)
        observation = observation.reshape(1, -1)

        obs = {'obs': observation, }

        for key in self.action_state_info.keys():
            obs[key] = action_dict[key]

        # obs = {'obs': observation}
        reward = {'total_reward': reward}

        return obs, reward, done, info

    # step for recurrent policy
    # def step_r(self, action_dict):
    #     action = action_dict['action']
    #     action *= 2
    #     observation, reward, done, info = self.env.step(action)
    #     observation = observation.reshape(1, -1)
    #
    #     obs = {'obs': observation, }
    #     for key, value in action_dict.items():
    #         if 'hidden' in key:
    #             obs[key] = value
    #     # obs = {'obs': observation}
    #     reward = {'total_reward': reward}
    #
    #     return obs, reward, done, info

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


def allocate_gpu(comm):
    rank = comm.Get_rank()
    if rank == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for k in range(len(physical_devices)):
                tf.config.experimental.set_memory_growth(physical_devices[k], True)
                print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
        else:
            print("Not enough GPU hardware devices available")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
