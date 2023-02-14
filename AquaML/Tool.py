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
            names=('obs', 'pos'),
            shapes=((3,), (2,)),
            dtypes=np.float32
        )

        # for rnn policy, we assume the hidden state of rnn is also the observation
        self.action_state_info = {}  # default is empty dict
        # self._obs_info = {'obs': (3,)}
        # self.episode_length = 200

    def set_action_state_info(self, actor_out_info: dict, actor_input_name: tuple):
        """
        set action state info.
        Judge the input is as well as the output of actor network.

        """
        for key, shape in actor_out_info.items():
            if key in actor_input_name:
                self.action_state_info[key] = shape
                self._obs_info.add_info(key, shape, np.float32)

    def reset(self):
        observation = self.env.reset()
        observation = observation.reshape(1, -1)

        # observation = tf.convert_to_tensor(observation, dtype=tf.float32)

        # obs = {'obs': observation}
        obs = {'obs': observation, 'pos': observation[:, :2]}

        for key, shape in self.action_state_info.items():
            obs[key] = np.zeros(shape=shape, dtype=np.float32).reshape(1, -1)

        return obs

    def step(self, action_dict):
        action = action_dict['action']
        action *= 2
        observation, reward, done, info = self.env.step(action)
        observation = observation.reshape(1, -1)

        obs = {'obs': observation, 'pos': observation[:, :2]}

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
