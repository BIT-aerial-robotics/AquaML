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

        # self._obs_info = {'obs': (3,)}
        # self.episode_length = 200

    def reset(self):
        observation = self.env.reset()
        observation = observation.reshape(1, -1)

        # observation = tf.convert_to_tensor(observation, dtype=tf.float32)

        # obs = {'obs': observation}
        obs = {'obs': observation, 'pos': observation[:, :2]}

        obs = self.initial_obs(obs)

        return obs, True

    def step(self, action_dict):
        action = action_dict['action']
        action *= 2
        observation, reward, done, info = self.env.step(action)
        observation = observation.reshape(1, -1)

        obs = {'obs': observation, 'pos': observation[:, :2]}

        obs = self.check_obs(obs, action_dict)

        # obs = {'obs': observation}
        reward = {'total_reward': reward}

        return obs, reward, done, info

    def close(self):
        return self.env.close()

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
