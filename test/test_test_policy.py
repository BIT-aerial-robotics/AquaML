from abc import ABC

from AquaML.rlalgo.CompletePolicy import CompletePolicy
from AquaML.rlalgo.TestPolicy import TestPolicy
import tensorflow as tf
import gym
from AquaML.DataType import DataInfo
from AquaML.core.RLToolKit import RLBaseEnv
import numpy as np


class TD3Actor_net(tf.keras.Model):

    def __init__(self):
        super(TD3Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.action_layer = tf.keras.layers.Dense(1)
        # self.log_std = tf.keras.layers.Dense(1)

        # self.learning_rate = 2e-5

        self.output_info = {'action': (1,), }

        self.input_name = ('obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 3e-4,
                     # 'epsilon': 1e-5,
                     # 'clipnorm': 0.5,
                     },
        }

    @tf.function
    def call(self, obs, mask=None):
        x = self.dense1(obs)
        x = self.dense2(x)
        action = self.action_layer(x)
        # log_std = self.log_std(x)

        return (action,)

    def reset(self):
        pass

class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.action_layer = tf.keras.layers.Dense(4)
        # self.log_std = tf.keras.layers.Dense(1)

        # self.learning_rate = 2e-5

        self.output_info = {'action': (4,), }

        self.input_name = ('obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 3e-4,
                     # 'epsilon': 1e-54
                     # 'clipnorm': 0.5,
                     },
        }

    @tf.function
    def call(self, obs, mask=None):
        x = self.dense1(obs)
        x = self.dense2(x)
        action = self.action_layer(x)
        # log_std = self.log_std(x)

        return (action,)

    def reset(self):
        pass
# class Actor_net(tf.keras.Model):
#
#     def __init__(self):
#         super(Actor_net, self).__init__()
#
#         self.dense1 = tf.keras.layers.Dense(256, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(256, activation='relu')
#         self.action_layer = tf.keras.layers.Dense(4, activation='tanh')
#         # self.log_std = tf.keras.layers.Dense(1)
#
#         # self.learning_rate = 2e-5
#
#         self.output_info = {'action': (4,), }
#
#         self.input_name = ('obs',)
#
#         self.optimizer_info = {
#             'type': 'Adam',
#             'args': {'learning_rate': 1e-4,
#                      # 'epsilon': 1e-5,
#                      # 'clipnorm': 0.5,
#                      },
#         }
#
#     @tf.function
#     def call(self, obs, mask=None):
#         x = self.dense1(obs)
#         x = self.dense2(x)
#         action = self.action_layer(x)
#         # log_std = self.log_std(x)
#
#         return (action,)
#
#     def reset(self):
#         pass
# class Actor_net(tf.keras.Model):
#
#     def __init__(self):
#         super(Actor_net, self).__init__()
#
#         self.dense1 = tf.keras.layers.Dense(256, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(256, activation='relu')
#         self.action_layer = tf.keras.layers.Dense(4)
#         # self.log_std = tf.keras.layers.Dense(1)
#
#         # self.learning_rate = 2e-5
#
#         self.output_info = {'action': (4,), }
#
#         self.input_name = ('obs',)
#
#         self.optimizer_info = {
#             'type': 'Adam',
#             'args': {'learning_rate': 1e-3,
#                      # 'epsilon': 1e-5,
#                      # 'clipnorm': 0.5,
#                      },
#         }
#
#     @tf.function
#     def call(self, obs, mask=None):
#         x = self.dense1(obs)
#         x = self.dense2(x)
#         action = self.action_layer(x)
#         # log_std = self.log_std(x)
#
#         return (action,)
#
#     def reset(self):
#         pass


class BipedalWalker(RLBaseEnv):
    def __init__(self, env_name="BipedalWalker-v3"):
        super().__init__()
        # TODO: update in the future
        self.step_s = 0
        self.env = gym.make(env_name,hardcore=True, render_mode="human")
        self.env_name = env_name

        # our frame work support POMDP env
        self._obs_info = DataInfo(
            names=('obs', 'step',),
            shapes=((24,), (1,)),
            dtypes=np.float32
        )

        self._reward_info = ['total_reward', 'indicate_1']

    def reset(self):
        observation = self.env.reset()
        observation = observation[0].reshape(1, -1)

        self.step_s = 0
        # observation = observation.

        # observation = tf.convert_to_tensor(observation, dtype=tf.float32)

        obs = {'obs': observation, 'step': self.step_s}

        obs = self.initial_obs(obs)

        return obs, True  # 2.0.1 new version

    def step(self, action_dict):
        self.step_s += 1
        action = action_dict['action']
        if isinstance(action, tf.Tensor):
            action = action.numpy()
        # action *= 2
        observation, reward, done, tru, info = self.env.step(action[0])
        observation = observation.reshape(1, -1)

        indicate_1 = reward
        #
        if reward <= -100:
            reward = -1
            done = True

        obs = {'obs': observation, 'step': self.step_s}

        obs = self.check_obs(obs, action_dict)

        reward = {'total_reward': reward, 'indicate_1': indicate_1}

        # if self.id == 0:
        #     print('reward', reward)

        return obs, reward, done, info

    def close(self):
        self.env.close()


    # def seed(self, seed):
    #     gym


env = BipedalWalker()
osb_shape_dict = env.obs_info.shape_dict

policy = CompletePolicy(
    actor=Actor_net,
    obs_shape_dict=osb_shape_dict,
    checkpoint_path='TD3BC',
    using_obs_scale=False,
)

test_policy = TestPolicy(
    env=env,
    policy=policy,
)

test_policy.collect(
    episode_num=2000,
    episode_steps=1000,
    data_path='traj'
)
