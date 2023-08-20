from abc import ABC

from AquaML.rlalgo.CompletePolicy import CompletePolicy
from AquaML.rlalgo.TestPolicy import TestPolicy
import tensorflow as tf
import gym
from AquaML.DataType import DataInfo
from AquaML.core.RLToolKit import RLBaseEnv
import numpy as np


class FEnv(RLBaseEnv):

    def __init__(self):
        super().__init__()

        self.s_1 = 30
        self.s_2 = 100

        self._obs_info = DataInfo(
            names=('obs', ),
            shapes=((2,), ),
            dtypes=np.float32
        )

        self._reward_info = ['total_reward', ]

    def reset(self):
        self.s_1 = np.squeeze(np.random.randint(35, 65))

        self.s_2 = np.squeeze(np.random.randint(80, 100))

        obs = {
            'obs': np.array([self.s_1, self.s_2]).reshape((1,2))
        }

        return obs, True

    def step(self, action):

        action = action['action']

        if isinstance(action, tf.Tensor):
            action = action.numpy()

        action_ = np.reshape(action, (2,)) * 100

        self.s_1 = self.stochastic_internal_dynamic_model(x=self.s_1, h=action_[0])

        self.s_2 = self.stochastic_internal_dynamic_model(x=self.s_2, h=action_[1])

        reward = self.s_1 + self.s_2 - 1000 * (abs(self.s_1 - self.s_2))

        reward = {
            'total_reward':reward
        }

        obs = {
            'obs': np.array([self.s_1, self.s_2]).reshape((1,2))
        }

        return obs, reward, False, None

    def close(self):
        pass

    def stochastic_internal_dynamic_model(slef, x, h):
        alpha = 0.8
        r = 1
        i = 7
        w_n = 15
        w_s = 280
        a = (1 - alpha) * ((x + w_n) * (1 + r) + w_n)
        b = (1 - alpha) * (w_s + (x - h) * (1 + i))
        g = (1 - alpha) * (h * (1 + i) - w_s) / ((1 + i) * (1 - alpha) - 1)
        if a > b:
            x_next = (1 - alpha) * ((x + w_n) * (1 + r) + w_n)
        elif x > h:
            x_next = (1 - alpha) * (w_s + (x - h) * (1 + r))
        else:
            x_next = (1 - alpha) * (w_s + (x - h) * (1 + i))
        # print(g)
        x_1 = (1 - alpha) * w_n * (2 + r) / (1 - (1 - alpha) * (1 + r))
        # print(x_1)
        x_2 = (1 - alpha) * (w_s - h * (1 + r)) / (1 - (1 - alpha) * (1 + r))
        # print(x_2)
        if a > b:
            s = np.random.uniform(-10, 10, size=1)
            x_next = x_next + s
        else:
            s = np.random.uniform(-15, 15, size=1)
            x_next = x_next + s

        return x_next



class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.action_layer = tf.keras.layers.Dense(2, activation='tanh')
        # self.log_std = tf.keras.layers.Dense(1)

        # self.learning_rate = 2e-5

        self.output_info = {'action': (2,), }

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


env = FEnv()
osb_shape_dict = env.obs_info.shape_dict

policy = CompletePolicy(
    actor=Actor_net,
    obs_shape_dict=osb_shape_dict,
    checkpoint_path='PPO',
    using_obs_scale=False,
)

test_policy = TestPolicy(
    env=env,
    policy=policy,
)

test_policy.evaluate(
    episode_num=1,
    episode_steps=200,
    data_path='traj'
)
