from abc import ABC

from AquaML.rlalgo.CompletePolicy import CompletePolicy
from AquaML.rlalgo.TestPolicy import TestPolicy
import tensorflow as tf
import gym
from AquaML.DataType import DataInfo
from AquaML.core.RLToolKit import RLBaseEnv
import numpy as np


class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.lstm = tf.keras.layers.LSTM(64, input_shape=(2,), return_sequences=True, return_state=True)

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.action_layer = tf.keras.layers.Dense(1)
        # self.log_std = tf.keras.layers.Dense(1)

        # self.learning_rate = 2e-5

        self.output_info = {'action': (1,), 'hidden1': (64,), 'hidden2': (64,)}

        self.input_name = ('mask_obs', 'hidden1', 'hidden2')

        self.rnn_flag = True

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 2e-3,
                     'epsilon': 1e-5,
                     'clipnorm': 0.5,
                     },
        }

    @tf.function
    def call(self, obs, hidden1, hidden2, mask=None):
        hidden_states = (hidden1, hidden2)
        whole_seq, last_seq, hidden_state = self.lstm(obs, hidden_states, mask=mask)
        x = self.dense1(whole_seq)
        x = self.dense2(x)
        action = self.action_layer(x)
        # log_std = self.log_std(x)

        return (action, last_seq, hidden_state,)

    def reset(self):
        pass


class PendulumWrapper(RLBaseEnv):
    def __init__(self, env_name="Pendulum-v1"):
        super().__init__()
        # TODO: update in the future
        self.step_s = 0
        self.env = gym.make(env_name)
        self.env_name = env_name

        # our frame work support POMDP env
        self._obs_info = DataInfo(
            names=('obs', 'mask_obs', 'step',),
            shapes=((3,), (2,), (1,),),
            dtypes=np.float32
        )

        self._reward_info = ['total_reward', ]

    def reset(self):
        observation = self.env.reset()
        observation = observation[0].reshape(1, -1)

        self.step_s = 0
        # observation = observation.

        # observation = tf.convert_to_tensor(observation, dtype=tf.float32)

        obs = {'obs': observation, 'step': self.step_s, 'mask_obs': observation[:, :2]}

        obs = self.initial_obs(obs)

        return obs, True  # 2.0.1 new version

    def step(self, action_dict):
        self.step_s += 1
        action = action_dict['action']
        if isinstance(action, tf.Tensor):
            action = action.numpy()
        action *= 2
        observation, reward, done, tru, info = self.env.step(action)
        observation = observation.reshape(1, -1)

        obs = {'obs': observation, 'step': self.step_s, 'mask_obs': observation[:, :2]}

        obs = self.check_obs(obs, action_dict)

        reward = {'total_reward': reward}

        # if self.id == 0:
        #     print('reward', reward)

        return obs, reward, done, info

    def close(self):
        self.env.close()


env = PendulumWrapper()
obs_shape_dict = {'mask_obs': (1, 2)}

policy = CompletePolicy(
    actor=Actor_net,
    obs_shape_dict=obs_shape_dict,
    checkpoint_path='cache/PPO',
    using_obs_scale=True,
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