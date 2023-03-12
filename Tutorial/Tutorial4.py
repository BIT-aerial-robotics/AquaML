"""
In this tutorial we will learn how to use the FusionPPO algorithm to train a RL agent with RNN policy.

The environment we use is the POMDP Pendulum-v1.
"""
import sys
sys.path.append('..')
from AquaML.Tool import allocate_gpu
from mpi4py import MPI

# get group communicator
comm = MPI.COMM_WORLD
allocate_gpu(comm)

import tensorflow as tf
from AquaML.DataType import DataInfo
from AquaML.BaseClass import RLBaseEnv
import numpy as np
import gym
from AquaML.rlalgo.FusionPPO import FusionPPO  # FusionPPO algorithm
from AquaML.rlalgo.Parameters import FusionPPO_parameter
from AquaML.starter.RLTaskStarter import RLTaskStarter  # RL task starter


class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.lstm = tf.keras.layers.LSTM(32, input_shape=(2,), return_sequences=True, return_state=True)
        # if using batch time trajectory, return state must be True
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')

        self.action_dense = tf.keras.layers.Dense(64, activation='relu')
        self.action_dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.action_layer = tf.keras.layers.Dense(1, activation='tanh')

        self.value_dense = tf.keras.layers.Dense(64, activation='relu')
        self.value_dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.value_layer = tf.keras.layers.Dense(1)

        self.learning_rate = 2e-3

        self.rnn_flag = True

        self.output_info = {'action': (1,), 'fusion_value': (1,), 'hidden1': (32,), 'hidden2': (32,)}

        self.input_name = ('pos', 'hidden1', 'hidden2')

        self.optimizer = 'Adam'

    @tf.function
    def call(self, vel, hidden1, hidden2):
        hidden_states = (hidden1, hidden2)
        whole_seq, last_seq, hidden_state = self.lstm(vel, hidden_states)
        x = self.dense2(whole_seq)
        action_x = self.action_dense(x)
        action_x = self.action_dense2(action_x)
        action = self.action_layer(action_x)
        value_x = self.value_dense(x)
        value_x = self.value_dense2(value_x)
        value = self.value_layer(value_x)

        return (action, value, last_seq, hidden_state)

    def reset(self):
        pass


class Critic_net(tf.keras.Model):
    def __init__(self):
        super(Critic_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense2 = tf.keras.layers.Dense(64, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense3 = tf.keras.layers.Dense(1, activation=None, kernel_initializer=tf.keras.initializers.orthogonal())

        self.learning_rate = 2e-3

        self.output_name = {'value': (1,)}

        self.input_name = ('obs',)

        self.optimizer = 'Adam'

    def call(self, obs):
        x = self.dense1(obs)
        x = self.dense2(x)
        value = self.dense3(x)

        return value

    def reset(self):
        pass


class PendulumWrapper(RLBaseEnv):
    def __init__(self, env_name: str):
        super().__init__()
        # TODO: update in the future
        self.env = gym.make(env_name)
        self.env_name = env_name

        self._reward_info = {'total_reward', 'indicate'}
        # our frame work support POMDP env
        self._obs_info = DataInfo(
            names=('obs', 'pos'),
            shapes=((3,), (2,)),
            dtypes=np.float32
        )

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
        reward = {'total_reward': (reward + 8) / 8, 'indicate': reward}

        return obs, reward, done, info

    def close(self):
        self.env.close()


env = PendulumWrapper('Pendulum-v1')

fusion_ppo_parameter = FusionPPO_parameter(
    epoch_length=200,
    n_epochs=2000,
    total_steps=6000,
    batch_size=20,
    update_times=4,
    update_actor_times=6,
    update_critic_times=6,
    gamma=0.99,
    epsilon=0.2,
    lambada=0.95,
    batch_trajectory=True
)

model_class_dict = {
    'actor': Actor_net,
    'critic': Critic_net,
}

starter = RLTaskStarter(
    env=env,
    model_class_dict=model_class_dict,
    algo=FusionPPO,
    algo_hyperparameter=fusion_ppo_parameter,
    name='FPPO2',
    mpi_comm=comm,
)

starter.run()
