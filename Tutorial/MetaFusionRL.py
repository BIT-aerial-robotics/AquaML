import sys
sys.path.append('..')
from AquaML.Tool import allocate_gpu
from mpi4py import MPI

# get group communicator
comm = MPI.COMM_WORLD
allocate_gpu(comm)


import tensorflow as tf
from AquaML.rlalgo.FusionPPO import FusionPPO  # SAC algorithm
from AquaML.rlalgo.Parameters import FusionPPO_parameter
from AquaML.meta.MetaFusionPPO import MGRL
from AquaML.meta.parameters import MetaGradientParameter
import gym
from AquaML.DataType import DataInfo
from AquaML.BaseClass import RLBaseEnv
import numpy as np


class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.lstm = tf.keras.layers.LSTM(32, input_shape=(2,), return_sequences=True, return_state=True)
        # if using batch time trajectory, return state must be True
        self.dense2 = tf.keras.layers.Dense(64)

        self.action_dense = tf.keras.layers.Dense(64)
        self.action_dense2 = tf.keras.layers.Dense(64)
        self.action_layer = tf.keras.layers.Dense(1, activation='tanh')

        self.value_dense = tf.keras.layers.Dense(64)
        self.value_dense2 = tf.keras.layers.Dense(64)
        self.value_layer = tf.keras.layers.Dense(1)

        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

        # self.learning_rate = self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=0.01,
        #     decay_steps=1,
        #     decay_rate=0.9,
        #
        # )

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
        x = self.leaky_relu(x)
        action_x = self.action_dense(x)
        action_x = self.leaky_relu(action_x)
        action_x = self.action_dense2(action_x)
        action_x = self.leaky_relu(action_x)
        action = self.action_layer(action_x)

        value_x = self.value_dense(x)
        value_x = self.leaky_relu(value_x)
        value_x = self.value_dense2(value_x)
        value_x = self.leaky_relu(value_x)
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

        # self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=0.05,
        #     decay_steps=1,
        #     decay_rate=0.9,
        #
        # )

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
        # reward = {'total_reward': reward, 'indicate': reward}
        return obs, reward, done, info

    def close(self):
        self.env.close()


env = PendulumWrapper('Pendulum-v1')

support_env = PendulumWrapper('Pendulum-v1')

fusion_ppo_parameter = FusionPPO_parameter(
    epoch_length=200,
    n_epochs=2000,
    total_steps=6000,
    batch_size=20,
    update_times=4,
    update_actor_times=4,
    update_critic_times=4,
    gamma=0.99,
    epsilon=0.1,
    lambada=0.95,
    batch_trajectory=True,
    entropy_coeff=0.1,
    batch_advantage_normalization=True,
)


meta_parameters = {
    'gamma': 0.99,
    'lambada': 0.95,
}

fusion_ppo_parameter.add_meta_parameters(meta_parameters)

model_class_dict = {
    'actor': Actor_net,
    'critic': Critic_net,
}

meta_parameters = MetaGradientParameter(
    actor_ratio=1,
    critic_ratio=1,
    learning_rate=1e-3,
    max_epochs=100,
    max_steps=200,
    total_steps=1000,
    batch_size=200,
    summary_episodes=10,
    multi_thread_flag=False,
)

meta_algo = MGRL(
    meta_core=FusionPPO,
    core_hyperparameter=fusion_ppo_parameter,
    core_model_class_dict=model_class_dict,
    core_env=env,
    support_env=support_env,
    meta_parameter=meta_parameters,
    # mpi_comm=comm,
    name='MGRL',
)

meta_algo.run()
