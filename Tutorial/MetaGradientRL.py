import sys
sys.path.append('..')
from AquaML.Tool import allocate_gpu
from mpi4py import MPI

# get group communicator
comm = MPI.COMM_WORLD
allocate_gpu(comm)


import tensorflow as tf
from AquaML.rlalgo.PPO import PPO  # SAC algorithm
from AquaML.rlalgo.Parameters import PPO_parameter
from AquaML.meta.MGRL import MGRL
from AquaML.meta.parameters import MetaGradientParameter
import gym
from AquaML.DataType import DataInfo
from AquaML.BaseClass import RLBaseEnv
import numpy as np


class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.action_layer = tf.keras.layers.Dense(1, activation='tanh')
        self.log_std = tf.keras.layers.Dense(1)

        self.learning_rate = 2e-4

        self.output_info = {'action': (1,), 'log_std': (1,)}

        self.input_name = ('obs',)

        self.optimizer = 'Adam'

    @tf.function
    def call(self, obs):
        x = self.dense1(obs)
        x = self.dense2(x)
        action = self.action_layer(x)
        log_std = self.log_std(x)

        return (action, log_std)

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

        self.input_name = ('obs', 'gamma', 'lambada')

        self.optimizer = 'Adam'

    def call(self, obs, gamma, lambada):
        x = tf.concat([obs, gamma, lambada], axis=1)

        x = self.dense1(x)
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

        # our frame work support POMDP env
        self._obs_info = DataInfo(
            names=('obs',),
            shapes=(3,),
            dtypes=np.float32
        )

        self.ratio = 1.0
        self.bias = 0.0

        self.meta_parameters = {
            'ratio': 1.0,
            'bias': 0.0,
        }

        self._reward_info = ('total_reward', 'indicate_reward')

        self.reward_fn_input = ('indicate_reward', 'ratio', 'bias')

    def reset(self):
        observation = self.env.reset()
        observation = observation.reshape(1, -1)

        # observation = tf.convert_to_tensor(observation, dtype=tf.float32)

        obs = {'obs': observation}

        obs = self.initial_obs(obs)

        return obs, True  # 2.0.1 new version

    def step(self, action_dict):
        action = action_dict['action']
        action *= 2
        observation, reward, done, info = self.env.step(action)
        observation = observation.reshape(1, -1)

        obs = {'obs': observation}

        obs = self.check_obs(obs, action_dict)

        reward_ = (reward + self.bias) * self.ratio

        reward = {'total_reward': reward_, 'indicate_reward': reward}

        return obs, reward, done, info

    def get_reward(self, indicate_reward, ratio, bias):
        new_reward = ratio * (indicate_reward + bias)
        return new_reward

    def close(self):
        self.env.close()


env = PendulumWrapper('Pendulum-v1')

support_env = PendulumWrapper('Pendulum-v1')

ppo_parameter = PPO_parameter(
    epoch_length=200,
    n_epochs=2000,
    total_steps=3000,
    batch_size=128,
    update_times=4,
    update_actor_times=4,
    update_critic_times=4,
    gamma=0.9,
    epsilon=0.2,
    lambada=0.95
)

meta_parameters = {
    'gamma': 0.99,
    'lambada': 0.95,
}

ppo_parameter.add_meta_parameters(meta_parameters)

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
    total_steps=200,
    batch_size=8,
    summary_episodes=10,
    multi_thread_flag=False,
)

meta_algo = MGRL(
    meta_core=PPO,
    core_hyperparameter=ppo_parameter,
    core_model_class_dict=model_class_dict,
    core_env=env,
    support_env=support_env,
    meta_parameter=meta_parameters,
    mpi_comm=comm,
    name='MGRL15',
)

meta_algo.run()
