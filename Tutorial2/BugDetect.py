import sys

sys.path.append('..')
from AquaML.Tool import allocate_gpu
from mpi4py import MPI

comm = MPI.COMM_WORLD
allocate_gpu(comm, 0)
from AquaML.rlalgo.AqauRL import AquaRL
from AquaML.rlalgo.AgentParameters import PPOAgentParameter
from AquaML.rlalgo.PPOAgent import PPOAgent
import numpy as np
import gym
from AquaML.DataType import DataInfo
from AquaML.BaseClass import RLBaseEnv
import tensorflow as tf


class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.action_layer = tf.keras.layers.Dense(1, activation='tanh')
        # self.log_std = tf.keras.layers.Dense(1)

        self.learning_rate = 2e-4

        self.output_info = {'action': (1,), }

        self.input_name = ('obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 2e-4}
        }

    @tf.function
    def call(self, obs):
        x = self.dense1(obs)
        x = self.dense2(x)
        action = self.action_layer(x)

        return (action,)

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
        #     initial_learning_rate=0.3,
        #     decay_steps=10,
        #     decay_rate=0.95,
        #
        # )

        self.output_name = {'value': (1,)}

        self.input_name = ('obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 2e-4}
        }

    @tf.function
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

        # our frame work support POMDP env
        self._obs_info = DataInfo(
            names=('obs',),
            shapes=(3,),
            dtypes=np.float32
        )

        self._reward_info = ['total_reward', 'indicate_reward']

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

        reward = {'total_reward': (reward + 8) / 8, 'indicate_reward': reward}

        return obs, reward, done, info

    def close(self):
        self.env.close()


env = PendulumWrapper('Pendulum-v1')

parameters = PPOAgentParameter(
    rollout_steps=4000,
    epochs=100,
    batch_size=128,
    update_times=2,
    update_actor_times=4,
    update_critic_times=4,
    eval_episodes=5,
    eval_interval=10,
    eval_episode_length=200,
    entropy_coef=0.1,
    batch_advantage_normalization=True,
)

agent_info_dict = {
    'actor': Actor_net,
    'critic': Critic_net,
    'agent_params': parameters,
}

rl = AquaRL(
    env=env,
    agent=PPOAgent,
    agent_info_dict=agent_info_dict,
    # comm=comm,
    name='debug'
)

rl.run()
