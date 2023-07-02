import sys

sys.path.append('..')
from AquaML.Tool import allocate_gpu
# from mpi4py import MPI
#
# comm = MPI.COMM_WORLD
# allocate_gpu(comm, 0)
from AquaML.rlalgo.AqauRL import AquaRL
from AquaML.rlalgo.AgentParameters import PPOAgentParameter
from AquaML.rlalgo.PPOAgent import PPOAgent
import numpy as np
import gym
from AquaML.DataType import DataInfo
from AquaML.core.RLToolKit import RLBaseEnv
from AquaML.core.RLToolKit import RLVectorEnv
import tensorflow as tf


class SharedActorCritic(tf.keras.Model):

    def __init__(self):
        super(SharedActorCritic, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.action_layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.action_layer2 = tf.keras.layers.Dense(1, activation='tanh')
        self.value_layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.value_layer2 = tf.keras.layers.Dense(1, activation='linear')
        # self.log_std = tf.keras.layers.Dense(1, activation='tanh')

        # self.learning_rate = 2e-5

        self.output_info = {'action': (1,), 'value': (1,)}

        self.input_name = ('obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 2e-3,
                     # 'epsilon': 1e-5,
                     # 'clipnorm': 0.5,
                     },
        }

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        action_1 = self.action_layer1(x)
        action = self.action_layer2(action_1)
        value_1 = self.value_layer1(x)
        value = self.value_layer2(value_1)
        return (action, value,)


class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.action_layer = tf.keras.layers.Dense(1, activation='tanh')
        # self.log_std = tf.keras.layers.Dense(1, activation='tanh')

        # self.learning_rate = 2e-5

        self.output_info = {'action': (1,), }

        self.input_name = ('obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 2e-3,
                     # 'epsilon': 1e-5,
                     # 'clipnorm': 0.5,
                     },
        }

    @tf.function
    def call(self, obs):
        x = self.dense1(obs)
        x = self.dense2(x)
        action = self.action_layer(x)
        # log_std = self.log_std(x)*10

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

        self.output_info = {'value': (1,)}

        self.input_name = ('obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 2e-3,
                     # 'epsilon': 1e-5,
                     # 'clipnorm': 0.5,
                     }
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
    def __init__(self, env_name="Pendulum-v1"):
        super().__init__()
        # TODO: update in the future
        self.step_s = 0
        self.env = gym.make(env_name)
        self.env_name = env_name

        # our frame work support POMDP env
        self._obs_info = DataInfo(
            names=('obs', 'id', 'step',),
            shapes=((3,), (1,), (1,)),
            dtypes=np.float32
        )

        self._reward_info = ['total_reward', ]

    def reset(self):
        observation = self.env.reset()
        observation = observation[0].reshape(1, -1)

        self.step_s = 0
        # observation = observation.

        # observation = tf.convert_to_tensor(observation, dtype=tf.float32)

        obs = {'obs': observation, 'id': self.id, 'step': self.step_s}

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

        obs = {'obs': observation, 'id': self.id, 'step': self.step_s}

        obs = self.check_obs(obs, action_dict)

        reward = {'total_reward': reward}

        # if self.id == 0:
        #     print('reward', reward)

        return obs, reward, done, info

    def close(self):
        self.env.close()

    # def seed(self, seed):
    #     gym


eval_env = PendulumWrapper()

vec_env = RLVectorEnv(PendulumWrapper, 20, normalize_obs=False)
parameters = PPOAgentParameter(
    rollout_steps=200,
    epochs=200,
    batch_size=2000,
    update_times=4,
    max_steps=200,
    update_actor_times=1,
    update_critic_times=2,
    eval_episodes=20,
    eval_interval=10000,
    eval_episode_length=200,
    entropy_coef=0.1,
    batch_advantage_normalization=True,
    checkpoint_interval=10,
    min_steps=200,
)

agent_info_dict = {
    'actor': SharedActorCritic,
    'agent_params': parameters,
}

rl = AquaRL(
    env=vec_env,
    agent=PPOAgent,
    agent_info_dict=agent_info_dict,
    eval_env=eval_env,
    # comm=comm,
    name='debug',
    reward_norm=True,
    state_norm=True,
)

rl.run()
