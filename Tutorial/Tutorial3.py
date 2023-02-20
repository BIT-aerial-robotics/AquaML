import tensorflow as tf
from AquaML.rlalgo.PPO import PPO  # SAC algorithm
from AquaML.rlalgo.Parameters import PPO_parameter
from AquaML.starter.RLTaskStarter import RLTaskStarter  # RL task starter
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

        self.learning_rate = 2e-4

        self.output_info = {'action': (1,), }

        self.input_name = ('obs',)

        self.optimizer = 'Adam'

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

        # our frame work support POMDP env
        self._obs_info = DataInfo(
            names=('obs',),
            shapes=(3,),
            dtypes=np.float32
        )

    def reset(self):
        observation = self.env.reset()
        observation = observation.reshape(1, -1)

        # observation = tf.convert_to_tensor(observation, dtype=tf.float32)

        obs = {'obs': observation}

        obs = self.initial_obs(obs)

        return obs

    def step(self, action_dict):
        action = action_dict['action']
        action *= 2
        observation, reward, done, info = self.env.step(action)
        observation = observation.reshape(1, -1)

        obs = {'obs': observation}

        obs = self.check_obs(obs, action_dict)

        reward = {'total_reward': reward}

        return obs, reward, done, info

    def close(self):
        self.env.close()


env = PendulumWrapper('Pendulum-v1')
ppo_parameter = PPO_parameter(
    epoch_length=200,
    n_epochs=2000,
    total_steps=4000,
    batch_size=128,
    update_times=1,
    update_actor_times=1,
    update_critic_times=1,
    gamma=0.99,
    epsilon=0.2,
    lambada=0.95
)

model_class_dict = {
    'actor': Actor_net,
    'critic': Critic_net,
}

starter = RLTaskStarter(
    env=env,
    model_class_dict=model_class_dict,
    algo=PPO,
    algo_hyperparameter=ppo_parameter,
)

starter.run()
