import sys

sys.path.append('..')
import tensorflow as tf

from AquaML.rlalgo.BehaviorCloning import BehaviorCloning
from AquaML.rlalgo.Parameters import BehaviorCloning_parameter
from AquaML.starter.RLTaskStarter import RLTaskStarter
import gym
from AquaML.DataType import DataInfo
from AquaML.BaseClass import RLBaseEnv
import numpy as np


class Expert_net(tf.keras.Model):

    def __init__(self):
        super(Expert_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.action_layer = tf.keras.layers.Dense(1, activation='tanh')

        self.learning_rate = 2e-4

        self.output_info = {'action': (1,), }

        self.input_name = ('obs',)

        self.optimizer = 'Adam'

        self.weight_path = 'actor.h5'

    @tf.function
    def call(self, obs):
        x = self.dense1(obs)
        x = self.dense2(x)
        action = self.action_layer(x)

        return (action,)

    def reset(self):
        pass


class Learner_net(tf.keras.Model):

    def __init__(self):
        super(Learner_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.action_layer = tf.keras.layers.Dense(1, activation='tanh')
        self.log_std = tf.keras.layers.Dense(1)

        self.learning_rate = 2e-4

        self.output_info = {'action': (1,), }

        self.input_name = ('obs',)

        self.optimizer = 'Adam'

    @tf.function
    def call(self, obs):
        x = self.dense1(obs)
        x = self.dense2(x)
        action = self.action_layer(x)

        return (action,)

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

        return obs, True # 2.0.1 new version

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

parameter = BehaviorCloning_parameter(
    epoch_length=200,
    n_epochs=200,
    batch_size=64,
    buffer_size=200,
)

model_class_dict = {
    'actor': Expert_net,
    'learner': Learner_net,
}

starter = RLTaskStarter(
    env=env,
    model_class_dict=model_class_dict,
    algo=BehaviorCloning,
    algo_hyperparameter=parameter,
)

starter.run()