import sys

sys.path.append('..')
import tensorflow as tf
from AquaML.rlalgo.PPO import PPO  # SAC algorithm
from AquaML.rlalgo.Parameters import PPO_parameter
from AquaML.starter.RLTaskStarter import RLTaskStarter  # RL task starter
from AquaML.DataType import DataInfo
from AquaML.BaseClass import RLBaseEnv
import numpy as np
from AquaML.env.BernoulliBandit import BernoulliBandit

num_of_bandits = 10


class Actor_net(tf.keras.Model):
    def __init__(self):
        super(Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')

        self.action_layer = tf.keras.layers.Dense(num_of_bandits, activation='softmax')

        self.learning_rate = 2e-4

        self.output_info = {'action': (1,)}  # 离散变量较为特殊，输出维度为1

        self.input_name = ('reward2',)

        self.optimizer = 'Adam'

    # @tf.function
    def call(self, reward):
        x = self.dense1(reward)
        action = self.action_layer(x)
        return (action,)

    def reset(self):
        pass


class Critic_net(tf.keras.Model):
    def __init__(self):
        super(Critic_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu', )

        self.value_layer = tf.keras.layers.Dense(1, activation=None)

        self.learning_rate = 2e-3

        self.output_name = {'value': (1,)}

        self.input_name = ('reward2',)

        self.optimizer = 'Adam'

    # @tf.function
    def call(self, reward):
        x = self.dense1(reward)
        value = self.value_layer(x)
        return value

    def reset(self):
        pass


env = BernoulliBandit(num_of_bandits=num_of_bandits)

ppo_parameter = PPO_parameter(
    epoch_length=200,
    n_epochs=2000,
    total_steps=200,
    batch_size=128,
    update_times=4,
    update_actor_times=4,
    update_critic_times=4,
    gamma=0.99,
    epsilon=0.2,
    lambada=0.95,
    action_space_type='discrete',
    eval_episodes=1,
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
