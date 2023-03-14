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

"""
该教程展示了如何使用AquaML中使用learning to learn的方法来解决多臂赌博机问题。

Reference:
1. Reinforcement Learning, Fast and Slow, 2017, https://arxiv.org/abs/1703.07326
2. Learning to Learn by Gradient Descent by Gradient Descent, 2016, https://arxiv.org/abs/1606.04474
"""

num_of_bandits = 4


class Actor_net(tf.keras.Model):
    def __init__(self):
        super(Actor_net, self).__init__()

        self.lstm = tf.keras.layers.LSTM(64, input_shape=(3,), return_sequences=True, return_state=True)

        self.action_layer = tf.keras.layers.Dense(num_of_bandits, activation='softmax')

        self.learning_rate = 2e-5

        self.output_info = {'action': (1,), 'hidden1': (64,), 'hidden2': (64,)}  # 离散变量较为特殊，输出维度为1

        self.input_name = ('reward2', 'last_action', 'times', 'hidden1', 'hidden2')

        self.optimizer = 'Adam'

        self.rnn_flag = True

    @tf.function
    def call(self, reward2, last_action, times, hidden1, hidden2):
        x = tf.concat([reward2, last_action, times], axis=2)
        hidden_states = (hidden1, hidden2)
        whole_seq, last_seq, hidden_state = self.lstm(x, hidden_states)
        action = self.action_layer(whole_seq)
        action = tf.math.log(action)
        return (action, last_seq, hidden_state)

    def reset(self):
        pass


class Critic_net(tf.keras.Model):
    def __init__(self):
        super(Critic_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu', )

        self.value_layer = tf.keras.layers.Dense(1, activation=None)

        self.learning_rate = 2e-3

        self.output_name = {'value': (1,)}

        self.input_name = ('reward2', 'action', 'times', 'best_idx', 'best_pr', 'pr')

        self.optimizer = 'Adam'

    @tf.function
    def call(self, reward, action, times, best_idx, best_pr, pr):
        inputs = tf.concat([reward, action, times, best_idx, best_pr, pr], axis=1)
        x = self.dense1(inputs)
        value = self.value_layer(x)
        return value

    def reset(self):
        pass


env = BernoulliBandit(num_of_bandits=num_of_bandits, random_freq=5)

ppo_parameter = PPO_parameter(
    epoch_length=100,
    n_epochs=2000,
    total_steps=2000,
    batch_size=256,
    update_times=1,
    update_actor_times=1,
    update_critic_times=1,
    gamma=0.99,
    epsilon=0.01,
    lambada=0.95,
    action_space_type='discrete',
    eval_episodes=5,
    eval_interval=1,
    batch_trajectory=False
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
