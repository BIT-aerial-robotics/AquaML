"""
Tutorial 2: How to use AquaML to train a RL agent with RNN policy
"""

import tensorflow as tf
from AquaML.rlalgo.SAC2 import SAC2  # SAC algorithm
from AquaML.rlalgo.Parameters import SAC2_parameter
from AquaML.starter.RLTaskStarter import RLTaskStarter  # RL task starter
from AquaML.Tool import GymEnvWrapper  # Gym environment wrapper
import tensorflow_probability as tfp
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[0]))
class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.lstm = tf.keras.layers.LSTM(32, input_shape=(2,), return_sequences=False, return_state=True)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.action_layer = tf.keras.layers.Dense(1, activation='tanh')
        self.log_std = tf.keras.layers.Dense(1)

        self.learning_rate = 2e-4

        self.output_info = {'action': (1,), 'log_std': (1,), 'hidden1': (32,), 'hidden2': (32,)}

        self.input_name = ('pos', 'hidden1', 'hidden2')

        self.optimizer = 'Adam'

    @tf.function
    def call(self, vel, hidden1, hidden2):
        hidden_states = (hidden1, hidden2)
        vel = tf.expand_dims(vel, axis=1)
        whole_seq, last_seq, hidden_state = self.lstm(vel, hidden_states)
        x = self.dense1(whole_seq)
        x = self.dense2(x)
        action = self.action_layer(x)
        log_std = self.log_std(x)

        return (action, log_std, last_seq, hidden_state)

    def reset(self):
        pass


class Q_net(tf.keras.Model):
    def __init__(self):
        super(Q_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense2 = tf.keras.layers.Dense(64, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense3 = tf.keras.layers.Dense(1, activation=None, kernel_initializer=tf.keras.initializers.orthogonal())

        # point out leaning rate
        # each model can have different learning rate
        self.learning_rate = 2e-3

        # point out optimizer
        # each model can have different optimizer
        self.optimizer = 'Adam'

        # point out input data name, this name must be contained in obs_info
        self.input_name = ('obs', 'action')

    def reset(self):
        # This model does not contain RNN, so this function is not necessary,
        # just pass

        # If the model contains RNN, you should reset the state of RNN
        pass

    @tf.function
    def call(self, obs, action):
        x = tf.concat([obs, action], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x

    # @property
    # def input_name(self):
    #     # although q function's input name is ('obs', 'action'),
    #     # the old action is not be used SAC, when training q function,
    #     return ('obs',)


Actor_net()
# create environment
env = GymEnvWrapper('Pendulum-v1')

# define the parameter of SAC

sac_parameter = SAC2_parameter(
    episode_length=200,
    n_epochs=200,
    batch_size=256,
    discount=0.99,
    tau=0.005,
    buffer_size=100000,
    mini_buffer_size=1000,
    update_interval=1,
    display_interval=1000,
    calculate_episodes=5,
    alpha_learning_rate=3e-3,
    update_times=1,
)

model_class_dict = {
    'actor': Actor_net,
    'qf1': Q_net,
    'qf2': Q_net,
}

starter = RLTaskStarter(
    env=env,
    model_class_dict=model_class_dict,
    algo=SAC2,
    algo_hyperparameter=sac_parameter,
    name=None
)

starter.run()
