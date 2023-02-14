"""
This tutorial is to show how to use AquaML to control pendulum-v0(https://gym.openai.com/envs/Pendulum-v0/).

The environment is a continuous action space environment. The action is a 1-dim vector. The observation is a 3-dim vector.

"""

# run your algo multi threads
import sys

sys.path.append('..')

from AquaML.Tool import allocate_gpu
from mpi4py import MPI

# get group communicator
comm = MPI.COMM_WORLD
allocate_gpu(comm)

import tensorflow as tf
from AquaML.rlalgo.SAC2 import SAC2  # SAC algorithm
from AquaML.rlalgo.Parameters import SAC2_parameter
from AquaML.starter.RLTaskStarter import RLTaskStarter  # RL task starter
from AquaML.Tool import GymEnvWrapper  # Gym environment wrapper
import tensorflow_probability as tfp

# class Actor_net(tf.keras.Model):
#     def __init__(self):
#         super(Actor_net, self).__init__()
#
#         self.dense1 = tf.keras.layers.Dense(64, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(64, activation='relu')
#         self.dense3 = tf.keras.layers.Dense(1, activation='tanh')
#         self.dense4 = tf.keras.layers.Dense(1)
#
#         # point out leaning rate
#         # each model can have different learning rate
#         self.learning_rate = 2e-4
#
#         # point out optimizer
#         # each model can have different optimizer
#         self.optimizer = 'Adam'
#
#         # point out input data name, this name must be contained in obs_info
#         self.input_name = ('obs',)
#
#         # actor network's output must be specified
#         # it is a dict, key is the name of output, value is the shape of output
#         # must contain 'action'
#         # self.output_info = {'action': (1,),'log_std': (1,)}
#         self.output_info = {'action': (1,)}
#
#         # Also, the model can control exploration policy by outputting a log_std etc.
#         # self._output_name = {'action': (1,), 'log_std': (1,)}
#
#     def reset(self):
#         # This model does not contain RNN, so this function is not necessary,
#         # just pass
#
#         # If the model contains RNN, you should reset the state of RNN
#         pass
#
#     @tf.function
#     def call(self, obs):
#         # print(obs)
#
#         x = self.dense1(obs)
#         x = self.dense2(x)
#         x = self.dense3(x)
#
#         # log_std = self.dense4(x)
#
#         # the output of actor network must be a tuple
#         # and the order of output must be the same as the order of output name
#
#         # return (x, log_std)
#         return (x,)


EPSILON = 1e-16


class Actor_net(tf.keras.Model):
    def __init__(self, n_hidden_layers=2, n_hidden_units=64, n_actions=1, logprob_epsilon=1e-7):
        super(Actor_net, self).__init__()
        self.logprob_epsilon = logprob_epsilon
        w_bound = 3e-3
        self.hidden = tf.keras.Sequential()

        for _ in range(n_hidden_layers):
            self.hidden.add(tf.keras.layers.Dense(n_hidden_units, activation="relu"))

        self.mean = tf.keras.layers.Dense(n_actions, activation=None, )

        self.stddev = tf.keras.layers.Dense(n_actions, activation=None, )

        # distribution
        mu = tf.zeros(n_actions)
        sigma = tf.ones(n_actions)
        self.normal_dist = tfp.distributions.Normal(mu, sigma)
        self.learning_rate = 2e-5
        self.optimizer = 'Adam'
        self.output_info = {'action': (1,), 'logprob': (1,)}
        self.input_name = ('obs',)

    @tf.function
    def call(self, inp):
        x = self.hidden(inp)
        mean = self.mean(x)
        # print(mean)
        std = self.stddev(x)
        sigma = tf.exp(std)
        # dist = tfp.distributions.Normal(mean, std)
        # normal_sample = dist.sample()
        noise, log_prob_ = self.noise_and_logprob(tf.shape(inp)[0])
        noise = tf.stop_gradient(noise)
        log_prob_ = tf.stop_gradient(log_prob_)
        normal_sample = mean + noise * sigma
        # log_prob = dist.log_prob(normal_sample)
        action = tf.tanh(normal_sample)
        log_prob = log_prob_ - tf.reduce_sum(tf.math.log(1 - tf.square(action) + self.logprob_epsilon), axis=1,
                                             keepdims=True)
        # print(action.numpy(), log_prob.numpy(), log_prob_.numpy())
        return action, log_prob

    def noise_and_logprob(self, batch_size):
        noise = self.normal_dist.sample(batch_size)
        prob = tf.clip_by_value(self.normal_dist.prob(noise), 0, 1)
        log_prob = tf.math.log(prob + self.logprob_epsilon)
        return noise, log_prob

    def reset(self):
        # This model does not contain RNN, so this function is not necessary,
        # just pass

        # If the model contains RNN, you should reset the state of RNN
        pass

    def _get_params(self):
        ''
        with self.graph.as_default():
            params = tf.trainable_variables()
        names = [p.name for p in params]
        values = self.sess.run(params)
        params = {k: v for k, v in zip(names, values)}
        return params

    def __getstate__(self):
        params = self._get_params()
        state = self.args_copy, params
        return state

    def __setstate__(self, state):
        args, params = state
        self.__init__(**args)
        self.restore_params(params)


# create Q network
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
    mini_buffer_size=5000,
    update_interval=1000,
    display_interval=1,
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
    # mpi_comm=comm,
    name=None
)

starter.run()
