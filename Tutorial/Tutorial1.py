"""
This tutorial is to show how to use AquaML to control pendulum-v0(https://gym.openai.com/envs/Pendulum-v0/).

The environment is a continuous action space environment. The action is a 1-dim vector. The observation is a 3-dim vector.

"""
import tensorflow as tf
from AquaML.rlalgo.SAC2 import SAC2  # SAC algorithm
from AquaML.rlalgo.Parameters import SAC2_parameter
from AquaML.starter.RLTaskStarter import RLTaskStarter  # RL task starter
from AquaML.Tool import GymEnvWrapper  # Gym environment wrapper


# create actor network

class Actor_net(tf.keras.Model):
    def __init__(self):
        super(Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='tanh')
        self.dense4 = tf.keras.layers.Dense(1)

        # point out leaning rate
        # each model can have different learning rate
        self.learning_rate = 2e-4

        # point out optimizer
        # each model can have different optimizer
        self.optimizer = 'Adam'

        # point out input data name, this name must be contained in obs_info
        self.input_name = ('obs',)

        # actor network's output must be specified
        # it is a dict, key is the name of output, value is the shape of output
        # must contain 'action'
        self.output_info = {'action': (1,),}

        # Also, the model can control exploration policy by outputting a log_std etc.
        # self._output_name = {'action': (1,), 'log_std': (1,)}

    def reset(self):
        # This model does not contain RNN, so this function is not necessary,
        # just pass

        # If the model contains RNN, you should reset the state of RNN
        pass

    @tf.function
    def call(self, obs):
        # print(obs)

        x = self.dense1(obs)
        x = self.dense2(x)
        x = self.dense3(x)

        # log_std = self.dense4(x)

        # the output of actor network must be a tuple
        # and the order of output must be the same as the order of output name

        return (x, )


# create Q network
class Q_net(tf.keras.Model):
    def __init__(self):
        super(Q_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation=None)

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
    epoch_length=200,
    n_epochs=1000,
    batch_size=256,
    discount=0.99,
    alpha=0.2,
    tau=0.005,
    buffer_size=20*200,
    mini_buffer_size=1000,
    update_interval=1,
    display_interval=1000,
    calculate_episodes=5,
    alpha_learning_rate=3e-4,
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
)

starter.run()
