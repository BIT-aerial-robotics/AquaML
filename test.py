import tensorflow as tf
import numpy as np


class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.lstm = tf.keras.layers.LSTM(32, input_shape=(2,), return_sequences=True, return_state=True)
        # if using batch time trajectory, return state must be True
        self.dense2 = tf.keras.layers.Dense(64)

        self.action_dense = tf.keras.layers.Dense(64)
        self.action_dense2 = tf.keras.layers.Dense(64)
        self.action_layer = tf.keras.layers.Dense(1, activation='tanh')

        self.value_dense = tf.keras.layers.Dense(64)
        self.value_dense2 = tf.keras.layers.Dense(64)
        self.value_layer = tf.keras.layers.Dense(1)

        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

        # self.learning_rate = self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=0.01,
        #     decay_steps=1,
        #     decay_rate=0.9,
        #
        # )

        self.learning_rate = 2e-3

        self.rnn_flag = True

        self.output_info = {'action': (1,), 'fusion_value': (1,), 'hidden1': (32,), 'hidden2': (32,)}

        self.input_name = ('pos', 'hidden1', 'hidden2')

        self.optimizer = 'Adam'

    # @tf.function
    def call(self, vel, hidden1, hidden2, mask=None):
        hidden_states = (hidden1, hidden2)
        whole_seq, last_seq, hidden_state = self.lstm(vel, hidden_states, mask=mask)
        x = self.dense2(whole_seq)
        x = self.leaky_relu(x)
        action_x = self.action_dense(x)
        action_x = self.leaky_relu(action_x)
        action_x = self.action_dense2(action_x)
        action_x = self.leaky_relu(action_x)
        action = self.action_layer(action_x)

        value_x = self.value_dense(x)
        value_x = self.leaky_relu(value_x)
        value_x = self.value_dense2(value_x)
        value_x = self.leaky_relu(value_x)
        value = self.value_layer(value_x)

        return (action, value, last_seq, hidden_state)

    def reset(self):
        pass




model = Actor_net()

a = tf.ones((1, 20, 2), dtype=tf.float32)

hidden1 = tf.ones((1, 32), dtype=tf.float32)
hidden2 = tf.ones((1, 32), dtype=tf.float32)
masks = np.zeros((1, 20))
tf_masks = tf.convert_to_tensor(masks, dtype=tf.bool)

model(a, hidden1, hidden2)


model(a, hidden1, hidden2, mask=tf_masks)