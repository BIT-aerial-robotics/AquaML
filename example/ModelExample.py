import tensorflow as tf


class LSTMActor1(tf.keras.Model):
    def __init__(self, obs_dims=2):
        """
        An example for LSTM, this model can't be used in burn_in.
        :param obs_dims:
        """
        super(LSTMActor1, self).__init__()

        self.lstm = tf.keras.layers.LSTM(32, input_shape=(obs_dims,), return_sequences=True, return_state=True)

        self.actor_model = tf.keras.Sequential(
            [tf.keras.layers.Dense(64, activation='relu'),
             tf.keras.layers.Dense(64, activation='relu'),
             tf.keras.layers.Dense(1, activation='tanh')]
        )

        self.hidden_state = (tf.zeros(shape=(1, 32), dtype=tf.float32), tf.zeros(shape=(1, 32), dtype=tf.float32))

    @tf.function
    def call_actor(self, whole_seq, training=False):
        action = self.actor_model(whole_seq, training=training)
        return action

    # @tf.autograph.experimental.do_not_convert
    # def call(self, obs, training=False):
    #     # if obs.dims == 2:
    #     if training:
    #         whole_seq, last_seq, hidden_state = self.lstm(obs, training=training)
    #     else:
    #         whole_seq, last_seq, hidden_state = self.lstm(obs, self.hidden_state, training=training)
    #         self.hidden_state = (last_seq, hidden_state)
    #     #     obs = tf.expand_dims(obs, axis=0)
    #
    #     action = self.call_actor(whole_seq, training=training)
    #     # print(action)
    #
    #     return action

    def call(self, obs, training=False):
        hidden_state = self.hidden_state
        action, hidden_state = self.run_model(obs, hidden_state, training)

        if training:
            pass
        else:
            self.hidden_state = hidden_state

        return action

    @tf.function
    def run_model(self, obs, hidden_state, training=False):
        whole_seq, last_seq, hidden_state = self.lstm(obs, hidden_state, training=training)
        action = self.actor_model(whole_seq, training=training)

        return action, (last_seq, hidden_state)

    def reset(self, batch_size):
        self.hidden_state = (
            tf.zeros(shape=(batch_size, 32), dtype=tf.float32), tf.zeros(shape=(batch_size, 32), dtype=tf.float32))


if __name__ == "__main__":
    import numpy as np
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))

    actor = LSTMActor1(2)
    # actor.build(input_shape=(2,))
    actor(np.zeros((1, 1, 2)))
