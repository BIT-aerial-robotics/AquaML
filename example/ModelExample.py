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


class LSTMActorValue1(tf.keras.Model):
    def __init__(self, obs_dims=2):
        """
        PPG joint model example.

        :param obs_dims:
        """
        super(LSTMActorValue1, self).__init__()

        self.lstm = tf.keras.layers.LSTM(32, input_dim=(obs_dims,), return_sequences=True, return_state=True)

        self.share_dense = tf.keras.layers.Dense(64, activation="relu")

        self.actor_model = tf.keras.Sequential(
            [tf.keras.layers.Dense(64, activation='relu'),
             tf.keras.layers.Dense(1, activation='tanh')]
        )

        self.value_model = tf.keras.Sequential(
            [tf.keras.layers.Dense(64, activation='tanh'),
             tf.keras.layers.Dense(1)]
        )

        self.hidden_state = (tf.zeros(shape=(1, 32), dtype=tf.float32), tf.zeros(shape=(1, 32), dtype=tf.float32))

    def call(self, obs, training=False):
        """
        must have
        :param obs:
        :param training:
        :return:
        """
        hidden_state = self.hidden_state
        action, hidden_state = self.run_actor_model(obs, hidden_state, training)

        if training:
            pass
        else:
            self.hidden_state = hidden_state

        return action

    # def build(self, input_shape):
    #     """
    #     must have
    #     :param input_shape:
    #     :return:
    #     """
    #     inputs = tf.zeros(shape=input_shape, dtype=tf.float32)
    #     whole_seq, last_seq, hidden_state = self.lstm(inputs)
    #     self.actor_model(whole_seq)
    #     self.value_model(whole_seq)

    def actor_critic(self, obs, training=False):
        """
        Must have.

        :param obs:
        :param training:
        :return:
        """
        hidden_state = self.hidden_state
        action, value, hidden_state = self.run_actor_value_model(obs, hidden_state, training)

        if training:
            pass
        else:
            self.hidden_state = hidden_state

        return action, value

    @tf.function
    def run_actor_value_model(self, obs, hidden_state, training=False):
        whole_seq, last_seq, hidden_state = self.lstm(obs, hidden_state, training=training)
        out = self.share_dense(whole_seq, training=training)
        action = self.actor_model(out, training=training)
        value = self.value_model(out, training=training)

        return action, value, (last_seq, hidden_state)

    @tf.function
    def run_actor_model(self, obs, hidden_state, training=False):
        whole_seq, last_seq, hidden_state = self.lstm(obs, hidden_state, training=training)
        out = self.share_dense(whole_seq, training=training)
        action = self.actor_model(out, training=training)
        self.value_model(out, training=False)

        return action, (last_seq, hidden_state)

    # @property
    # def trainable_variables(self):
    #     """
    #     Must have.
    #     :return:
    #     """
    #     return self.lstm.trainable_variables + self.actor_model.trainable_variables

    # @property
    # def get_all_variable(self):
    #     """
    #     Must have
    #     :return:
    #     """
    #     return self.lstm.trainable_variables + self.actor_model.trainable_variables + self.value_model.trainable_variables

    def reset(self, batch_size):
        """
        must have
        :param batch_size:
        :return:
        """
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

    actor = LSTMActorValue1(2)
    actor.build((1, 1, 2))
    actor.save_weights('test.h5', overwrite=True)
    # actor.build(input_shape=(2,))
    # print(actor(np.zeros((1, 1, 2))))
    # print(actor.get_actor_critic_variable)
