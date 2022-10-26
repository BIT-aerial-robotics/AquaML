import tensorflow as tf


class VAEPolicy(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(VAEPolicy, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        mu, log_std = self.encoder(inputs, training=training)
        epsilon = tf.random.normal(shape=mu.shape)
        z = mu + epsilon * tf.exp(log_std)

        x = self.decoder(z)

        return x, mu, log_std

    @tf.function
    def encode(self, inputs):
        mu, log_std = self.encoder(inputs)
        epsilon = tf.random.normal(shape=mu.shape)
        z = mu + epsilon * tf.exp(log_std)
        return z, mu, log_std

    @tf.function
    def decode(self, inputs):
        return self.decoder(inputs)
