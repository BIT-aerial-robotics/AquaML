from AquaML.common.AlgoBase import AlgoBase
from AquaML.policy.VAEPolicy import VAEPolicy
import tensorflow as tf
import numpy as np


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


class VAE(AlgoBase):

    def _optimize(self, data_dict, args: dict):
        """"
        data_dict contains key: data
        """
        data = data_dict['data']
        # label = data_dict['label']
        with tf.GradientTape() as tape:
            z, mu, log_std = self.policy.encode(data)
            reconstruction = self.policy.decode(z)

            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction, labels=data)
            # cross_ent = data * -tf.math.log(reconstruction) + (1-data)*-tf.math.log(1-reconstruction)
            # print(cross_ent)
            # cross_ent = tf.keras.losses.binary_crossentropy(data, reconstruction, from_logits=True)
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
            logpz = log_normal_pdf(z, 0., 0.)
            logqz_x = log_normal_pdf(z, mu, log_std)
            loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)

        grad = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(list(zip(grad, self.policy.trainable_variables)))

        loss_info = {
            'loss': loss,
            'px_z': logpx_z,
            'pz': logpz,
            'qz_x': logqz_x
        }
        return loss_info

    def __init__(self, algo_args, data_collector, policy: VAEPolicy):
        super().__init__(algo_args, data_collector, policy)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=algo_args.learning_rate)
