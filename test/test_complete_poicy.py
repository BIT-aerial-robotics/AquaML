from AquaML.rlalgo.CompletePolicy import CompletePolicy
import tensorflow as tf


class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.action_layer = tf.keras.layers.Dense(1, activation='tanh')
        # self.log_std = tf.keras.layers.Dense(1, activation='tanh')

        # self.learning_rate = 2e-5

        self.output_info = {'action': (1,), }

        self.input_name = ('obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 2e-4,
                     # 'epsilon': 1e-5,
                     # 'clipnorm': 0.5,
                     },
        }

    @tf.function
    def call(self, obs):
        x = self.dense1(obs)
        x = self.dense2(x)
        action = self.action_layer(x)
        # log_std = self.log_std(x)*10

        return (action,)

    def reset(self):
        pass


obs_shape_dict = {'obs': (1, 3)}


policy = CompletePolicy(
    actor=Actor_net,
    obs_shape_dict=obs_shape_dict,
    checkpoint_path='cache',
    using_obs_scale=True,
)


print(policy.get_action({'obs': tf.constant([[1, 2, 3]], dtype=tf.float32)}))