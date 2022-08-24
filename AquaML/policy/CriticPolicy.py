from AquaML.policy.BasePolicy import BasePolicy
import tensorflow as tf


class CriticPolicy(BasePolicy):
    def __init__(self, model: tf.keras.Model, name: str, reset_flag=False):
        super().__init__(model=model, name=name, reset_flag=reset_flag)

    @tf.function
    def __call__(self, *args):
        return self.model(*args)
