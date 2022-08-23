from AquaML.policy.BasePolicy import BasePolicy
import tensorflow as tf


class CriticPolicy(BasePolicy):
    def __init__(self, model: tf.keras.Model, name: str):
        super().__init__(model=model, name=name)

    @tf.function
    def __call__(self, *args):
        return self.model(* args)
