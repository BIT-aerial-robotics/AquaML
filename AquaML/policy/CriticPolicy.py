from AquaML.policy.BasePolicy import BasePolicy


# import tensorflow as tf


class CriticPolicy(BasePolicy):
    def __init__(self, model, name: str, tf_handle, reset_flag=False):
        super().__init__(model=model, name=name, tf_handle=tf_handle, reset_flag=reset_flag)

        @tf_handle.function
        def call(*args):
            return self.model(*args)

        self.call = call

    def __call__(self, *args):
        return self.call(*args)
