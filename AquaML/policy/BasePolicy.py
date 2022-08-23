import tensorflow as tf
import abc


class BasePolicy(abc.ABC):
    def __init__(self, model: tf.keras.Model, name: str):
        self.model = model
        self.name = name

        self.target_model = None

    def set_target_model(self):
        """
        Create target model which weights equal to the initial self.model.

        :return: None
        """

        self.target_model = tf.keras.models.clone_model(self.model)

    def soft_update(self, tau):
        new_weights = []
        target_weight = self.target_model.weights

        for i, weight in enumerate(self.model.weights):
            new_weights.append(target_weight[i] * (1 - tau) + tau * weight)

        self.target_model.set_weights(new_weights)

    def save_weights(self, file_path, save_target=False):
        self.model.save_weights(file_path + '/' + self.name + '.h5', overwrite=True)
        if save_target:
            if self.target_model:
                self.target_model.save_weights(file_path + '/' + self.name + '_target.h5', overwrite=True)
            else:
                raise ValueError('Target model is not used.')

    def load_weights(self, file_path, load_target=False):
        self.model.load_weights(file_path + '/' + self.name + '.h5')
        if load_target:
            if self.target_model:
                self.model.load_weights(file_path + '/' + self.name + '_target.h5')
            else:
                raise ValueError('Target model is not used.')

    def save_model(self, file_path, save_target=False):
        tf.keras.models.save_model(self.model, file_path + '/' + self.name + '.h5')
        if save_target:
            if self.target_model:
                tf.keras.models.save_model(self.model, file_path + '/' + self.name + '_target.h5')
            else:
                raise ValueError('Target model is not used.')

    def load_model(self, file_path, load_targets=False):
        tf.keras.models.load_model(file_path + '/' + self.name + '.h5', compile=False)
        if load_targets:
            tf.keras.models.load_model(file_path + '/' + self.name + '_target.h5', compile=False)

    @property
    def get_variable(self):
        """"
        Get self.models train variable.
        """

        return self.model.trainable_variables

    # def __ceil__(self, *args):
    #
    #     return self.model(*args)
