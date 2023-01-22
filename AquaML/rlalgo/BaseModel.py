import abc
import tensorflow as tf

class BaseModel(abc.ABC,tf.keras.Model):
    """All the neral network models should inherit this class.
    
    The default optimizer is Adam. If you want to change it, you can set _optimizer.
    such as, self._optimizer = 'SGD'
    
    The learning rate is 0.001. If you want to change it, you can set _lr.
    """
    
    def __init__(self):
        super(BaseModel, self).__init__()
        self.rnn_flag = False
        self._optimizer = 'Adam'
        self._learning_rate = 0.001
    
    @abc.abstractmethod
    def reset(self):
        """
        Reset the model.
        Such as reset the rnn state.
        """
    
    @abc.abstractmethod
    def call(self, *args, **kwargs):
        """
        The call function of keras model.
        
        Return is tuple or tf.Tensor.
        
        When the model is actor, the return is tuple like (action, (h, c)).
        When the model is critic, the return is q_value or value.
        """
        
    @property
    def optimizer(self):
        return self._optimizer
    
    @property
    def learning_rate(self):
        return self._learning_rate