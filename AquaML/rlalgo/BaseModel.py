import abc
import tensorflow as tf

class BaseModel(abc.ABC,tf.keras.Model):
    """All the neral network models should inherit this class.
    """
    
    def __init__(self,rnn_flag:bool=False):
        super(BaseModel, self).__init__()
        self.rnn_flag = rnn_flag
    
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
        
        Return is dict.
        Such as:
        {'action':action, 'log_std':log_std}
        """