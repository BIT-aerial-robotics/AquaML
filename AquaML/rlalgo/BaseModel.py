import abc
import tensorflow as tf

class BaseModel(abc.ABC,tf.keras.Model):
    """All the neral network models should inherit this class.
    
    The default optimizer is Adam. If you want to change it, you can set _optimizer.
    such as, self._optimizer = 'SGD'
    
    The learning rate is 0.001. If you want to change it, you can set _lr.
    
    If the model is Q network, _input_name should not contain 'action'
    
    You can specify the input name by set self._input_name.
    If not set, the input name is determined by the RLIOInfo(AquaML.DataTypes.RLIOInfo).
    """
    
    def __init__(self):
        super(BaseModel, self).__init__()
        self.rnn_flag = False
        self._optimizer = 'Adam'
        self._learning_rate = 0.001
        
        self._input_name = None
    
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
    
    @property
    def input_name(self):
        return self._input_name