# import sys
# sys.path.append('C:/Users/29184/Documents/GitHub/EvolutionRL')
import sys
sys.path.append('/Users/yangtao/Documents/code.nosync/EvolutionRL')

import abc
from AquaML.core.DataInfo import DataInfo

class ModelBase(abc.ABC):
    
    '''
    模型基类，用于规定模型的基本结构。
    
    使用方法：
    .. code-block:: python
    import tensorflow as tf
    class TestModel(ModelBase, tf.keras.Model):

        
        def __init__(self):
            ModelBase.__init__(self)
            tf.keras.Model.__init__(self)
            
            self.dense = tf.keras.layers.Dense(10)
            
        def call(self, inputs):
            return self.dense(inputs)
    '''
    
    def __init__(self,
                 ):
        
        self._learning_rate = 0.001
        self._optimizer_type = 'Adam'
        self._output_info:DataInfo = None
        self._input_names:tuple = None
        self._optimizer_other_args:dict = dict()
        
        
    ##############################
    # 通用接口
    ##############################
    @property
    def learning_rate(self):
        return self._learning_rate
    
    @property
    def output_info(self)->DataInfo:
        return self._output_info
    
    @property
    def input_names(self)->tuple:
        return self._input_names
    
    @property
    def optimizer_other_args(self)->dict:
        return self._optimizer_other_args

    @property
    def optimizer_type(self):
        return self._optimizer_type

if __name__ == '__main__':
    import tensorflow as tf
    class TestModel(ModelBase, tf.keras.Model):
        def __init__(self):
            ModelBase().__init__()
            tf.keras.Model.__init__(self)
            
            self.dense = tf.keras.layers.Dense(10)
            
        def call(self, inputs):
            return self.dense(inputs)
        
    test_model = TestModel()
    print(test_model.learning_rate)
    print(test_model.output_info)
    print(test_model.input_names)