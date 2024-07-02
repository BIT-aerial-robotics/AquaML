from AquaML.algo.AlgoBase import AlgoBase
from AquaML.algo.RLAlgoBase import RLAlgoBase
from AquaML.algo.ModelBase import ModelBase
from AquaML import logger, data_module,settings
import tensorflow as tf
import numpy as np


class TFAlgoBase(AlgoBase):
    """
    TFAlgoBase是基于TensorFlow的算法的基类。
    """
    
    def __init__(self,
                 hyper_params,
                 model_dict,
                 ):
        """
        TFAlgoBase的构造函数。
        
        Args:
            hyper_params (ParamBase): 超参数。
            model_dict (dict): 模型字典。
        """
        # super(TFAlgoBase, self).__init__(hyper_params, model_dict)
        pass
        
    
    def initialize_network(self, model:ModelBase):
        """

        初始化网络参数, 提前创建静态图。

        Args:
            model (_type_): _description_
        """

        input_data_names = model.input_names

        # create tensor according to input data name
        input_data = []

        for name in input_data_names:

            # 从data_module中获取数据信息
            data_info = data_module.query_data(
                name=name,
                set_name=self._algo_name
            )[0]
            
            data = tf.zeros(shape=data_info.shape, dtype=tf.float32)
            data = tf.expand_dims(data, axis=0)
            input_data.append(data)

        model(*input_data)
        
        logger.info('{} network initialized.'.format(model.name))
    
    def create_optimizer(self, model: ModelBase):

        opt_type = model.optimizer_type
        learning_rate = model.learning_rate
        args = model.optimizer_other_args

        optimizer = getattr(tf.keras.optimizers, opt_type)(learning_rate=learning_rate, **args)
        
        return optimizer
    
    def get_corresponding_data(self, 
                               data_dict: dict, 
                               names: tuple, 
                               prefix: str = '', 
                               concat_axis: int = None,
                               tf_tensor: bool = False,
                               dtype=tf.float32
                               ):
        """
        获取对应的数据。
        
        Args:
            data_dict (dict): 数据字典。
            names (tuple): 数据名称。
            prefix (str): 数据名称前缀。
            concat_axis (int): 拼接的轴。
            tf_tensor (bool): 是否返回tf tensor。
        """

        data = []

        for name in names:
            name = prefix + name
            buffer_ = data_dict[name]
            if concat_axis is not None:
                buffer = np.concatenate(buffer_, axis=concat_axis)
            else:
                buffer = buffer_
                
            if tf_tensor:
                tf_buffer = tf.convert_to_tensor(buffer, dtype=dtype)
                data.append(tf_buffer)
            else:
                data.append(buffer)

        return data
    
    def concat_list(self, data_list: list, axis: int = 0, tf_tensor: bool = False,dtype=tf.float32):
        """
        拼接列表。

        Args:
            data_list (list): 数据列表。
            axis (int, optional): 拼接的轴. Defaults to 0.
            tf_tensor (bool, optional): 是否返回tf tensor. Defaults to False.
        """
        
        ret_list = []
        
        for data in data_list:
            if tf_tensor:
                tf_data = tf.convert_to_tensor(np.concatenate(data, axis=axis), dtype=dtype)
                # tf_data = tf.concat(data, axis=axis)
                ret_list.append(tf_data)
            else:
                ret_list.append(np.concatenate(data, axis=axis))
        
        return ret_list
        
class TFRLAlgoBase(TFAlgoBase, RLAlgoBase):
    """
    TFRLAlgoBase是基于TensorFlow的RL算法的基类。
    """
    
    def __init__(self,
                 hyper_params,
                 model_dict,
                 ):
        """
        TFRLAlgoBase的构造函数。
        
        Args:
            hyper_params (RLParmBase): 超参数。
            model_dict (dict): 模型字典。
        """
        TFAlgoBase.__init__(self, hyper_params, model_dict)
        RLAlgoBase.__init__(self, hyper_params, model_dict)


    def init(self):
        self._action_size = settings.env_num

        for name, model in self._model_dict.items():
            
            # 初始化网络
            self.initialize_network(model)
            
            # 创建优化器
            setattr(self, name + '_optimizer', self.create_optimizer(model))
        
        
        
    

 