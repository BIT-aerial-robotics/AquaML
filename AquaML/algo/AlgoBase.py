'''
规定算法的基类
'''

import abc
from AquaML.param.ParamBase import ParamBase
from AquaML import settings, logger, communicator
from AquaML.core.Recorder import Recorder
from AquaML.core.DataInfo import DataInfo
import numpy as np
from AquaML.Tool import LossTracker

class AlgoBase(abc.ABC):
    """
    这个版本将为所有的算法提供一个基类，用于规定算法的基本结构。
    """
    
    def __init__(self,
                 hyper_params:ParamBase,
                 model_dict:dict,
                 ):
        """
            所有算法的基类。所有的算法的输入可以分为两部分，一部分是超参数，一部分是模型。

            超参数是一个 ParamBase 对象，用于规定算法的超参数。
            model_dict 是一个字典，用于规定算法的模型。model_dict 的键是模型的名称，值是模型的实例。例如：

            .. code-block:: python
            {
                'actor': Model1:ModelBase,
                'critic': Model2:ModelBase,
            }

            model_dict 的键（key）必须和算法需要的模型名称一致。

            Args:
                hyper_params (ParamBase): 超参数对象，用于规定算法的超参数。
                model_dict (dict): 包含模型名称和模型实例的字典。
        """
        ############################
        # 1. 初始化参数
        ############################
        self._hyper_params = hyper_params
        
        
        # 重要参数，实现新算法时注意修改
        self._algo_name = None # 算法的名称
        
        #########################
        # 2. 接口类型数据
        #########################
        
        # 某些算法需要额外的数据信息，可以在这里添加
        self._algo_data_info = DataInfo() 
        self._model_dict = {} # 模型字典,在初始化时会进行一些操作
        
        ############################
        # 3. Wandb设置
        ############################
        
        # # 使用默认的wandb设置     
        # if settings.wandb_config is None:
        #     logger.info("Wandb config not set, use default config")
        #     hyper_params_dict = self._hyper_params.get_param_dict()
            
        #     # TODO:未来收集模型的学习率
        #     # 搜集模型的参数
        #     # model_params = {}
            
        #     # for model_name, model in model_dict.items():
        #     #     model_params[model_name+'_learning_rate'] = model.learning_rate
            
        #     wandb_config = {
        #         **hyper_params_dict,
        #         # **model_params
        #     }
        
        # # 使用用户设置的wandb设置
        # else:
        #     wandb_config = settings.wandb_config
        
        # if communicator.rank == 0:
        #     self._recorder = Recorder(
        #         wandb_project=settings.wandb_project,
        #         config=wandb_config
        #     )
        
        # communicator.Barrier()
        ############################
        # 4. 额外的工具包
        ############################
        # 设置LossTracker
        self.loss_tracker = LossTracker()
        
    
    ############################
    # 需要实现的接口
    ############################            
    @abc.abstractmethod
    def train(self,)->LossTracker:
        """
        算法的训练部分，所需要的数据从data_module中获取。
        """
    
    ############################
    # 取信息接口
    ############################
    
    @property
    def hyper_params(self)->ParamBase:
        """
        获取超参数。
        """
        return self._hyper_params
    
    @property
    def model_dict(self)->dict:
        """
        获取模型字典。
        """
        return self._model_dict
    
    @property
    def algo_data_info(self)->DataInfo:
        """
        获取算法数据信息。
        """
        return self._algo_data_info
    
    @property
    def algo_name(self)->str:
        """
        算法的名称。
        """
        # if self._algo_name is None:
        #     logger.error("Please check the algo_name in algo")
        return self._algo_name
    
    ############################
    # 功能性接口
    ############################
    def get_corresponding_data(self, 
                               data_dict: dict, 
                               names: tuple, 
                               prefix: str = '',
                               concat_axis: int = None,
                               ):
        """

        Get corresponding data from data dict.

        Args:
            data_dict (dict): data dict.
            names (tuple): name of data.
            prefix (str): prefix of data name.
            tf_tensor (bool): if return tf tensor.
        Returns:
            corresponding data. list or tuple.
        """

        data = []

        for name in names:
            name = prefix + name
            buffer_ = data_dict[name]
            if concat_axis is not None:
                buffer = np.concatenate(buffer_, axis=concat_axis)
            else:
                buffer = buffer_
            data.append(buffer)

        return data
    
    def concat_list(self, data_list:list, axis:int=0):
        """
        
        将输入的list中每个数据进行拼接。

        Args:
            data_list (list): list of data.
            axis (int, optional): axis to concat. Defaults to 0.
        """

        ret_list = []
        
        for data in data_list:
            ret_list.append(np.concatenate(data, axis=axis))
            
        return ret_list
    

