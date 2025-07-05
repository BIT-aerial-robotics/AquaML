import abc
from AquaML.communicator.CommunicatorBase import CommunicatorBase
from AquaML.core.old.DataModule import DataModule
import keras
import numpy as np
from AquaML.algo_base.AlgoBase import AlgoBase
from AquaML.core.old.FileSystem import FileSystemBase


class PolicyCandidateBase(AlgoBase,abc.ABC):
    
    def __init__(self, 
                 name:str,
                 communicator:CommunicatorBase,
                 data_module:DataModule,
                 file_system:FileSystemBase
                 ):
        """
        Algo算法基类，用于定义Algo算法的基本功能。

        Args:
            name(str): 算法的名称。
            communicator (CommunicatorBase): 通信模块。用于多进程通信以及log等。由系统自动传入。
            data_module (DataModule): 数据模块。用于获取数据的shape等信息。由系统自动传入。
            file_system (FileSystem): 文件系统。用于文件的存储和读取。由系统自动传入。
        """
        super().__init__(name=name,
                         hyper_params=None,
                         communicator=communicator,
                         data_module=data_module,
                         file_system=file_system)
        
        # 读取接口中candidate数据的大小
        candidate_action_size = self._data_module.candidate_action.size
        self._candidate_action_size = candidate_action_size
        
        
    @abc.abstractmethod
    def select_action(self,):
        """
        根据输入的state选择action。

        Args:
            state (np.ndarray): 输入的state。

        Returns:
            np.ndarray: 选择的action。
        """
        pass
    
    ############################### 通用功能接口 ################################
    def get_corrosponding_state(self, data_unit_dict:dict, name_list:list)->list:
        """
        获取对应的state。
        

        Args:
            data_unit_dict (dict): 数据单元的字典。
            name_list (list): 需要获取的数据单元的名称。
        """
        
        return_list = []
        
        for name in name_list:
            try:
                data_unit = data_unit_dict[name]
                self._communicator.debug_debug('PolicyCandidateBase','Get data unit:{}'.format(name))
            except KeyError:
                self._communicator.debug_error('PolicyCandidateBase','Data unit:{} not found'.format(name))
                raise KeyError('Data unit:{} not found'.format(name))
            data = data_unit.get_data()
            
            # 对数据进行扩充，使得数据的维度和action的维度一致
            shape = data_unit.shape
            
            size_shape = [self._candidate_action_size, *list(shape)]
            
            extend_array = np.zeros(size_shape) 
            
            data = data * extend_array      
            
            
            return_list.append(data)
        
        return return_list