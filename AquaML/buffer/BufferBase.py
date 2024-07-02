import abc
from AquaML.communicator.CommunicatorBase import CommunicatorBase
from AquaML.core.old.DataModule import DataModule
import numpy as np
import tensorflow as tf
import os

# class BufferBase:
#     """
#     Buffer算法基类，用于定义Buffer算法的基本功能。
#     """
    
#     def __init__(self,
#                  capacity:int,
#                  data_names:list,
#                  data_module:DataModule,
#                  communicator:CommunicatorBase,
#                  shared_list=False
#                  ):
#         """
#         Buffer算法基类，用于定义Buffer算法的基本功能。

#         Args:
#             capacity (int): buffer的容量。
#             data_names (list): 数据的名称列表。
#             data_module (DataModule): 数据模块。用于数据的存储和读取。
#             communicator (CommunicatorBase): 通信模块。用于多进程通信以及log等。
#             shared_list (bool, optional): 是否共享数据。默认为False。未来支持的功能。
#         """
#         # TODO: 我们需要在合适的时候共享数据。
#         communicator.logger_info('BufferBase', 'Init BufferBase')
#         self._capacity = capacity
#         self._data_module = data_module
#         self._communicator = communicator
#         self._data_names = data_names
        
#         # 功能性参数
#         self.capacity_count = 0
        
#         # 创建存储数据的字典
#         # TODO: 
#         self.data = {}
        
#         # TODO:未来在这里添加共享列表操作
        
#         for data_name in data_names:
#             self.data[data_name] = []
        
#         # self.data_names = data_names
    
#     def append(self, data: dict):
#         """
#         Appends data to the buffer.

#         所有的数据长度必须保持一致，并且一batch size的形式存储进去

#         Args:
#             data (dict): The data to be appended to the buffer.
#         """

#         # 数据处理
#         data_dict = self._process_data(data)

#         # 数据存储
#         if self.capacity_count < self.capacity:
#             for data_name in self.data_names:
#                 self.data[data_name].append(data_dict[data_name])
#                 self._communicator.debug_info('BufferBase', 'Append {} data to buffer'.format(data_name))
#             self.capacity_count += 1
#             self._communicator.debug_info('BufferBase', 'Buffer capacity count: {}'.format(self.capacity_count))
#         else:
#             Index = self.capacity_count % self.capacity
#             for data_name in self.data_names:
#                 self.data[data_name][Index] = data_dict[data_name]
#                 self._communicator.debug_info('BufferBase', 'Append {} data to buffer'.format(data_name))
#             self.capacity_count += 1
#             self._communicator.debug_info('BufferBase', 'Buffer capacity count: {}'.format(self.capacity_count))
            
#     def load_npy(self, data_names: list, path: str):
#         """
#         从npy文件中加载数据。

#         Args:
#             data_names (list): 需要加载的数据名称。
#             path (str): npy文件的路径。
#         """
#         for data_name in data_names:
#             data = np.load(os.path.join(path, data_name + '.npy'))
#             self.data[data_name] = data
#             self._communicator.logger_info('BufferBase', 'Load {} data from {}'.format(data_name, path))

# TODO: 该函数效能较低，需要cuda和cpu不断交互数据，需要优化。
# 在jetson设备中，由于ram和dram是共享的，性能影响不高。                
class BufferBase(abc.ABC):
    """
    Buffer算法基类，用于定义Buffer算法的基本功能。
    """
    
    def __init__(self,
                 data_module:DataModule,
                 communicator:CommunicatorBase,
                 shared_list=False
                 ):
        """
        Buffer算法基类，用于定义Buffer算法的基本功能。

        Args:
            capacity (int): buffer的容量。
            data_module (DataModule): 数据模块。用于数据的存储和读取。
            communicator (CommunicatorBase): 通信模块。用于多进程通信以及log等。
            shared_list (bool, optional): 是否共享数据。默认为False。未来支持的功能。
        """
        
        communicator.logger_info('BufferBase: Init BufferBase')
        
        # 成员变量
        self._data_module = data_module
        self._communicator = communicator
        self._shared_list = shared_list
    
    
    def load_npy(self, data_names: list, path: str):
        """
        从npy文件中加载数据。

        Args:
            data_names (list): 需要加载的数据名称。这应该是一个包含字符串的列表，每个字符串都是一个npy文件的名称（不包括'.npy'扩展名）。
            path (str): npy文件的路径。

        Raises:
            FileNotFoundError: 如果任何npy文件不存在。
            ValueError: 如果任何npy文件无法加载。

        Notes:
            加载的数据将被存储在self.data字典中，其中键是数据名称，值是加载的数据。
        """
        for data_name in data_names:
            try:
                data = np.load(os.path.join(path, data_name + '.npy'))
            except FileNotFoundError:
                self._communicator.logger_error('BufferBase: File {} not found in {}'.format(data_name, path))
                raise
            except ValueError:
                self._communicator.logger_error('BufferBase: File {} could not be loaded from {}'.format(data_name, path))
                raise

            self.data[data_name] = data
            self._communicator.logger_info('BufferBase: Load {} data from {}'.format(data_name, path))
            
    ###############################功能函数################################

    # 获取数据的最大值
    def get_max(self, data_names: list)->list:
        """
        获取数据的最大值。

        Args:
            data_names (list): 需要获取最大值的数据名称。

        Returns:
            list: 该列表包含了每个数据名称对应的最大值。
        """
        
        max_data_list = []
        
        for data_name in data_names:
            max_data = np.max(self.data[data_name],axis=0)
            
            self._communicator.debug_info('BufferBase: Get {} max data.'.format(data_name))
        
            max_data_list.append(max_data)
            
        return max_data_list
    

    def get_min(self, data_names: list)->list:
        """
        获取数据的最小值。

        Args:
            data_names (list): 需要获取最小值的数据名称。

        Returns:
            list: 该列表包含了每个数据名称对应的最小值。
        """
        
        min_data_list = []
        
        for data_name in data_names:
            min_data = np.min(self.data[data_name],axis=0)
            self._communicator.debug_info('BufferBase: Get {} min data.'.format(data_name))
            min_data_list.append(min_data)
            
        return min_data_list
    
    def get_max_min_single(self, data_name: str):
        """
        获取数据的最大值和最小值。

        Args:
            data_name (str): 需要获取最大值和最小值的数据名称。

        Returns:
            list: 该列表包含了最大值和最小值。
        """
        
        max_data = np.max(self.data[data_name],axis=0)
        min_data = np.min(self.data[data_name],axis=0)
        
        self._communicator.logger_info('BufferBase: Get {} max and min data.'.format(data_name))
        
        return max_data, min_data

    # 最大值最小值归一化
    def min_max_scale(self, data_names:list):
        """
        将制定的数据进行最大最小值归一化。

        Args:
            data_names (list): 需要归一化的数据名称。
        """
        
        for data_name in data_names:
            max_data = np.max(self.data[data_name],axis=0)
            min_data = np.min(self.data[data_name],axis=0)
            self.data[data_name] = (self.data[data_name] - min_data) / (max_data - min_data)
            self._communicator.debug_info('BufferBase', 'Min max scale {} data.'.format(data_name))
    
    def get_corresponding_data(self, data_dict:dict,
                                data_names:list, 
                                prefix:str='', 
                                convert_to_tensor=True,
                                filter=[]
                                )->list:
        """
        根据采样的数据获取对应的数据。
        
        prefix用于添加前缀，convert_to_tensor用于将数据转换为tensor。当convert_to_tensor为True时，
        返回的数据会根据当前所采用的计算引擎进行转换。当前优先支持的计算引擎为tensorflow，后续逐渐添加其他
        计算引擎。
        
        filter用于过滤数据，当数据名称在filter中时，不会被返回。
        Args:
            data_dict (dict): 采样的数据。
            data_names (list): 需要获取的数据名称。
            prefix (str, optional): 数据名称的前缀。默认为''。
            convert_to_tensor (bool, optional): 是否将数据转换为tensor。默认为False。
            
        Returns:
            list: 该列表包含了对应的数据。并且数据的顺序和data_names一致。
        """
        
        corresponding_data_list = []
        
        for data_name in data_names:
            
            if data_name in filter:
                continue
            
            name = prefix + data_name
            
            try:
                data = data_dict[name]
            except KeyError:
                self._communicator.logger_error('BufferBase: Can not find {} in data_dict.'.format(name))
                raise RuntimeError('Can not find {} in data_dict.'.format(name))
            
            if convert_to_tensor:
                # convert to tensor
                # TODO:保持和原数据一样的长度
                data = tf.convert_to_tensor(data, dtype=tf.float32)
            
            corresponding_data_list.append(data)
            
        return corresponding_data_list
                  

    ###############################功能接口################################
    @abc.abstractmethod
    def sample_data(self, batch_size:int)->dict:
        """
        
        每一种buffer都应该实现这个函数，所有的算法在获取数据的时候都应该调用这个函数。
        
        函数返回的数据是一个字典，该字典会包含所有需要的数据。

        Args:
            batch_size (int): 采样的数据数量。

        Returns:
            dict: 该字典包含了采样的数据。
        """
        