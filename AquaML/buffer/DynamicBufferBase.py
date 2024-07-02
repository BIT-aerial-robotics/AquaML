'''

全动态缓冲区基类
'''

import abc
from AquaML.communicator.CommunicatorBase import CommunicatorBase
from AquaML.core.old.DataModule import DataModule
from AquaML.core.old.FileSystem import DefaultFileSystem
from AquaML.buffer.BufferBase import BufferBase
import numpy as np
import tensorflow as tf
import os


class DynamicBufferBase(BufferBase, abc.ABC):
    '''
    全动态缓冲区基类，和BufferBase的区别在于，这个类数据的存储和读取都是动态的。
    '''

    def __init__(self,
                 capacity: int,
                 data_names: list,
                 data_module: DataModule,
                 communicator: CommunicatorBase,
                 file_system: DefaultFileSystem,
                 ):
        '''
        初始化全动态缓冲区基类。
        
        该类会自动创建一个数据字典，字典里面是一个list

        Args:
            capacity (int): buffer的容量。
            data_names (list): 数据的名称列表。
            data_module (DataModule): 数据模块。用于数据的存储和读取。
            communicator (CommunicatorBase): 通信器。
        '''
        communicator.logger_info('DynamicBufferBase Init DynamicBufferBase')
        self._capacity = capacity
        self._data_module = data_module
        self._communicator = communicator
        self._data_names = data_names
        self._file_system = file_system

        self.capacity_count = 0

        self.data = {}

        for data_name in data_names:
            self.data[data_name] = []

    def append(self, data: dict):
        """
        Appends data to the buffer.

        所有的数据长度必须保持一致，并且一batch size的形式存储进去

        Args:
            data (dict): The data to be appended to the buffer.
        """

        # 数据处理
        data_dict = data

        # 数据存储
        if self.capacity_count < self._capacity:
            for data_name in self._data_names:
                self.data[data_name].append(data_dict[data_name])
                self._communicator.debug_info('BufferBase: Append {} data to buffer'.format(data_name))
            self.capacity_count += 1
            self._communicator.debug_info('BufferBase: Buffer capacity count: {}'.format(self.capacity_count))
        else:
            Index = self.capacity_count % self._capacity
            for data_name in self._data_names:
                self.data[data_name][Index] = data_dict[data_name]
                self._communicator.debug_info('BufferBase: Append {} data to buffer'.format(data_name))
            self.capacity_count += 1
            self._communicator.debug_info('BufferBase: Buffer capacity count: {}'.format(self.capacity_count))

    @abc.abstractmethod
    def sample_data(self, batch_size: int):
        """
        
        每一种buffer都应该实现这个函数，所有的算法在获取数据的时候都应该调用这个函数。
        
        函数返回的数据是一个字典，该字典会包含所有需要的数据。

        Args:
            batch_size (int): 采样的数据数量。

        Returns:
            dict: 该字典包含了采样的数据。
        """
