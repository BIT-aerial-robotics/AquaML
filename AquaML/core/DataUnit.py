'''
单个数据集合，数据存储最小单元，每一个Unit包含唯一的数据类型，如numpy.ndarray

A single collection of data, the smallest unit of data, each containing a unique data type, such as numpy.ndarray
'''
try:
    from multiprocessing import shared_memory
except:
    # 输出红色文本
    print("\033[shared_memory is not supported in this version of python!\033[0m")
    print("\033[Please use python 3.8 or higher!\033[0m")
    print("\033[Or avoid using shared_memory!\033[0m")


import numpy as np
from AquaML.core.Protocol import data_unit_info_protocol
from AquaML.core.Tool import dtype2str

# from AquaML.communicator.CommunicatorBase import CommunicatorBase
from AquaML import logger
import time

# TODO:增加自动检测功能
def check_unit_info(unit_info) -> tuple:
    """
    检查unit_info是否合法
    
    Check whether unit_info is legal
    
    :param unit_info: dict type, unit_info must contain 'dtype', 'shape' and 'nbytes'
    """
    if not isinstance(unit_info, dict):
        raise TypeError('unit_info must be dict type')
    else:
        # 检测unit_info是否合法
        for key in unit_info:
            if key == 'nbytes':
                continue
            if key not in ['dtype', 'shape', 'size']:
                raise KeyError('unit_info must contain %s' % key)

        dtype = unit_info['dtype']

        if isinstance(dtype, str):
            dtype = eval(dtype)

        shape = unit_info['shape']

        size = unit_info['size']

        if isinstance(size, str):
            size = eval(size)

        if not isinstance(size, int):
            raise TypeError('size must be int type')

        if not isinstance(shape, tuple):
            raise TypeError("shape must be tuple type")

        size_shape = (size,) + shape

        return dtype, shape, size, size_shape


def compute_nbytes(dtype, shape) -> int:
    """
    计算数据大小
    
    Calculate data size
    
    :param dtype: data type
    :param shape: data shape
    """
    return np.dtype(dtype).itemsize * np.prod(shape)

# TODO: num_env*step*shape
class DataUnit:
    """
    数据单元，规定了数据单元的基本功能，包括数据的访问，写入，删除，修改等功能。 以及数据的基本属性，如数据类型，数据大小，数据维度等。
    
    每个unit由两个部分组成，一个是数据基本信息name_info，一个是数据内容name(_buffer),基本信息可以理解成解析协议，用于解析数据内容。
    
    我们使用SharedMemory来存储数据内容，使用ShareableList来存储数据基本信息。这样在每台机器中，数据都被注册上去，方便数据的访问。这有点类似ros，但是数据通信更高效。
    
    """

    def __init__(self, name: str,
                 unit_info: dict = None, 
                 exist: bool = False,
                 org_shape: tuple = None):
        """
        初始化数据单元，包括数据的基本信息，如数据类型，数据大小，数据维度等。
        
        Initialize the data unit, including the basic information of the data, such as the data type, data size, data dimension, etc.

        Args:
            name (str): 数据的唯一标识符号，用于数据的访问，写入，删除，修改等功能。
            unit_info (dict, optional): 数据基本信息，包括数据类型，数据大小，数据维度等。 Defaults to None. Template: {'dtype':np.float32, 'shape':(100, 100, 100), 'sie': 30}, shape tuple,describe the shape of data, size int, describe the size of data.
            exist (bool, optional): 是否为已存在的数据单元，如果是，则不需要创建共享内存，直接读取即可。 Defaults to False.
            org_shape (tuple, optional): 原始数据的shape，用于数据的还原。 Defaults to None.
        """
        # TODO: 优化非numpy数据类型的支持
        self._name = name  # 数据的唯一标识符号，

        # 数据基本信息解析协议
        logger.debug('Creating data unit {} ...'.format(name))

        self._info_protocol = data_unit_info_protocol  # shape tuple

        self.is_create_thread = False  # 是否为该数据块的创建者进程，用于关闭数据块

        # from AquaML.core.old.FileSystem import FileSystemBase
        # self._file_system:FileSystemBase = file_system
        self._shm_info = dict()
        

            

        if exist:
            # self._shm_info = shared_memory.ShareableList(name=name + '_info')
            self.read_shared_memory(unit_info)
            # self.read_unit_info()
        else:
            # 检测unit_info是否合法
            if unit_info is None:
                pass  # 支持先定义后续再添加数据格式
            else:
                self.create_shared_memory(unit_info)
                # self.write_unit_info()
                
        if org_shape is not None:
            self._org_shape = org_shape
        else:
            self._org_shape = self._shape
        

    def write_unit_info(self):
        """
        更新数据单元的信息，包括数据类型，数据大小，数据维度等。
        
        Update the information of the data unit, including data type, data size, data dimension, etc.
        """
        cache = dict()

        for key in self._info_protocol:
            if key == 'dtype':
                cache[key] = dtype2str(self.__getattribute__('_' + key))
            else:
                cache[key] = str(self.__getattribute__('_' + key))

            # length = len(self._shm_info)
        for key, value in cache.items():
            self._shm_info[key] = value
            # 更新yaml文件
        # self._file_system.write_data_unit_yaml(
        #     unit_name=self._name,
        #     unit_info=cache
        #     )
        
        logger.info('Update data unit info: %s' % self._name)


    # def read_unit_info(self):
    #     """
    #     读取数据单元的信息，包括数据类型，数据大小，数据维度等。
        
    #     Read the information of the data unit, including data type, data size, data dimension, etc.
    #     """

    #     # info_protocol_dict = dict(zip(self._info_protocol, self._shm_info))
        
    #     info_dict = self._file_system.read_data_unit_yaml(self._name)

    #     for key, value in info_dict.items():
    #         self.__setattr__('_' + key, eval(value))

    #     self._size_shape = (self._size,) + self._shape

    def create_shared_memory(self, unit_info: dict = None):
        """
        创建共享内存，用于存储数据内容。
        
        Create shared memory to store data content.
        
        :Args:
            unit_info (dict, optional): 数据基本信息，包括数据类型，数据大小，数据维度等。 Defaults to None.
        """
        if unit_info is not None:
            dtype, shape, size, size_shape = check_unit_info(unit_info)

            self._dtype = dtype

            self._shape = shape

            self._size = size

            self._size_shape = size_shape

            self._nbytes = compute_nbytes(dtype, size_shape)

        self._shm = shared_memory.SharedMemory(name=self._name, create=True, size=int(self._nbytes))

        self._buffer = np.ndarray(shape=self._size_shape, dtype=self._dtype, buffer=self._shm.buf)

        self.is_create_thread = True

    def read_shared_memory(self, unit_info: dict = None):
        """
        读取共享内存，返回数据内容。
        
        Read shared memory and return data content.
        
        :Args:
            unit_info (dict, optional): 数据基本信息，包括数据类型，数据大小，数据维度等。 Defaults to None.
    
        """

        if unit_info is not None:
            dtype, shape, size, size_shape = check_unit_info(unit_info)

            self._dtype = dtype

            self._shape = shape

            self._nbytes = compute_nbytes(dtype, shape)

            self._size = size

            self._size_shape = size_shape

        self._shm = shared_memory.SharedMemory(name=self._name, size=self._nbytes)

        self._buffer = np.ndarray(shape=self._size_shape, dtype=self._dtype, buffer=self._shm.buf)

    def get_data(self):
        """
        获取整个数据内容。
        """

        return self._buffer

    def set_data(self, data):
        """
        设置整个数据内容。
        """

        self._buffer[:] = data[:]

    def reset_zero(self):
        """
        将数据内容全部置为0。
        """

        a = np.zeros_like(self._buffer)

        self.set_data(a)

    def reset_true(self):
        """
        将数据内容全部置为True。
        """

        a = np.ones_like(self._buffer)
        a = a.astype(np.bool_)

        self.set_data(a)

    def reset_false(self):

        a = np.zeros_like(self._buffer)
        a = a.astype(np.bool_)

        self.set_data(a)

    ############################### 重载运算符 ################################   
    def __getitem__(self, index):
        """
        重载运算符，用于访问数据内容。

        Args:
            index (_type_): 

        Returns:
            _type_: array or value
        """
        return self._buffer[index]

    def __setitem__(self, index, value):
        """
        重载运算符，用于修改数据内容。

        Args:
            index (_type_): 
            value (_type_): 

        Returns:
            None
        """
        self._buffer[index] = value

    ############################### 属性 ################################
    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self):
        return self._size

    # @property
    def get_data_unit_info(self, str_all: bool = False) -> dict:
        """
        获取数据单元的基本信息，包括数据类型，数据大小，数据维度等。

        Args:
            str_all (bool, optional): 是否将数据类型，数据大小，数据维度等转换为str类型。 Defaults to False.

        Returns:
            dict: 数据单元的基本信息，包括数据类型，数据大小，数据维度等。
        """

        info = dict(zip(self._info_protocol, self._shm_info))
        info['full_name'] = self._name

        if str_all:
            dtype = info['dtype']

            if not isinstance(dtype, np.dtype):
                dtype = str(dtype)
            else:
                if dtype == np.float32:
                    dtype = 'np.float32'
                elif dtype == np.float64:
                    dtype = 'np.float64'
                elif dtype == np.int32:
                    dtype = 'np.int32'
                elif dtype == np.int64:
                    dtype = 'np.int64'
                elif dtype == np.uint8:
                    dtype = 'np.uint8'
                elif dtype == np.uint16:
                    dtype = 'np.uint16'
                elif dtype == np.bool_:
                    dtype = 'np.bool_'
            info['dtype'] = dtype

            info['shape'] = str(info['shape'])

        return info

    ############################### 方法 ################################

    def close(self):

        self.__del__()

    def __del__(self):
        # 释放共享内存，注意先后顺序
        if self.is_create_thread:
            time.sleep(2)
        self._shm.close()
        self._shm.unlink()
        # self._shm_info.shm.close()
        # self._shm_info.shm.unlink()
        # super().__del__()
