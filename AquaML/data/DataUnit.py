import numpy as np
from multiprocessing import shared_memory
import copy
import warnings
import os
import json


class DataUnit:
    """
    The smallest data storage unit. It can be used in HPC (MPI) and 
    shared memory system.

    It can load from exit numpy array for data set learning.
    """

    def __init__(self, name: str, shape=None, dtype=np.float32, computer_type='PC', dataset: np.ndarray = None,
                 level=None):
        """Create data unit. If shape is not none, this data unit is used in main thread.
        The unit is created depend on your computer type. If you use in high performance
        computer(HPC), shared memmory isn't used.

        Args:
            name (str): _description_
            shape (tuple, None): shape of data unit.If shape is none, the unit is in sub thread. 
            (buffersize, dims)
             Defaults to None.
            dtype (nd.type): type of this data unit. Defaults to np.float32.
            computer_type(str):When computer type is 'PC', mutlti thread is based on shared memory.Defaults to 'PC'.
            dataset(np.ndarry, None): create unit from dataset. Defaults to None
            level(int, None): Clarify level.

        """

        self.name = name

        self._shape = shape
        self._dtype = dtype

        self._computer_type = computer_type

        if shape is not None:
            self.level = 0
            self._buffer = np.zeros(shape=shape, dtype=self._dtype)
            self.__nbytes = self._buffer.nbytes
        else:
            self.level = 1
            self._buffer = None
            self.__nbytes = None

        if dataset is not None:
            self.copy_from_exist_array(dataset)

        if level is not None:
            self.level = level

        self.shm_buffer = None

    def copy_from_exist_array(self, dataset: np.ndarray, level: int = 0):
        """Copy data from exist np array.

        Args:
            dataset (np.ndarray): Dataset.
            level (int): Thread level. Defaults to 0.
        """
        self._buffer = copy.deepcopy(dataset)
        del dataset
        self.level = level
        self.__nbytes = self._buffer.nbytes
        self._dtype = self._buffer.dtype

    def create_shared_memory(self):
        """Create shared-memory.
        """
        if self._computer_type == 'HPC':
            warnings.warn("HPC can't support shared memory!")

        if self.level == 0:
            self.shm_buffer = shared_memory.SharedMemory(create=True, size=self.__nbytes, name=self.name)
            self._buffer = np.ndarray(self._shape, dtype=self._dtype, buffer=self.shm_buffer.buf)
        else:
            raise Exception("Current thread is sub thread!")

    def create_shared_memory_V2(self, name, shape: tuple, nbytes: int, dtype: str):
        """Create shared-memory.

        Args:
            name (str): Name of shared memory.
            shape (tuple): Buffer shape.
            dtype (str): Buffer dtype.
        """
        dtype = self.str_to_dtype(dtype)

        self._shape = shape
        self.__nbytes = nbytes
        self._dtype = dtype
        self.name = name
        self.shm_buffer = shared_memory.SharedMemory(create=True, size=self.__nbytes, name=self.name)
        self._buffer = np.ndarray(self._shape, dtype=self._dtype, buffer=self.shm_buffer.buf)

    def read_shared_memory(self, shape: tuple):
        """Read shared memory.

        Args:
            shape (tuple): Buffer shape.

        Raises:
            Exception: can't be used in main thread!
        """
        if self._computer_type == 'HPC':
            warnings.warn("HPC can't support shared memory!")

        if self.level == 1:
            self.__nbytes = self.compute_nbytes(shape)
            self._shape = shape
            self.shm_buffer = shared_memory.SharedMemory(name=self.name, size=self.__nbytes)
            self._buffer = np.ndarray(self._shape, dtype=self._dtype, buffer=self.shm_buffer.buf)
        else:
            raise Exception("Current thread is main thread!")

    def read_shared_memory_V2(self, name, shape: tuple, nbytes: int, dtype: str):

        dtype = self.str_to_dtype(dtype)

        self._shape = shape
        self.__nbytes = nbytes
        self._dtype = dtype
        self.name = name
        self.shm_buffer = shared_memory.SharedMemory(name=self.name, size=self.__nbytes)
        self._buffer = np.ndarray(self._shape, dtype=self._dtype, buffer=self.shm_buffer.buf)

    def get_key_info(self):
        """Get key info.

        Returns:
            _type_: dict
        """

        dtype = self.get_str_from_dtype(self._dtype)

        dic = {
            self.name: {
                'shape': self._shape,
                'dtype': dtype,
                'nbytes': self.__nbytes,
            },
        }

        return dic

    def compute_nbytes(self, shape: tuple) -> int:
        """Compute numpy array nbytes.

        Args:
            shape (tuple): _description_

        Returns:
            _type_: int
        """

        a = np.arange(1, dtype=self._dtype)

        single_nbytes = a.nbytes

        total_size = 1

        for size in shape:
            total_size = total_size * size

        total_size = total_size * single_nbytes

        return total_size

    def store(self, data, index: int):
        """Store data in buffer.

        Args:
            data (any): feature in the training.
            index (int): index of data.
        """
        self._buffer[index] = data

    def store_sequence(self, data, start: int, end: int):
        """Store data in buffer.

        Args:
            data (any): feature in the training.
            start (int): start index.
            end (int): end index.
        """
        self._buffer[start:end] = data[:]

    # set value. This is for args
    def set_value(self, value):
        """Set value.

        Args:
            value (any): value.
        """
        if self.shape == (1,):
            self._buffer[0] = value
        else:
            self._buffer[:] = value

    @property
    def __call__(self):
        if self.shape == (1,):
            return self._buffer[0]
        else:
            return self._buffer

    # get data slice
    def get_slice(self, start: int, end: int):
        """Get slice.

        Args:
            start (int): start index.
            end (int): end index.

        Returns:
            _type_: np.ndarray
        """
        return self._buffer[start:end]

    def get_data_by_indices(self, indices):
        """Get data by indenes.

        Args:
            indices : indices.

        Returns:
            _type_: np.ndarray
        """
        return self._buffer[indices]

    @property
    def buffer(self):
        """Get buffer.

        Returns:
            _type_: np.ndarray
        """
        return self._buffer

    def close(self):
        # TODO: 元数据里面不要加入等待指令
        """
        delete data.
        """

        if self.shm_buffer is not None:
            if self.level == 1:
                self.shm_buffer.close()
                self.shm_buffer.unlink()
            else:
                import time
                time.sleep(0.5)
                self.shm_buffer.close()
        # del self._buffer

    def clear(self):
        if self.shm_buffer is not None:
            if self.level == 1:
                self.shm_buffer.close()
                self.shm_buffer.unlink()
            else:
                import time
                self.shm_buffer.close()

    @property
    def shape(self):
        """Get shape.

        Returns:
            _type_: tuple
        """
        return self._shape

    @property
    def dtype(self):
        """Get dtype.

        Returns:
            _type_: np.type
        """
        return self._dtype

    @staticmethod
    def get_dtype_from_str(dtype: str):
        """Get dtype from str.

        Args:
            dtype (str): dtype in str.

        """
        if dtype == 'np.float':
            dtype = np.float
        elif dtype == 'np.float16':
            dtype = np.float16
        elif dtype == 'np.float32':
            dtype = np.float32
        elif dtype == 'np.float64':
            dtype = np.float64
        elif dtype == 'np.int':
            dtype = np.int
        elif dtype == 'np.int32':
            dtype = np.int32
        elif dtype == 'np.int64':
            dtype = np.int64

        return dtype

    @staticmethod
    def get_str_from_dtype(dtype: np.dtype):
        """Get str from dtype.

        Args:
            dtype (np.dtype): dtype.

        """
        if dtype == np.float_:
            dtype = 'np.float'
        elif dtype == np.float16:
            dtype = 'np.float16'
        elif dtype == np.float32:
            dtype = 'np.float32'
        elif dtype == np.float64:
            dtype = 'np.float64'
        elif dtype == np.int_:
            dtype = 'np.int_'
        elif dtype == np.int32:
            dtype = 'np.int32'
        elif dtype == np.int64:
            dtype = 'np.int64'

        return dtype


# print(DataUnit.get_dtype_from_str('np.float32'))
# print(DataUnit.get_str_from_dtype(np.float32))
if __name__ == '__main__':
    unit = DataUnit('test', (1,), np.float32, level=0)

    print(unit())
