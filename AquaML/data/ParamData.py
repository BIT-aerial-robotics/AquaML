import copy

import numpy as np
from multiprocessing import shared_memory


class ParamData:
    def __init__(self, name: str, shape: tuple, dtype=np.float32, share_memory=False):
        """
        DataPool is the data storage unit and is used to create a data storage warehouse and synchronize the shared-memory address.
        :param name:(str) The name of the warehouse.
        :param shape:(tuple) Shape of a single data.\n If the data is image, shape: (width, height, channel)
        :param total_length:(int) Total length of this data.
        :param dtype: (np.type) Type of this data.
        :param share_memory: (bool) Usually used by single computer multiprocess.
        """

        self.name = name

        self.shape = shape  # single data shape

        self.share_memory = share_memory

        self.dtype = dtype

        self.shapes = shape  # total length, single shape

        if share_memory:
            data_pool = np.zeros(shape=shape, dtype=dtype)
            try:
                self.shm_data_pool = shared_memory.SharedMemory(create=True, size=data_pool.nbytes, name=self.name)
                self.master_thread = True
            except Exception:
                self.shm_data_pool = shared_memory.SharedMemory(size=data_pool.nbytes, name=self.name)
                self.master_thread = False

            self._data = np.ndarray(data_pool.shape, dtype=dtype, buffer=self.shm_data_pool.buf)

        else:
            self._data = np.zeros(shape=shape, dtype=dtype)

    def store(self, data, index):
        """
        store data in data pool.

        :param data:(ndarray)
        :param index: the index of the data pool.
        :return: None
        """

        self._data[index] = data

    def close(self):
        """
        Release shared_memory.

        :return: None.
        """
        del self._data

        if self.share_memory:
            self.shm_data_pool.close()
            try:
                self.shm_data_pool.unlink()
            except Exception:
                pass

    def data_block(self, start, end):
        return self._data[start:end]

    @property
    def data(self):
        c = copy.deepcopy(self._data)
        return c

    def set_value(self, value):
        if self.share_memory:
            if self.master_thread:
                self._data[:] = value[:]
        else:
            self._data[:] = value[:]
