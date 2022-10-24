import numpy as np
from multiprocessing import shared_memory


class DataPool:
    def __init__(self, name: str, shape: tuple, total_length: int, hierarchical: int = 0, dtype=np.float32):
        """
        DataPool is the data storage unit and is used to create a data storage warehouse and synchronize the shared-memory address.

        :param name:(str) The name of the warehouse.
        :param shape:(tuple) Shape of a single data.\n If the data is image, shape: (width, height, channel)
        :param total_length:(int) Total length of this data.
        :param hierarchical:(int) The thread level to this DataPool. Default is 0.If the input is MAIN_THREAD, the data pool will create a block of shared-memory.If the input is SUB_THREAD,  the data pool will load a block of shared-memory.
        :param dtype: (np.type) Type of this data.
        """

        self.name = name

        shapes = []
        shapes.append(total_length)
        # self.shapes = shapes
        for value in shape:
            shapes.append(value)

        self.shm_data_pool = None

        if hierarchical == 0:
            self.data_pool = np.zeros(shape=shapes, dtype=dtype)
        else:
            data_pool = np.zeros(shape=shapes, dtype=dtype)
            if hierarchical == 1:
                self.shm_data_pool = shared_memory.SharedMemory(create=True, size=data_pool.nbytes, name=self.name)
            elif hierarchical == 2:
                self.shm_data_pool = shared_memory.SharedMemory(size=data_pool.nbytes, name=self.name)
            else:
                raise ValueError(
                    "Invalid input of hierarchical. You input:{}, but it just receives:0, MAIN_THREAD, SUB_THREAD.".format(
                        hierarchical))

            self.data_pool = np.ndarray(data_pool.shape, dtype=dtype, buffer=self.shm_data_pool.buf)

        self.hierarchical = hierarchical

    def store(self, data, index):
        """
        store data in data pool.

        :param data:(ndarray)
        :param index: the index of the data pool.
        :return: None
        """

        self.data_pool[index] = data

    @property
    def shape(self):
        return self.data_pool.shape

    @property
    def data(self):
        return self.data_pool

    def close(self):
        """
        Release shared_memory.

        :return: None.
        """
        del self.data_pool

        if self.hierarchical > 0:
            if self.hierarchical == 1:
                import time
                time.sleep(0.5)
            self.shm_data_pool.close()

            if self.hierarchical == 1:
                self.shm_data_pool.unlink()

    # @property
    def data_block(self, start, end):
        return self.data_pool[start: end]
