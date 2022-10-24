from multiprocessing import shared_memory
import numpy as np
import copy


# TODO: change the name.
class ParamData:
    def __init__(self, name: str, shape: tuple, hierarchical: int):
        """
        Store 1-D parameters of algorithm, policy at el. And sync parameters for each thread.

        :param name: (str) Name of this parameter.
        :param shape: (tuple) Parameter's shape.
        :param hierarchical: (int) The thread level to this DataPool. Default is 0.If the input is MAIN_THREAD, the data pool will create a block of shared-memory.If the input is SUB_THREAD,  the data pool will load a block of shared-memory.
        """
        # shape = (1, *shape)

        if hierarchical == 0:
            self.param = np.zeros(shape=shape, dtype=np.float32)
        else:
            param = np.zeros(shape=shape, dtype=np.float32)
            if hierarchical == 1:
                self.shm_param = shared_memory.SharedMemory(create=True, name=name, size=param.nbytes)
            elif hierarchical == 2:
                self.shm_param = shared_memory.SharedMemory(name=name, size=param.nbytes)
            else:
                raise ValueError(
                    "Invalid input of hierarchical. You input:{}, but it just receives:0, MAIN_THREAD, SUB_THREAD.".format(
                        hierarchical))

            self.param = np.ndarray(param.shape, dtype=np.float32, buffer=self.shm_param.buf)

        self.hierarchical = hierarchical

    def close(self):
        """
        Release shared_memory.

        :return: None.
        """
        del self.param

        if self.hierarchical > 0:
            # if self.hierarchical == 1:
            # import time
            # time.sleep(0.5)
            self.shm_param.close()

            if self.hierarchical == 1:
                self.shm_param.unlink()

    @property
    def data(self):
        # c = np.zeros_like(self.param)
        c = copy.deepcopy(self.param)
        return c

    def set_value(self, value):
        if self.hierarchical == 1 or self.hierarchical == 0:
            self.param[:] = value[:]
        else:
            raise ValueError("Can't modify the data in the sub-thread.")

    def __call__(self):
        return self.param
