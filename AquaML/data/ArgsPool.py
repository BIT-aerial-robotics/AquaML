from AquaML.data.DataUnit import DataUnit
from AquaML.data.BasePool import BasePool
from AquaML.BaseClass import BaseParameter
import numpy as np


class ArgsPool(BasePool):
    """Create and manage parameters. It can be used in parameter tuning by meta learning.
    """

    def __init__(self, name, level: int, computer_type: str = 'PC'):
        super().__init__(
            name=name,
            level=level,
            computer_type=computer_type
        )
        self.data_pool = dict()

    def create_buffer_from_tuple(self, info_tuple: tuple or list):
        """
        Create buffer.
        """
        for name in info_tuple:
            self.data_pool[name] = DataUnit(
                name=self.name + '_' + name,
                shape=(1,),
                dtype=np.float32,
                computer_type=self._computer_type,
                level=self.level
            )

    # def read_shared_memory_from_tuple(self, info_tuple: tuple or list):

    def create_shared_memory(self):
        """create shared memory from dic.
        """
        for key, data_unit in self.data_pool.items():
            data_unit.create_shared_memory()

    def multi_init(self):
        """
        multi thread initial.

        We suppose, after the shared memory is created, shared memory will be read.
        """

        if self.level == 0:
            for name, unit in self.data_pool.items():
                unit.create_shared_memory()
        else:
            import time
            time.sleep(6)
            for name, unit in self.data_pool.items():
                unit.read_shared_memory(shape=unit.shape)

    def get_param(self, name):
        return self.data_pool[name].buffer[0]

    def set_param_by_dict(self, param_dict: dict):
        for key, value in param_dict.items():
            self.data_pool[key].set_value(value)

    def set_param_by_name(self, name, value):
        self.data_pool[name].set_value(value)



