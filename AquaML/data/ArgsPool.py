from AquaML.data.DataUnit import DataUnit
from AquaML.data.BasePool import BasePool
from AquaML.DataType import DataInfo
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

    def create_buffer_from_tuple(self, info_tuple: tuple):
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



