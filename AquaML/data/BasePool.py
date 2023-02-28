from AquaML.DataType import DataInfo
from AquaML.data.DataUnit import DataUnit


class BasePool:

    def __init__(self, name, level: int, computer_type: str = 'PC'):
        self.name = name
        self._computer_type = computer_type
        self.level = level

    def create_share_memory(self):
        """create shared memory!
        """
        for key, data_unit in self.data_pool.items():
            data_unit.create_shared_memory()

    def read_shared_memory(self, info_dic: DataInfo):
        """read shared memory.

        Sub thread.

        Args:
            info_dic (DataInfo): store data information.
        """
        # create void data unit

        for name in info_dic.names:
            self.data_pool[name] = DataUnit(self.name + '_' + name, computer_type=self._computer_type, level=self.level,
                                            dtype=info_dic.type_dict[name])

        # read shared memory
        for name, data_unit in self.data_pool.items():
            data_unit.read_shared_memory(info_dic.shape_dict[name])

    def get_unit(self, name: str):
        """get data unit.

        Args:
            name (str): second level name. Algo search param via this name.

        Returns:
            DataUnit: data unit.
        """
        return self.data_pool[name]

        # get data from data unit

    def get_unit_data(self, name: str):
        """get data from data unit.

        Args:
            name (str): second level name. Algo search param via this name.

        Returns:
            np.ndarray: data.
        """
        return self.data_pool[name].buffer

        # close shared memory buffer

    def close(self):
        """close shared memory buffer.
        """
        for name, data_unit in self.data_pool.items():
            self.data_pool[name].close()

    def add_unit(self, name: str, data_unit: DataUnit):
        """add data unit.

        Args:
            name (str): second level name. Algo search param via this name.
            data_unit (DataUnit): data unit.
        """
        self.data_pool[name] = data_unit