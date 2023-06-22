from AquaML.DataType import DataInfo
from AquaML.data.DataUnit import DataUnit
import json


class BasePool:

    def __init__(self, name, level: int, computer_type: str = 'PC'):
        self.name = name
        self._computer_type = computer_type
        self.level = level

        self.pool_file = None
        self.data_pool = dict()

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

    def read_shared_memory_V2(self):
        """read shared memory.

        Sub thread.

        Args:
            info_dic (DataInfo): store data information.
        """

        for name, data_unit in self.data_pool.items():
            data_unit.read_shared_memory_V2(
                name=data_unit.name,
                shape=data_unit.shape,
                dtype=data_unit.dtype,
                nbytes=data_unit.nbytes,
            )



    def create_shared_memory_from_dic(self, info_dic: dict):
        """read shared memory from dic.

        Sub thread.

        Args:
            info_dic (dict): store data information.
        """
        # create void data unit

        for key, unit in self.data_pool.items():
            name = unit.name
            info = info_dic[name]
            self.data_pool[key] = unit.create_shared_memory_V2(
                name=name,
                shape=info['shape'],
                dtype=info['dtype'],
                nbytes=info['nbytes'],
            )

    def read_shared_memory_from_dict_direct(self, info_dic: dict, prefix=None):
        """read shared memory from dic.

        Sub thread.

        Args:
            info_dic (dict): store data information.
        """
        # create void data unit

        if prefix is None:
            start_index = 0
        else:
            start_index = len(prefix) + 1

        for key, value in info_dic.items():
            name = key[start_index:]
            self.data_pool[name].read_shared_memory_V2(
                name=key,
                shape=value['shape'],
                dtype=value['dtype'],
                nbytes=value['nbytes'],
            )

    def create_buffer_from_dict_direct(self, info_dic: dict, prefix=None):

        if prefix is None:
            start_index = 0
        else:
            start_index = len(prefix) + 1

        for key, value in info_dic.items():
            dtype = DataUnit.get_dtype_from_str(value['dtype'])
            self.data_pool[key[start_index:]] = DataUnit(
                name=key,
                shape=value['shape'],
                dtype=dtype,
                computer_type=self._computer_type,
                level=self.level,
            )

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

    def clear(self):
        """clear data pool.
        """
        for name, data_unit in self.data_pool.items():
            self.data_pool[name].clear()

    def add_unit(self, name: str, data_unit: DataUnit):
        """add data unit.

        Args:
            name (str): second level name. Algo search param via this name.
            data_unit (DataUnit): data unit.
        """
        self.data_pool[name] = data_unit

    def save_units_info_json(self, file: str):
        """write units info to json file.

        Args:
            file (str): path.
        """

        info_dics = {}

        for name, unit in self.data_pool.items():

            info_dic = unit.get_key_info()

            for key, value in info_dic.items():
                info_dics[key] = value

        # with open(path+'/pool_config.jason', 'w') as f:
        json.dump(info_dics, open(file, 'w'), indent=3)

    def set_pool_file(self, file):
        self.pool_file = file

    def get_data(self, data_name: str):
        
        return self.get_unit_data(data_name)
