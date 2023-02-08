from AquaML.data.DataUnit import DataUnit
from AquaML.DataType import DataInfo
import numpy as np


class DataPool:
    """Create and manage data units. When using supervised learning, the datapool contains 
    features and lables. As for reinforcement learning, (s,a,r,s') will be contained. 

    It can be used in parameter tuning

    """

    def __init__(self, name, level: int, computer_type: str = 'PC'):

        self.name = name  # first level name
        self.data_pool = dict()

        self._computer_type = computer_type
        self.level = level

    # TODO: 多线程格式的统一
    def copy_from_exist_array(self, dataset: np.ndarray, name: str):
        """copy data from np array and create data units.

        This function works in main thread.

        Args:
            dataset (np.ndarray): _description_
            name (str): second level name. Algo serch param via this name.
        """
        unit_name = self.name + '_' + name

        self.data_pool[name] = DataUnit(unit_name, dataset=dataset, level=self.level)

    def create_buffer_from_dic(self, info_dic: DataInfo):
        """ Create buffer. Usually, used in reinforcement learning.

        Main thread.

        Args:
            info_dic (DataInfo): store data information.
        """
        for key, shape in info_dic.shape_dict.items():
            self.data_pool[key] = DataUnit(self.name + '_' + key, shape=shape, dtype=info_dic.type_dict[key],
                                           computer_type=self._computer_type, level=self.level)

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

    # TODO: 当前还需要指定shape，升级自动获取版本
    def multi_init(self, info_dic: DataInfo, type: str = 'dataset'):
        """multi thread initial.
        Args:
            info_dic (DataInfo): store data information.
            type (str, optional): 'dataset' or 'buffer'. Defaults to 'dataset'.
        """

        if type == 'dataset':
            """
            1. copy from exist array
            2. create share memory
            3. read shared memory
            """
            if self.level == 0:
                for key, dataset in info_dic.dataset_dict.items():
                    self.copy_from_exist_array(dataset, key)
                if self._computer_type == 'PC':
                    self.create_share_memory()
            else:
                # waite for main thread to create share memory
                import time
                time.sleep(6)
                self.read_shared_memory(info_dic)

        elif type == 'buffer':
            """
            Used for reinforcement learning.
            1. create buffer from dic
            2. create share memory
            3. read shared memory
            
            """
            if self.level == 0:
                self.create_buffer_from_dic(info_dic)
                if self._computer_type == 'PC':
                    self.create_share_memory()
            else:
                import time
                time.sleep(6)
                self.read_shared_memory(info_dic)

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

    def store(self, data_dict: dict, index: int):
        """store data.
        
        This function stores data in data units. data_dict no need to contain datapool's all data units.

        Args:
            data (dict): data.
            index (int): index.
        """
        for name, data in data_dict.items():
            self.data_pool[name].store(data, index)

    def get_data_by_indices(self, indices, names: tuple):
        """get data by indices.

        Args:
            indices (list): indice.
            names (tuple): second level name. Algo search param via this name.

        Returns:
            dict: data.
        """
        data_dict = dict()

        for name in names:
            data_dict[name] = self.data_pool[name].get_data_by_indices(indices)

        return data_dict


# test in sub thread
if __name__ == '__main__':
    test_info = DataInfo(names=('test',), shapes=((4, 1),), dtypes=np.float32)

    test = DataPool(name='test', level=1, computer_type='PC')

    test.multi_sync(test_info, type='buffer')
