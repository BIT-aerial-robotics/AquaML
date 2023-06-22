from AquaML.data.DataUnit import DataUnit
from AquaML.data.BasePool import BasePool
from AquaML.DataType import DataInfo
import numpy as np


class DataPool(BasePool):
    """Create and manage data units. When using supervised learning, the datapool contains 
    features and lables. As for reinforcement learning, (s,a,r,s') will be contained. 

    It can be used in parameter tuning

    """

    def __init__(self, name, level: int, computer_type: str = 'PC'):
        super().__init__(
            name=name,
            level=level,
            computer_type=computer_type
        )
        

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

    def create_buffer_from_dict(self, info_dic: DataInfo):
        """ Create buffer.

        Main thread.

        Args:
            info_dic (DataInfo): store data information.
        """
        for key, shape in info_dic.shape_dict.items():
            # print(key, shape)
            self.data_pool[key] = DataUnit(self.name + '_' + key, shape=shape, dtype=info_dic.type_dict[key],
                                           computer_type=self._computer_type, level=self.level)

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
                # time.sleep(6)
                self.read_shared_memory(info_dic)

        elif type == 'buffer':
            """
            Used for reinforcement learning.
            1. create buffer from dic
            2. create share memory
            3. read shared memory
            
            """
            if self.level == 0:
                self.create_buffer_from_dict(info_dic)
                if self._computer_type == 'PC':
                    self.create_share_memory()
            else:
                import time
                # time.sleep(6)
                self.read_shared_memory(info_dic)
        

    def store(self, data_dict: dict, index: int, prefix: str = ''):
        """store data.
        
        This function stores data in data units. data_dict no need to contain datapool's all data units.

        Args:
            data_dict (dict): data.
            index (int): index.
            prefix (str, optional): prefix. Defaults to ''.
        """
        for name, data in data_dict.items():
            self.data_pool[prefix + name].store(data, index)
            
    def store_sequence(self, name: str, data: np.ndarray, start_index: int, end_index: int):
        """store sequence data.

        Args:
            name (str): data unit name.
            data (np.ndarray): data.
            index (int): index.
        """
        self.data_pool[name].store_sequence(data, start_index, end_index)

    def store_all(self, name: str, data: np.ndarray):
        """store all data.

        Args:
            name (str): data unit name.
            data (np.ndarray): data.
        """
        self.data_pool[name].set_value(data)

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

    def get_numpy_dict(self):
        """get numpy dict.

        Returns:
            dict: data.
        """
        data_dict = dict()

        for name, unit in self.data_pool.items():
            data_dict[name] = unit.buffer

        return data_dict




