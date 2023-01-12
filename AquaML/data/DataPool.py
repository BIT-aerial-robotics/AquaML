from AquaML.data.DataUnit import DataUnit
from AquaML.DataType import DataInfo
import numpy as np

class DataPool:
    """Create and manage data units. When using supervised learning, the datapool contains 
    features and lables. As for reinforcement learning, (s,a,r,s') will be contained. 

    It can be used in parameter tuning

    """

    def __init__(self, name, level:int ,computer_type:str='PC'):
        
        self.name = name # first level name
        self.data_pool = dict()

        self._computer_type = computer_type
        self.level = level

    # TODO: 多线程格式的统一
    def copy_from_exist_array(self,dataset:np.ndarray, name:str):
        """copy data from np array and create data units.

        This function works in main thread.

        Args:
            dataset (np.ndarray): _description_
            name (str): second level name. Algo serch param via this name.
        """
        unit_name = self.name + '_' + name

        self.data_pool[name] = DataUnit(unit_name,dataset=dataset, level=self.level)

    def create_buffer_from_dic(self, info_dic:DataInfo):
        """ Create buffer. Usually, used in reinforcement learning.

        Main threaad.

        Args:
            info_dic (DataInfo): store data information.
        """
        for key, shape in info_dic.shape_dict.items():
            self.data_pool[key] = DataUnit(self.name +'_'+key, shape=shape, dtype=info_dic.type_dict[key], computer_type=self._computer_type, level=self.level)
    
    def create_share_memory(self):
        """create shared memory!
        """
        for key, data_unit in self.data_pool.items():
            data_unit.create_shared_memory()

    def read_shared_memory(self, info_dic:DataInfo):
        """read shared memory.

        Sub thread.

        Args:
            info_dic (DataInfo): store data information.
        """
        # create void data unit

        for name in info_dic.names:
            self.data_pool[name] = DataUnit(self.name +'_'+name, computer_type=self._computer_type, level=self.level, dtype=info_dic.type_dict[name])

        # read shared memory
        for name, data_unit in self.data_pool.items():
            data_unit.read_shared_memory(info_dic.shape_dict[name])
        

    # TODO: 当前还需要指定shape，升级自动获取版本
    def multi_sync(self, info_dic:DataInfo, type:str='dataset'):
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
                self.read_shared_memory(info_dic)