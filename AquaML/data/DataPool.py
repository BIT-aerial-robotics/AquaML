from AquaML.data.DataUnit import DataUnit
import numpy as np

class DataPool:
    """Create and manage data units. When using supervised learning, the datapool contains 
    features and lables. As for reinforcement learning, (s,a,r,s') will be contained. 

    It can be used in parameter tuning

    """

    def __init__(self, name, computer_type:str='PC'):
        
        self.name = name # first level name
        self.data_pool = dict()

        self._computer_type = computer_type

    # Todo: 多线程格式的统一
    def copy_from_exist_array(self,dataset:np.ndarray, name:str):
        unit_name = self.name + '_' + name

        self.data_pool[name] = DataUnit(unit_name,dataset=dataset)