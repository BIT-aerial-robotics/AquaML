import abc
from AquaML.data.DataPool import DataPool
from AquaML.DataType import RLIOInfo

class BaseRLalgo(abc.ABC):

    # TODO:统一输入接口
    # TODO:判断是否启动多线程   

    def __init__(self, rl_io_info:RLIOInfo, computer_type:str='PC', level:int=0):

        self.rl_io_info = rl_io_info
        self._computer_type = computer_type
        self.level = level

        # create data pool according to thread level
        self.data_pool = DataPool(self._computer_type, self.level) # data_pool is a handle
        