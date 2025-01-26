from AquaML import logger
import numpy as np
from AquaML.data.base_unit import BaseUnit

class NumpyUnit(BaseUnit):
    """
    numpy数据类。
    该类型支持共享内存，支持多进程通过TCPIP通信。
    主要用于分布式通信。
    和TensorUnit一块使用可以实现数据共享。
    """
    
    def __init__(self,
                 name:str,
                 unit_info:dict=None,
                 create_first: bool = True,
                 ):
        """
        创建一个numpy数据类。

        Args:
            name (str): 数据名称
            unit_info (dict, optional): 数据信息。如: 
            
            >>>> {
            >>>>   'dtype':np.float32, 
            >>>>    'shape':(100, 100, 100),
            >>>>    'size': 30
            >>>> },
            create_first (bool, optional): 是否创建矩阵。默认为True。
        """

        super().__init__(name,unit_info,"numpy")

        self.data_ = None # 数据
        self.create_first_ = create_first # 是否创建矩阵
        
    
    def createData(self):
        """
        创建numpy数据。
        """
        
        if self.create_first_:
            self.data_ = np.zeros(self.unit_info_['shape'],dtype=self.unit_info_['dtype'])
            logger.info("successfully create data {}".format(self.name_))
        else:
            logger.info("data {} not created".format(self.name_))
        
    