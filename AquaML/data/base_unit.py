
from AquaML import logger
from abc import ABC, abstractmethod

class BaseUnit(ABC):
    """
    所有数据类的基类。
    """
    
    def __init__(self,
                 name:str,
                 unit_info:dict=None,
                 mode: str = "numpy",
                 ):
        
        """
        初始化数据类。
        
        Args:
            name (str): 数据名称
            unit_info (dict, optional): 数据信息。如: 
            
            >>>> {
            >>>>   'dtype':np.float32, 
            >>>>    'shape':(100, 100, 100),
            >>>>    'size': 30
            >>>> },
            mode (str, optional): 数据的存储模式。默认为"numpy"。
        """
        
        self.name_ = name # unit名称
        
        self.mode_ = mode # 数据的存储模式
        self.unit_info_ = unit_info # unit信息

        # 检查unit_info是否正确
        if unit_info is not None:
            self.checkUintInfo(unit_info)
            
        self.data_ = None # 数据
        logger.info("data {} use mode {}".format(self.name_,self.mode_))
        
    def checkUintInfo(self,unit_info:dict):
        """
        检查unit_info是否正确。
        """
        
        # 检查unit_info是否正确
        
        if 'dtype' not in unit_info:
            logger.error("unit_info must contain dtype")
            raise ValueError("unit_info must contain dtype")
        if 'shape' not in unit_info:
            logger.error("unit_info must contain shape")
            raise ValueError("unit_info must contain shape")
        if'size' not in unit_info:
            logger.error("unit_info must contain size")
            raise ValueError("unit_info must contain size")
        
    def __call__(self):
        """
        调用数据。
        """

        if self.data_ is None:
            logger.warning("data {} is not initialized".format(self.name_))
            # self.initData()
        return self.data_
    
    def __getitem__(self,key):
        """
        获取数据。
        """

        if self.data_ is None:
            logger.warning("data {} is not initialized".format(self.name_))
            # self.initData()
        return self.data_[key]

    def __setitem__(self,key,value):
        """
        设置数据。
        """
        if self.data_ is None:
            logger.warning("data {} is not initialized".format(self.name_))
            # self.initData()
        self.data_[key] = value
        
    @abstractmethod
    def createData(self):
        """
        根据unit_info创建数据。
        """
        pass


        