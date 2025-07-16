from loguru import logger
import numpy as np
from AquaML.data.base_unit import BaseUnit
from . import unitCfg
from AquaML import coordinator


@coordinator.registerDataUnit
class NumpyUnit(BaseUnit):
    """
    numpy数据类。
    该类型支持共享内存，支持多进程通过TCPIP通信。
    主要用于分布式通信。
    和TensorUnit一块使用可以实现数据共享。
    """

    def __init__(self,
                 unit_cfg: unitCfg,
                 ):
        """
        创建一个numpy数据类。

        Args:
            unit_cfg (unitCfg): 数据信息。
        """

        super().__init__(unit_cfg)

        self.data_ = None  # 数据
        # self.create_first_ = create_first # 是否创建矩阵

    def createData(self, create_first: bool = True):
        """
        创建numpy数据。
        numpy支持共享内存，当需要使用时请先创建该numpy array然后使用多线程去读取。
        """

        self.data_ = np.zeros(self.unit_cfg_.shape, dtype=self.unit_cfg_.dtype)
        logger.info("successfully create data {}".format(self.name_))

    def computeBytes(self) -> int:
        """
        计算数据的字节数。

        """

        return np.dtype(self.unit_cfg_.dtype).itemsize * np.prod(self.unit_cfg_.shape)

        
