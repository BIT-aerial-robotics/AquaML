from loguru import logger
from AquaML.data.base_unit import BaseUnit
import torch
from . import unitCfg
import numpy as np
from AquaML import coordinator

@coordinator.registerDataUnit
class TensorUnit(BaseUnit):
    """
    torch数据类。
    该类型在分布式通信中只能通过NumPyUnit进行通信。
    """

    def __init__(self,
                 unit_cfg: unitCfg = None,
                 ):
        """
        创建一个torch数据类。

        Args:
            name (str): 数据名称
            unit_cfg(unitCfg, optional): 数据信息。
        """

        super().__init__(unit_cfg,)

        self.data_ = None

    def createData(self):
        """
        创建torch数据。
        """

        self.data_ = torch.zeros(
            self.unit_cfg_.shape, dtype=self.unit_cfg_.dtype, device=self.unit_cfg_.device)
        logger.info("successfully create data {}".format(self.name_))

    
    def computeBytes(self)->int:
        """
        计算数据的字节数。

        """

        return np.dtype(self.unit_cfg_.dtype).itemsize * np.prod(self.unit_cfg_.shape)