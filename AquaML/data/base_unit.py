
from loguru import logger
from abc import ABC, abstractmethod
from . import unitCfg
import torch


class BaseUnit(ABC):
    """
    所有数据类的基类。
    """

    def __init__(self,
                 unit_cfg: unitCfg = None,
                 ):
        """
        初始化数据类。

        Args:
            unit_cfg (unitCfg, optional): 数据信息。
        """

        self.name_ = unit_cfg.name  # unit名称

        self.mode_ = unit_cfg.mode  # 数据的存储模式
        self.unit_cfg_ = unit_cfg  # unit信息

        # 检查unit_cfg是否正确
        if unit_cfg is not None:
            self.checkUintCfg(unit_cfg)

        self.data_ = None  # 数据
        logger.info("data {} use mode {}".format(self.name_, self.mode_))

    def checkUintCfg(self, unit_cfg: unitCfg):
        """
        检查unit_info是否正确。
        """

        # 检查unit_info是否正确

        if unit_cfg.dtype is None:
            logger.error("unit_cfg must clarify dtype!")
            raise ValueError("unit_cfg must clarify dtype!")
        if unit_cfg.single_shape is None:
            logger.error("unit_cfg must clarify single_shape!")
            raise ValueError("unit_cfg must clarify single_shape!")
        if unit_cfg.size is None:
            logger.error("unit_cfg must clarify size!")
            raise ValueError("unit_cfg must clarify size!")

        # 计算shape
        self.unit_cfg_.shape = (unit_cfg.size,) + unit_cfg.single_shape

        # 计算字节数
        self.unit_cfg_.bytes = self.computeBytes()

    def __call__(self):
        """
        调用数据。
        """

        if self.data_ is None:
            logger.warning("data {} is not initialized".format(self.name_))
            # self.initData()
        return self.data_

    def __getitem__(self, key):
        """
        获取数据。
        """

        if self.data_ is None:
            logger.warning("data {} is not initialized".format(self.name_))
            # self.initData()
        return self.data_[key]

    def __setitem__(self, key, value):
        """
        设置数据。
        """
        if self.data_ is None:
            logger.warning("data {} is not initialized".format(self.name_))
            # self.initData()
        self.data_[key] = value

    @property
    def name(self):
        """
        获取数据的名称。
        """
        return self.name_

    @property
    def shape(self):
        """
        获取数据的形状。
        """
        if self.unit_cfg_.shape is None:
            logger.warning("data {} is not initialized".format(self.name_))
            raise ValueError("data {} is not initialized".format(self.name_))

        return self.data_.shape

    @property
    def single_shape(self):
        """
        获取单个数据的形状。
        """
        return self.unit_cfg_.single_shape

    @property
    def size(self):
        """
        获取数据的长度。
        """
        return self.unit_cfg_.size

    @property
    def dtype(self):
        """
        获取数据的类型。
        """
        return self.unit_cfg_.dtype

    @abstractmethod
    def createData(self):
        """
        根据unit_info创建数据。
        """
        pass

    @abstractmethod
    def computeBytes(self) -> int:
        """
        计算数据的字节数。
        使用该函数时，数据并未创建，因此需要根据unit_cfg计算数据的字节数。
        """
        pass
