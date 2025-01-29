from loguru import logger

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data.base_unit import BaseUnit
    from .file_system.base_file_system import BaseFileSystem


class AquaMLCoordinator:
    """
    AquaML协调器类，用于管理AquaML的核心。
    该函数将存储数据格式和工作流。简化其他部件的传递参数。
    比如将数据格式注册到协调器中，其他部件只需要获取协调器中的数据格式即可。
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        创建AquaML协调器实例。
        """
        if not cls._instance:
            # 用蓝色打印欢迎信息(Welcome to use AquaML),场面热烈
            print("\033[1;34mWelcome to use AquaML!\033[0m")

            cls._instance = super().__new__(cls)
            cls._instance._init_core(*args, **kwargs)
        return cls._instance

    def _init_core(self, *args, **kwargs):
        """
        初始化AquaML协调器,用于存储数据单元实例及其信息。
        """

        # TODO: 看看有没有更好的存储方式
        self.data_units_ = {}  # 记录数据单元实例
        self.file_system_:BaseFileSystem = None  # 文件系统

    def registerDataUnit(self, data_unit_cls):
        """
        注册数据单元实例，方便集中管理。
        并且能够同步不同模块对数据单元的操作。

        Args:
            data_unit_cls: 数据单元类。
        """

        def wrapper(*args, **kwargs):
            """
            注册数据单元实例。
            """
            isinstance: BaseUnit = data_unit_cls(*args, **kwargs)

            # 记录数据单元实例
            self.data_units_[isinstance] = isinstance

            logger.info(
                ' Successfully register data unit {}'.format(isinstance.name))

            return isinstance

        return wrapper
    
    def registerFileSystem(self, file_system_cls):
        """
        注册文件系统实例，方便集中管理。
        并且能够同步不同模块对文件系统的操作。

        Args:
            file_system_cls: 文件系统类。
        """

        def wrapper(*args, **kwargs):
            """
            注册文件系统实例。
            
            每个学习方法只有一个文件系统实例。
            """
            
            
            if self.file_system_ is not None:
                logger.error("file system already exists!")
                raise ValueError("file system already exists!")
            
            self.file_system_ = file_system_cls(*args, **kwargs)
            
            logger.info(
                ' Successfully register file system')

            return self.file_system_
        
        return wrapper
