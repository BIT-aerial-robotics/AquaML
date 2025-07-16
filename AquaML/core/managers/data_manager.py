"""Data Manager for AquaML Framework

This module provides specialized management for data manager instances.
"""

from typing import Any, Callable
from loguru import logger


class DataManager:
    """Manager for data manager registration and retrieval"""

    def __init__(self):
        """Initialize the data manager"""
        self._data_manager = None  # 数据管理器实例
        logger.debug("DataManager initialized")

    def register_data_manager(self, data_manager_cls: type) -> Callable:
        """注册数据管理器类

        Args:
            data_manager_cls: 数据管理器类

        Returns:
            Wrapper function
        """

        def wrapper(*args, **kwargs):
            """注册数据管理器实例"""
            if self._data_manager is not None:
                logger.warning("Data manager already exists, replacing it")

            data_manager_instance = data_manager_cls(*args, **kwargs)
            self._data_manager = data_manager_instance

            logger.info("Successfully registered data manager")

            return data_manager_instance

        return wrapper

    def get_data_manager(self) -> Any:
        """获取注册的数据管理器

        Returns:
            数据管理器实例

        Raises:
            ValueError: 如果数据管理器不存在
        """
        if self._data_manager is None:
            logger.error("Data manager not exists!")
            raise ValueError("Data manager not exists!")

        return self._data_manager

    def set_data_manager(self, data_manager_instance: Any) -> None:
        """直接设置数据管理器实例

        Args:
            data_manager_instance: 数据管理器实例
        """
        if self._data_manager is not None:
            logger.warning("Data manager already exists, replacing it")

        self._data_manager = data_manager_instance
        logger.info("Data manager set")

    def data_manager_exists(self) -> bool:
        """检查数据管理器是否存在

        Returns:
            True if data manager exists, False otherwise
        """
        return self._data_manager is not None

    def remove_data_manager(self) -> None:
        """移除数据管理器实例"""
        if self._data_manager is not None:
            self._data_manager = None
            logger.info("Removed data manager")
        else:
            logger.warning("No data manager to remove")

    def get_data_manager_info(self) -> dict:
        """获取数据管理器信息

        Returns:
            数据管理器信息字典
        """
        if self._data_manager is None:
            return {"exists": False, "type": None}

        return {"exists": True, "type": type(self._data_manager).__name__}

    def get_status(self) -> dict:
        """获取管理器状态

        Returns:
            状态字典
        """
        return {
            "data_manager_registered": self.data_manager_exists(),
            "data_manager_info": self.get_data_manager_info(),
        }
