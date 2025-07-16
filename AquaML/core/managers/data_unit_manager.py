"""Data Unit Manager for AquaML Framework

This module provides specialized management for data unit instances.
"""

from typing import Any, Dict, Callable
from loguru import logger


class DataUnitManager:
    """Manager for data unit registration and retrieval"""

    def __init__(self):
        """Initialize the data unit manager"""
        self.data_units_ = {}  # 记录数据单元实例
        logger.debug("DataUnitManager initialized")

    def register_data_unit(self, data_unit_cls: type) -> Callable:
        """注册数据单元实例，方便集中管理

        Args:
            data_unit_cls: 数据单元类

        Returns:
            Wrapper function
        """

        def wrapper(*args, **kwargs):
            """注册数据单元实例"""
            data_unit_instance = data_unit_cls(*args, **kwargs)

            # 记录数据单元实例
            unit_name = getattr(data_unit_instance, "name", data_unit_cls.__name__)

            if unit_name in self.data_units_:
                logger.warning(f"Data unit {unit_name} already exists, replacing it")

            self.data_units_[unit_name] = data_unit_instance

            logger.info(f"Successfully registered data unit: {unit_name}")

            return data_unit_instance

        return wrapper

    def get_data_unit(self, unit_name: str) -> Any:
        """获取数据单元实例

        Args:
            unit_name: 数据单元名称

        Returns:
            数据单元实例

        Raises:
            ValueError: 如果数据单元不存在
        """
        if unit_name not in self.data_units_:
            logger.error(f"Data unit {unit_name} not exists!")
            raise ValueError(f"Data unit {unit_name} not exists!")

        return self.data_units_[unit_name]

    def set_data_unit(self, unit_name: str, data_unit_instance: Any) -> None:
        """直接设置数据单元实例

        Args:
            unit_name: 数据单元名称
            data_unit_instance: 数据单元实例
        """
        if unit_name in self.data_units_:
            logger.warning(f"Data unit {unit_name} already exists, replacing it")

        self.data_units_[unit_name] = data_unit_instance
        logger.info(f"Data unit set: {unit_name}")

    def data_unit_exists(self, unit_name: str) -> bool:
        """检查数据单元是否存在

        Args:
            unit_name: 数据单元名称

        Returns:
            True if data unit exists, False otherwise
        """
        return unit_name in self.data_units_

    def remove_data_unit(self, unit_name: str) -> None:
        """移除数据单元实例

        Args:
            unit_name: 数据单元名称
        """
        if unit_name not in self.data_units_:
            logger.warning(f"Data unit {unit_name} not exists, cannot remove")
            return

        del self.data_units_[unit_name]
        logger.info(f"Removed data unit: {unit_name}")

    def list_data_units(self) -> list:
        """列出所有已注册的数据单元

        Returns:
            数据单元名称列表
        """
        return list(self.data_units_.keys())

    def get_data_units_count(self) -> int:
        """获取已注册数据单元数量

        Returns:
            数据单元数量
        """
        return len(self.data_units_)

    def clear_data_units(self) -> None:
        """清空所有数据单元"""
        self.data_units_.clear()
        logger.info("Cleared all data units")

    def save_data_unit_info(
        self, runner_name: str, file_system_manager
    ) -> Dict[str, Any]:
        """保存数据单元信息

        Args:
            runner_name: 运行器名称
            file_system_manager: 文件系统管理器

        Returns:
            保存的数据单元状态字典
        """
        save_dict = {}

        # 将数据单元的状态保存到字典中
        for key, value in self.data_units_.items():
            try:
                save_dict[key] = value.getUnitStatusDict()
            except AttributeError:
                logger.warning(
                    f"Data unit {key} does not have getUnitStatusDict method"
                )
                save_dict[key] = {}

        # 保存数据单元到文件系统中
        try:
            file_system = file_system_manager.get_file_system()
            file_system.saveDataUnit(
                runner_name=runner_name, data_unit_status=save_dict
            )
            logger.info(f"Saved data unit info for {len(save_dict)} units")
        except Exception as e:
            logger.error(f"Failed to save data unit info: {e}")

        return save_dict

    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态

        Returns:
            状态字典
        """
        return {
            "total_data_units": self.get_data_units_count(),
            "data_unit_names": self.list_data_units(),
        }
