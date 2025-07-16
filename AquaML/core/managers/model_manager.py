"""Model Manager for AquaML Framework

This module provides specialized management for model registration and retrieval.
"""

from typing import Dict, Any
from loguru import logger
from torch.nn import Module


class ModelManager:
    """Manager for model registration and retrieval"""

    def __init__(self):
        """Initialize the model manager"""
        self.models_dict_ = {}  # 记录模型实例和模型的状态
        logger.debug("ModelManager initialized")

    def register_model(self, model: Module, model_name: str) -> None:
        """将模型注册到模型字典中

        Args:
            model: 模型实例
            model_name: 模型名称

        Raises:
            ValueError: 如果模型已经存在
        """
        # 检测当前模型是否已经注册
        if model_name in self.models_dict_:
            logger.error("model {} already exists!".format(model_name))
            raise ValueError("model {} already exists!".format(model_name))

        model_dict = {"model": model, "status": {}}  # 简化状态管理

        self.models_dict_[model_name] = model_dict
        logger.info(f"Successfully registered model: {model_name}")

    def get_model(self, model_name: str) -> Dict[str, Any]:
        """获取模型实例和当前状态

        Args:
            model_name: 模型名称

        Returns:
            模型字典，包含 'model' 和 'status' 键

        Raises:
            ValueError: 如果模型不存在
        """
        if model_name not in self.models_dict_:
            logger.error("model {} not exists!".format(model_name))
            raise ValueError("model {} not exists!".format(model_name))

        return self.models_dict_[model_name]

    def get_model_instance(self, model_name: str) -> Module:
        """获取模型实例

        Args:
            model_name: 模型名称

        Returns:
            模型实例
        """
        model_dict = self.get_model(model_name)
        return model_dict["model"]

    def update_model_status(self, model_name: str, status: Dict[str, Any]) -> None:
        """更新模型状态

        Args:
            model_name: 模型名称
            status: 状态字典
        """
        if model_name not in self.models_dict_:
            raise ValueError(f"Model {model_name} not exists!")

        self.models_dict_[model_name]["status"].update(status)
        logger.debug(f"Updated status for model: {model_name}")

    def list_models(self) -> list:
        """列出所有已注册的模型

        Returns:
            模型名称列表
        """
        return list(self.models_dict_.keys())

    def model_exists(self, model_name: str) -> bool:
        """检查模型是否存在

        Args:
            model_name: 模型名称

        Returns:
            True if model exists, False otherwise
        """
        return model_name in self.models_dict_

    def remove_model(self, model_name: str) -> None:
        """移除模型

        Args:
            model_name: 模型名称
        """
        if model_name not in self.models_dict_:
            logger.warning(f"Model {model_name} not exists, cannot remove")
            return

        del self.models_dict_[model_name]
        logger.info(f"Removed model: {model_name}")

    def get_models_count(self) -> int:
        """获取已注册模型数量

        Returns:
            模型数量
        """
        return len(self.models_dict_)

    def clear_models(self) -> None:
        """清空所有模型"""
        self.models_dict_.clear()
        logger.info("Cleared all models")

    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态

        Returns:
            状态字典
        """
        return {
            "total_models": self.get_models_count(),
            "model_names": self.list_models(),
        }
