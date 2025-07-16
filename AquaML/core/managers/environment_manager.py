"""Environment Manager for AquaML Framework

This module provides specialized management for environment instances.
"""

from typing import Any, Optional, Callable
from loguru import logger


class EnvironmentManager:
    """Manager for environment registration and retrieval"""

    def __init__(self):
        """Initialize the environment manager"""
        self.env_ = None  # 环境实例
        logger.debug("EnvironmentManager initialized")

    def register_env(self, env_cls: type) -> Callable:
        """注册环境实例，方便集中管理

        Args:
            env_cls: 环境类

        Returns:
            Wrapper function
        """

        def wrapper(*args, **kwargs):
            """注册环境实例"""
            if self.env_ is not None:
                logger.warning("Environment already exists, replacing it")

            env_instance = env_cls(*args, **kwargs)

            # 记录环境实例
            self.env_ = env_instance

            env_name = getattr(env_instance, "name", env_cls.__name__)
            logger.info(f"Successfully registered env: {env_name}")

            return env_instance

        return wrapper

    def get_env(self) -> Any:
        """获取环境实例

        Returns:
            环境实例

        Raises:
            ValueError: 如果环境不存在
        """
        if self.env_ is None:
            logger.error("env not exists!")
            raise ValueError("env not exists!")

        return self.env_

    def set_env(self, env_instance: Any) -> None:
        """直接设置环境实例

        Args:
            env_instance: 环境实例
        """
        self.env_ = env_instance
        env_name = getattr(env_instance, "name", "Unknown")
        logger.info(f"Environment set: {env_name}")

    def env_exists(self) -> bool:
        """检查环境是否存在

        Returns:
            True if environment exists, False otherwise
        """
        return self.env_ is not None

    def remove_env(self) -> None:
        """移除环境实例"""
        if self.env_ is not None:
            env_name = getattr(self.env_, "name", "Unknown")
            self.env_ = None
            logger.info(f"Removed environment: {env_name}")
        else:
            logger.warning("No environment to remove")

    def get_env_info(self) -> dict:
        """获取环境信息

        Returns:
            环境信息字典
        """
        if self.env_ is None:
            return {"exists": False, "name": None, "type": None}

        return {
            "exists": True,
            "name": getattr(self.env_, "name", "Unknown"),
            "type": type(self.env_).__name__,
        }

    def get_status(self) -> dict:
        """获取管理器状态

        Returns:
            状态字典
        """
        return {"env_registered": self.env_exists(), "env_info": self.get_env_info()}
