"""Runner Manager for AquaML Framework

This module provides specialized management for runner instances.
"""

from typing import Optional
from loguru import logger
from datetime import datetime


class RunnerManager:
    """Manager for runner registration and retrieval"""

    def __init__(self):
        """Initialize the runner manager"""
        self.runner_name_ = None  # runner名称
        logger.debug("RunnerManager initialized")

    def register_runner(self, runner_name: Optional[str] = None) -> str:
        """注册runner名称，用于记录当前运行的runner名称

        Args:
            runner_name: runner名称，如果为None则自动生成日期时间名称

        Returns:
            实际使用的runner名称
        """
        # 如果未提供runner名称，自动生成日期时间名称
        if runner_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            runner_name = f"runner_{timestamp}"
            logger.info(f"Auto-generated runner name: {runner_name}")
        
        if self.runner_name_ is not None:
            logger.warning(
                f"Runner already exists: {self.runner_name_}, replacing with: {runner_name}"
            )

        self.runner_name_ = runner_name
        logger.info(f"Successfully registered runner: {runner_name}")
        return runner_name

    def get_runner(self) -> str:
        """获取runner名称

        Returns:
            runner名称

        Raises:
            ValueError: 如果runner不存在
        """
        if self.runner_name_ is None:
            logger.error("Runner not exists!")
            raise ValueError("Runner not exists!")

        return self.runner_name_

    def set_runner(self, runner_name: str) -> None:
        """设置runner名称

        Args:
            runner_name: runner名称
        """
        if self.runner_name_ is not None:
            logger.warning(
                f"Runner already exists: {self.runner_name_}, replacing with: {runner_name}"
            )

        self.runner_name_ = runner_name
        logger.info(f"Runner set: {runner_name}")

    def runner_exists(self) -> bool:
        """检查runner是否存在

        Returns:
            True if runner exists, False otherwise
        """
        return self.runner_name_ is not None

    def remove_runner(self) -> None:
        """移除runner"""
        if self.runner_name_ is not None:
            runner_name = self.runner_name_
            self.runner_name_ = None
            logger.info(f"Removed runner: {runner_name}")
        else:
            logger.warning("No runner to remove")

    def get_runner_info(self) -> dict:
        """获取runner信息

        Returns:
            runner信息字典
        """
        return {"exists": self.runner_exists(), "name": self.runner_name_}

    def get_status(self) -> dict:
        """获取管理器状态

        Returns:
            状态字典
        """
        return {
            "runner_registered": self.runner_exists(),
            "runner_info": self.get_runner_info(),
        }
