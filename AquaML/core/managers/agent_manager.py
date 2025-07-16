"""Agent Manager for AquaML Framework

This module provides specialized management for agent instances.
"""

from typing import Any, Callable
from loguru import logger


class AgentManager:
    """Manager for agent registration and retrieval"""

    def __init__(self):
        """Initialize the agent manager"""
        self.agent_ = None  # 智能体实例
        logger.debug("AgentManager initialized")

    def register_agent(self, agent_cls: type) -> Callable:
        """注册智能体实例，方便集中管理

        Args:
            agent_cls: 智能体类

        Returns:
            Wrapper function
        """

        def wrapper(*args, **kwargs):
            """注册智能体实例"""
            if self.agent_ is not None:
                logger.error("currently do not support multiple agents!")
                raise ValueError("agent already exists!")

            self.agent_ = agent_cls(*args, **kwargs)

            agent_name = getattr(self.agent_, "name", agent_cls.__name__)
            logger.info(f"Successfully registered agent: {agent_name}")

            return self.agent_

        return wrapper

    def get_agent(self) -> Any:
        """获取智能体实例

        Returns:
            智能体实例

        Raises:
            ValueError: 如果智能体不存在
        """
        if self.agent_ is None:
            logger.error("Agent not exists!")
            raise ValueError("Agent not exists!")

        return self.agent_

    def set_agent(self, agent_instance: Any) -> None:
        """直接设置智能体实例

        Args:
            agent_instance: 智能体实例
        """
        if self.agent_ is not None:
            logger.warning("Agent already exists, replacing it")

        self.agent_ = agent_instance
        agent_name = getattr(agent_instance, "name", "Unknown")
        logger.info(f"Agent set: {agent_name}")

    def agent_exists(self) -> bool:
        """检查智能体是否存在

        Returns:
            True if agent exists, False otherwise
        """
        return self.agent_ is not None

    def remove_agent(self) -> None:
        """移除智能体实例"""
        if self.agent_ is not None:
            agent_name = getattr(self.agent_, "name", "Unknown")
            self.agent_ = None
            logger.info(f"Removed agent: {agent_name}")
        else:
            logger.warning("No agent to remove")

    def get_agent_info(self) -> dict:
        """获取智能体信息

        Returns:
            智能体信息字典
        """
        if self.agent_ is None:
            return {"exists": False, "name": None, "type": None}

        return {
            "exists": True,
            "name": getattr(self.agent_, "name", "Unknown"),
            "type": type(self.agent_).__name__,
        }

    def get_status(self) -> dict:
        """获取管理器状态

        Returns:
            状态字典
        """
        return {
            "agent_registered": self.agent_exists(),
            "agent_info": self.get_agent_info(),
        }
