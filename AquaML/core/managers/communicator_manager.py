"""Communicator Manager for AquaML Framework

This module provides specialized management for communicator instances.
"""

from typing import Any, Callable
from loguru import logger


class CommunicatorManager:
    """Manager for communicator registration and retrieval"""
    
    def __init__(self):
        """Initialize the communicator manager"""
        self.communicator_ = None  # 通信器实例
        logger.debug("CommunicatorManager initialized")
    
    def register_communicator(self, communicator_cls: type) -> Callable:
        """注册通信器实例，方便集中管理
        
        Args:
            communicator_cls: 通信器类
            
        Returns:
            Wrapper function
        """
        def wrapper(*args, **kwargs):
            """注册通信器实例"""
            if self.communicator_ is not None:
                logger.error('currently do not support multiple communicators!')
                raise ValueError("communicator already exists!")
            
            self.communicator_ = communicator_cls(*args, **kwargs)
            
            comm_name = getattr(self.communicator_, 'name', communicator_cls.__name__)
            logger.info(f"Successfully registered communicator: {comm_name}")
            
            return self.communicator_
        
        return wrapper
    
    def get_communicator(self) -> Any:
        """获取通信器实例
        
        Returns:
            通信器实例
            
        Raises:
            ValueError: 如果通信器不存在
        """
        if self.communicator_ is None:
            logger.error("Communicator not exists!")
            raise ValueError("Communicator not exists!")
        
        return self.communicator_
    
    def set_communicator(self, communicator_instance: Any) -> None:
        """直接设置通信器实例
        
        Args:
            communicator_instance: 通信器实例
        """
        if self.communicator_ is not None:
            logger.warning("Communicator already exists, replacing it")
        
        self.communicator_ = communicator_instance
        comm_name = getattr(communicator_instance, 'name', 'Unknown')
        logger.info(f"Communicator set: {comm_name}")
    
    def communicator_exists(self) -> bool:
        """检查通信器是否存在
        
        Returns:
            True if communicator exists, False otherwise
        """
        return self.communicator_ is not None
    
    def remove_communicator(self) -> None:
        """移除通信器实例"""
        if self.communicator_ is not None:
            comm_name = getattr(self.communicator_, 'name', 'Unknown')
            self.communicator_ = None
            logger.info(f"Removed communicator: {comm_name}")
        else:
            logger.warning("No communicator to remove")
    
    def get_communicator_info(self) -> dict:
        """获取通信器信息
        
        Returns:
            通信器信息字典
        """
        if self.communicator_ is None:
            return {'exists': False, 'name': None, 'type': None}
        
        return {
            'exists': True,
            'name': getattr(self.communicator_, 'name', 'Unknown'),
            'type': type(self.communicator_).__name__
        }
    
    def get_status(self) -> dict:
        """获取管理器状态
        
        Returns:
            状态字典
        """
        return {
            'communicator_registered': self.communicator_exists(),
            'communicator_info': self.get_communicator_info()
        } 