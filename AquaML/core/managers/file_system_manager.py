"""File System Manager for AquaML Framework

This module provides specialized management for file system instances.
"""

from typing import Any, Callable
from loguru import logger


class FileSystemManager:
    """Manager for file system registration and retrieval"""
    
    def __init__(self):
        """Initialize the file system manager"""
        self.file_system_ = None  # 文件系统实例
        logger.debug("FileSystemManager initialized")
    
    def register_file_system(self, file_system_cls: type) -> Callable:
        """注册文件系统实例，方便集中管理
        
        Args:
            file_system_cls: 文件系统类
            
        Returns:
            Wrapper function
        """
        def wrapper(*args, **kwargs):
            """注册文件系统实例"""
            if self.file_system_ is not None:
                logger.error("file system already exists!")
                raise ValueError("file system already exists!")
            
            self.file_system_ = file_system_cls(*args, **kwargs)
            
            logger.info("Successfully registered file system")
            
            return self.file_system_
        
        return wrapper
    
    def get_file_system(self) -> Any:
        """获取文件系统实例
        
        Returns:
            文件系统实例
            
        Raises:
            ValueError: 如果文件系统不存在
        """
        if self.file_system_ is None:
            logger.error("File system not exists!")
            raise ValueError("File system not exists!")
        
        return self.file_system_
    
    def set_file_system(self, file_system_instance: Any) -> None:
        """直接设置文件系统实例
        
        Args:
            file_system_instance: 文件系统实例
        """
        if self.file_system_ is not None:
            logger.warning("File system already exists, replacing it")
        
        self.file_system_ = file_system_instance
        logger.info("File system set")
    
    def file_system_exists(self) -> bool:
        """检查文件系统是否存在
        
        Returns:
            True if file system exists, False otherwise
        """
        return self.file_system_ is not None
    
    def remove_file_system(self) -> None:
        """移除文件系统实例"""
        if self.file_system_ is not None:
            self.file_system_ = None
            logger.info("Removed file system")
        else:
            logger.warning("No file system to remove")
    
    def config_runner(self, runner_name: str) -> None:
        """配置运行器
        
        Args:
            runner_name: 运行器名称
        """
        if self.file_system_ is None:
            logger.error("File system not exists!")
            raise ValueError("File system not exists!")
        
        try:
            self.file_system_.configRunner(runner_name)
            logger.info(f"Configured runner: {runner_name}")
        except AttributeError:
            logger.warning("File system does not have configRunner method")
    
    def get_file_system_info(self) -> dict:
        """获取文件系统信息
        
        Returns:
            文件系统信息字典
        """
        if self.file_system_ is None:
            return {'exists': False, 'type': None}
        
        return {
            'exists': True,
            'type': type(self.file_system_).__name__
        }
    
    def get_status(self) -> dict:
        """获取管理器状态
        
        Returns:
            状态字典
        """
        return {
            'file_system_registered': self.file_system_exists(),
            'file_system_info': self.get_file_system_info()
        } 