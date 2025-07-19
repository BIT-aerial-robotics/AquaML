"""Data Unit Manager for AquaML Framework

This module provides specialized management for data unit instances with
support for (num_env, steps, dims) format data collection.
"""

from typing import Any, Dict, Callable, Optional, List, Union
import numpy as np
from loguru import logger


class DataUnitManager:
    """Manager for data unit registration and retrieval with enhanced
    support for (num_env, steps, dims) format data collection"""

    def __init__(self):
        """Initialize the data unit manager"""
        self.data_units_ = {}  # 记录数据单元实例
        self.data_buffers_ = {}  # 缓存数据，格式：(num_env, steps, dims)
        self.buffer_configs_ = {}  # 缓存配置
        self.current_step_ = {}  # 每个buffer的当前步数
        self.callback_functions_ = {}  # 数据满时的回调函数
        logger.debug("DataUnitManager initialized with buffer support")

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

    def setup_data_buffer(self, 
                         buffer_name: str,
                         num_envs: int,
                         max_steps: int,
                         data_shape: tuple,
                         dtype: np.dtype = np.float32,
                         callback_fn: Optional[Callable] = None) -> None:
        """设置数据缓存，格式为(num_env, steps, dims)
        
        Args:
            buffer_name: 缓存名称
            num_envs: 环境数量
            max_steps: 最大步数
            data_shape: 单个数据的形状
            dtype: 数据类型
            callback_fn: 缓存满时的回调函数
        """
        full_shape = (num_envs, max_steps) + data_shape
        self.data_buffers_[buffer_name] = np.zeros(full_shape, dtype=dtype)
        self.buffer_configs_[buffer_name] = {
            'num_envs': num_envs,
            'max_steps': max_steps,
            'data_shape': data_shape,
            'dtype': dtype
        }
        self.current_step_[buffer_name] = 0
        if callback_fn:
            self.callback_functions_[buffer_name] = callback_fn
        
        logger.info(f"Setup data buffer '{buffer_name}' with shape {full_shape}")
    
    def add_data_to_buffer(self, 
                          buffer_name: str,
                          data: np.ndarray,
                          env_indices: Optional[Union[int, List[int]]] = None) -> bool:
        """向缓存添加数据
        
        Args:
            buffer_name: 缓存名称
            data: 数据，形状应为(num_envs, *data_shape)或(*data_shape,)
            env_indices: 环境索引，如果为None则使用所有环境
            
        Returns:
            是否触发了回调（缓存已满）
        """
        if buffer_name not in self.data_buffers_:
            logger.error(f"Buffer '{buffer_name}' not found")
            return False
            
        current_step = self.current_step_[buffer_name]
        max_steps = self.buffer_configs_[buffer_name]['max_steps']
        
        if current_step >= max_steps:
            logger.warning(f"Buffer '{buffer_name}' is full, resetting")
            self.current_step_[buffer_name] = 0
            current_step = 0
            
        # 处理数据维度
        if data.ndim == len(self.buffer_configs_[buffer_name]['data_shape']):
            # 单环境数据，扩展为多环境
            data = data[np.newaxis, ...]
        
        if env_indices is None:
            self.data_buffers_[buffer_name][:, current_step] = data
        else:
            if isinstance(env_indices, int):
                env_indices = [env_indices]
            if data.shape[0] == 1:
                # 广播到指定环境
                for idx in env_indices:
                    self.data_buffers_[buffer_name][idx, current_step] = data[0]
            else:
                self.data_buffers_[buffer_name][env_indices, current_step] = data[env_indices]
            
        self.current_step_[buffer_name] += 1
        
        # 检查是否满了
        if self.current_step_[buffer_name] >= max_steps:
            if buffer_name in self.callback_functions_:
                try:
                    self.callback_functions_[buffer_name](self.data_buffers_[buffer_name])
                    logger.debug(f"Triggered callback for buffer '{buffer_name}'")
                    return True
                except Exception as e:
                    logger.error(f"Error in callback for buffer '{buffer_name}': {e}")
                    
        return False
    
    def get_buffer_data(self, 
                       buffer_name: str,
                       copy: bool = True) -> Optional[np.ndarray]:
        """获取缓存数据
        
        Args:
            buffer_name: 缓存名称
            copy: 是否返回副本
            
        Returns:
            缓存数据，形状为(num_env, steps, dims)
        """
        if buffer_name not in self.data_buffers_:
            logger.error(f"Buffer '{buffer_name}' not found")
            return None
            
        current_step = self.current_step_[buffer_name]
        if current_step == 0:
            logger.warning(f"Buffer '{buffer_name}' is empty")
            return None
            
        data = self.data_buffers_[buffer_name][:, :current_step]
        return data.copy() if copy else data
    
    def clear_buffer(self, buffer_name: str) -> None:
        """清空指定缓存
        
        Args:
            buffer_name: 缓存名称
        """
        if buffer_name in self.data_buffers_:
            self.current_step_[buffer_name] = 0
            logger.debug(f"Cleared buffer '{buffer_name}'")
    
    def get_buffer_status(self, buffer_name: str) -> Optional[Dict[str, Any]]:
        """获取缓存状态
        
        Args:
            buffer_name: 缓存名称
            
        Returns:
            缓存状态字典
        """
        if buffer_name not in self.data_buffers_:
            return None
            
        config = self.buffer_configs_[buffer_name]
        current_step = self.current_step_[buffer_name]
        
        return {
            'buffer_name': buffer_name,
            'shape': self.data_buffers_[buffer_name].shape,
            'current_step': current_step,
            'max_steps': config['max_steps'],
            'progress': current_step / config['max_steps'] if config['max_steps'] > 0 else 0,
            'is_full': current_step >= config['max_steps'],
            'has_callback': buffer_name in self.callback_functions_
        }
    
    def list_buffers(self) -> List[str]:
        """列出所有缓存名称
        
        Returns:
            缓存名称列表
        """
        return list(self.data_buffers_.keys())
    
    def clear_all_buffers(self) -> None:
        """清空所有缓存"""
        for buffer_name in self.data_buffers_:
            self.clear_buffer(buffer_name)
        logger.info("Cleared all buffers")
    
    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态

        Returns:
            状态字典
        """
        buffer_status = {}
        for buffer_name in self.data_buffers_:
            buffer_status[buffer_name] = self.get_buffer_status(buffer_name)
            
        return {
            "total_data_units": self.get_data_units_count(),
            "data_unit_names": self.list_data_units(),
            "total_buffers": len(self.data_buffers_),
            "buffer_names": self.list_buffers(),
            "buffer_status": buffer_status
        }
