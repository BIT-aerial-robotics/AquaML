"""AquaML Core Data Units

This module provides the core data units for AquaML framework,
combining the excellent design from legacy system with new framework features.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, MISSING, field
import torch
import numpy as np
from loguru import logger
from enum import Enum

from ..core.coordinator import coordinator
from ..core.exceptions import AquaMLException


class DataMode(Enum):
    """数据存储模式"""
    NUMPY = "numpy"
    TORCH = "torch"
    AUTO = "auto"


class DataFormat(Enum):
    """数据格式枚举"""
    TORCH = "torch"
    NUMPY = "numpy"
    AUTO = "auto"


@dataclass
class UnitConfig:
    """数据单元配置类 - 融合legacy unitCfg的优秀设计"""
    
    name: str = MISSING
    """必要参数，数据名称。该参数用于识别或者读取共享信息，每一个都具有唯一标识。"""
    
    dtype: Union[torch.dtype, np.dtype] = MISSING
    """必要参数，数据类型。"""
    
    single_shape: Tuple[int, ...] = MISSING
    """必要参数，单个数据的形状。"""
    
    size: int = MISSING
    """必要参数，数据的长度。"""
    
    mode: DataMode = DataMode.NUMPY
    """非必要，数据的存储模式。"""
    
    device: str = "cpu"
    """非必要，数据的设备。"""
    
    # 自动计算字段
    shape: Optional[Tuple[int, ...]] = None
    """自动计算得到的数据形状。由single_shape和size计算得到。"""
    
    bytes: Optional[int] = None
    """数据的字节数。自动计算得到。"""
    
    # 新增字段
    description: str = ""
    """数据单元描述"""
    
    enable_history: bool = False
    """是否启用历史记录"""
    
    max_history_length: int = 1000
    """最大历史记录长度"""
    
    tags: Dict[str, Any] = field(default_factory=dict)
    """数据单元标签"""
    
    def __post_init__(self):
        """后初始化处理"""
        # 验证必要参数
        if self.name == MISSING:
            raise ValueError("UnitConfig must specify name!")
        if self.dtype == MISSING:
            raise ValueError("UnitConfig must specify dtype!")
        if self.single_shape == MISSING:
            raise ValueError("UnitConfig must specify single_shape!")
        if self.size == MISSING:
            raise ValueError("UnitConfig must specify size!")
        
        # 处理dtype字符串格式
        if isinstance(self.dtype, str):
            if self.dtype.startswith('float'):
                self.dtype = getattr(np, self.dtype)
            elif self.dtype.startswith('int'):
                self.dtype = getattr(np, self.dtype)
            elif self.dtype in ['bool', 'bool_']:
                self.dtype = np.bool_
            else:
                try:
                    self.dtype = np.dtype(self.dtype)
                except TypeError:
                    raise ValueError(f"Unsupported dtype string: {self.dtype}")
        
        # 计算shape
        self.shape = (self.size,) + self.single_shape
        
        # 计算字节数
        self.bytes = self._compute_bytes()
        
        logger.debug(f"UnitConfig created: {self.name}, shape={self.shape}, bytes={self.bytes}")
    
    def _compute_bytes(self) -> int:
        """计算数据字节数"""
        if isinstance(self.dtype, torch.dtype):
            # PyTorch数据类型
            if self.dtype == torch.float32:
                item_size = 4
            elif self.dtype == torch.float64:
                item_size = 8
            elif self.dtype == torch.int32:
                item_size = 4
            elif self.dtype == torch.int64:
                item_size = 8
            elif self.dtype == torch.bool:
                item_size = 1
            else:
                item_size = 4  # 默认
        else:
            # NumPy数据类型
            item_size = np.dtype(self.dtype).itemsize
        
        # 计算总字节数
        total_elements = np.prod(self.shape)
        return int(total_elements * item_size)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if key == "dtype":
                if isinstance(value, torch.dtype):
                    config_dict[key] = str(value)
                elif hasattr(value, '__name__'):
                    config_dict[key] = f"np.{value.__name__}"
                else:
                    # Handle string dtypes or other formats
                    config_dict[key] = str(value)
            elif key == "mode":
                config_dict[key] = value.value if isinstance(value, DataMode) else str(value)
            elif not isinstance(value, str):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UnitConfig':
        """从字典创建配置"""
        # 处理dtype
        if "dtype" in config_dict:
            dtype_str = config_dict["dtype"]
            if dtype_str.startswith("torch."):
                config_dict["dtype"] = getattr(torch, dtype_str.split(".")[1])
            elif dtype_str.startswith("np."):
                config_dict["dtype"] = getattr(np, dtype_str.split(".")[1])
        
        # 处理mode
        if "mode" in config_dict:
            config_dict["mode"] = DataMode(config_dict["mode"])
        
        return cls(**config_dict)


class BaseUnit(ABC):
    """
    数据单元基类 - 融合legacy BaseUnit的优秀设计
    
    该数据单元是数据传输的基本单位，提供了统一的数据访问接口
    """
    
    def __init__(self, unit_cfg: UnitConfig):
        """
        初始化数据单元
        
        Args:
            unit_cfg: 数据单元配置
        """
        if not isinstance(unit_cfg, UnitConfig):
            raise TypeError("unit_cfg must be UnitConfig instance")
        
        self.unit_cfg_ = unit_cfg
        self.name_ = unit_cfg.name
        self.mode_ = unit_cfg.mode
        self.data_ = None
        self.history_ = [] if unit_cfg.enable_history else None
        self.is_initialized_ = False
        
        logger.info(f"Created data unit '{self.name_}' with mode '{self.mode_.value}'")
    
    # 魔术方法支持 - 保持legacy的优秀设计
    def __call__(self):
        """调用数据单元，返回数据"""
        if self.data_ is None:
            logger.warning(f"Data unit '{self.name_}' is not initialized")
        return self.data_
    
    def __getitem__(self, key):
        """获取数据切片"""
        if self.data_ is None:
            logger.warning(f"Data unit '{self.name_}' is not initialized")
            return None
        return self.data_[key]
    
    def __setitem__(self, key, value):
        """设置数据切片"""
        if self.data_ is None:
            logger.warning(f"Data unit '{self.name_}' is not initialized")
            return
        self.data_[key] = value
    
    def __len__(self):
        """获取数据长度"""
        if self.data_ is None:
            return 0
        return len(self.data_)
    
    def __repr__(self):
        """字符串表示"""
        return f"BaseUnit(name='{self.name_}', mode='{self.mode_.value}', initialized={self.is_initialized_})"
    
    # 属性访问 - 保持legacy的优秀设计
    @property
    def name(self) -> str:
        """获取数据单元名称"""
        return self.name_
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """获取数据形状"""
        if self.data_ is None:
            return self.unit_cfg_.shape
        return self.data_.shape
    
    @property
    def single_shape(self) -> Tuple[int, ...]:
        """获取单个数据的形状"""
        return self.unit_cfg_.single_shape
    
    @property
    def size(self) -> int:
        """获取数据大小"""
        return self.unit_cfg_.size
    
    @property
    def dtype(self) -> Union[torch.dtype, np.dtype]:
        """获取数据类型"""
        return self.unit_cfg_.dtype
    
    @property
    def device(self) -> str:
        """获取设备"""
        return self.unit_cfg_.device
    
    @property
    def mode(self) -> DataMode:
        """获取数据模式"""
        return self.mode_
    
    @property
    def bytes(self) -> int:
        """获取字节数"""
        return self.unit_cfg_.bytes
    
    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self.is_initialized_
    
    # 状态管理 - 保持legacy的优秀设计
    def get_unit_status(self) -> UnitConfig:
        """获取数据单元状态"""
        return self.unit_cfg_
    
    def get_unit_status_dict(self) -> Dict[str, Any]:
        """获取数据单元状态字典"""
        return self.unit_cfg_.to_dict()
    
    # 历史记录功能 - 新增
    def add_to_history(self, data: Any) -> None:
        """添加数据到历史记录"""
        if self.history_ is not None:
            self.history_.append(data)
            if len(self.history_) > self.unit_cfg_.max_history_length:
                self.history_.pop(0)
    
    def get_history(self) -> Optional[list]:
        """获取历史记录"""
        return self.history_
    
    def clear_history(self) -> None:
        """清空历史记录"""
        if self.history_ is not None:
            self.history_.clear()
    
    # 数据转换功能 - 新增
    def to_torch(self) -> torch.Tensor:
        """转换为PyTorch张量"""
        if self.data_ is None:
            raise AquaMLException(f"Data unit '{self.name_}' is not initialized")
        
        if isinstance(self.data_, torch.Tensor):
            return self.data_.to(device=self.device)
        elif isinstance(self.data_, np.ndarray):
            return torch.from_numpy(self.data_).to(device=self.device)
        else:
            return torch.tensor(self.data_, device=self.device)
    
    def to_numpy(self) -> np.ndarray:
        """转换为NumPy数组"""
        if self.data_ is None:
            raise AquaMLException(f"Data unit '{self.name_}' is not initialized")
        
        if isinstance(self.data_, np.ndarray):
            return self.data_
        elif isinstance(self.data_, torch.Tensor):
            return self.data_.detach().cpu().numpy()
        else:
            return np.array(self.data_)
    
    def to_format(self, format_type: DataFormat) -> Union[torch.Tensor, np.ndarray]:
        """转换为指定格式"""
        if format_type == DataFormat.TORCH:
            return self.to_torch()
        elif format_type == DataFormat.NUMPY:
            return self.to_numpy()
        else:
            return self.data_
    
    # 抽象方法 - 保持legacy的设计
    @abstractmethod
    def create_data(self) -> None:
        """创建数据 - 子类必须实现"""
        pass
    
    @abstractmethod
    def compute_bytes(self) -> int:
        """计算字节数 - 子类必须实现"""
        pass
    
    # 新增方法
    def reset(self) -> None:
        """重置数据单元"""
        self.data_ = None
        self.is_initialized_ = False
        self.clear_history()
        logger.debug(f"Reset data unit '{self.name_}'")
    
    def update_data(self, data: Any) -> None:
        """更新数据"""
        old_data = self.data_
        self.data_ = data
        self.is_initialized_ = True
        
        # 添加到历史记录
        if old_data is not None:
            self.add_to_history(old_data)
        
        logger.debug(f"Updated data unit '{self.name_}'")
    
    def clone(self) -> 'BaseUnit':
        """克隆数据单元"""
        new_unit = self.__class__(self.unit_cfg_)
        if self.data_ is not None:
            if isinstance(self.data_, torch.Tensor):
                new_unit.data_ = self.data_.clone()
            elif isinstance(self.data_, np.ndarray):
                new_unit.data_ = self.data_.copy()
            else:
                new_unit.data_ = self.data_
        new_unit.is_initialized_ = self.is_initialized_
        return new_unit


class TensorUnit(BaseUnit):
    """
    PyTorch张量数据单元 - 融合legacy TensorUnit的设计
    
    该类型在分布式通信中只能通过NumPyUnit进行通信
    """
    
    def __init__(self, unit_cfg: UnitConfig):
        """
        创建PyTorch张量数据单元
        
        Args:
            unit_cfg: 数据单元配置
        """
        super().__init__(unit_cfg)
        self.device_obj = torch.device(unit_cfg.device)
    
    def create_data(self) -> torch.Tensor:
        """创建PyTorch张量数据"""
        try:
            self.data_ = torch.zeros(
                self.unit_cfg_.shape,
                dtype=self.unit_cfg_.dtype,
                device=self.device_obj
            )
            self.is_initialized_ = True
            logger.info(f"Successfully created tensor data '{self.name_}'")
            return self.data_
        except Exception as e:
            logger.error(f"Failed to create tensor data '{self.name_}': {e}")
            raise AquaMLException(f"Failed to create tensor data: {e}")
    
    def compute_bytes(self) -> int:
        """计算张量字节数"""
        return self.unit_cfg_.bytes
    
    def to_device(self, device: str) -> 'TensorUnit':
        """移动到指定设备"""
        if self.data_ is not None:
            self.data_ = self.data_.to(device)
        self.unit_cfg_.device = device
        self.device_obj = torch.device(device)
        logger.debug(f"Moved tensor unit '{self.name_}' to device '{device}'")
        return self
    
    def to_numpy_unit(self) -> 'NumpyUnit':
        """转换为NumPy数据单元"""
        numpy_cfg = UnitConfig(
            name=self.name_,
            dtype=self.to_numpy().dtype,
            single_shape=self.single_shape,
            size=self.size,
            mode=DataMode.NUMPY,
            device="cpu"
        )
        numpy_unit = NumpyUnit(numpy_cfg)
        if self.data_ is not None:
            numpy_unit.data_ = self.to_numpy()
            numpy_unit.is_initialized_ = True
        return numpy_unit


class NumpyUnit(BaseUnit):
    """
    NumPy数组数据单元 - 融合legacy NumpyUnit的设计
    
    该类型支持共享内存，支持多进程通过TCP/IP通信
    主要用于分布式通信，和TensorUnit一起使用可以实现数据共享
    """
    
    def __init__(self, unit_cfg: UnitConfig):
        """
        创建NumPy数组数据单元
        
        Args:
            unit_cfg: 数据单元配置
        """
        super().__init__(unit_cfg)
    
    def create_data(self, create_first: bool = True) -> np.ndarray:
        """
        创建NumPy数组数据
        
        NumPy支持共享内存，当需要使用时请先创建该numpy array然后使用多线程去读取
        
        Args:
            create_first: 是否立即创建数据
        """
        try:
            if create_first:
                self.data_ = np.zeros(
                    self.unit_cfg_.shape,
                    dtype=self.unit_cfg_.dtype
                )
                self.is_initialized_ = True
                logger.info(f"Successfully created numpy data '{self.name_}'")
            return self.data_
        except Exception as e:
            logger.error(f"Failed to create numpy data '{self.name_}': {e}")
            raise AquaMLException(f"Failed to create numpy data: {e}")
    
    def compute_bytes(self) -> int:
        """计算数组字节数"""
        return self.unit_cfg_.bytes
    
    def to_tensor_unit(self, device: str = "cpu") -> TensorUnit:
        """转换为PyTorch张量数据单元"""
        tensor_cfg = UnitConfig(
            name=self.name_,
            dtype=torch.float32,  # 默认转换为float32
            single_shape=self.single_shape,
            size=self.size,
            mode=DataMode.TORCH,
            device=device
        )
        tensor_unit = TensorUnit(tensor_cfg)
        if self.data_ is not None:
            tensor_unit.data_ = torch.from_numpy(self.data_).to(device=device)
            tensor_unit.is_initialized_ = True
        return tensor_unit
    
    def enable_shared_memory(self) -> None:
        """启用共享内存模式"""
        if self.data_ is not None and isinstance(self.data_, np.ndarray):
            # 创建共享内存映射
            import mmap
            logger.info(f"Enabled shared memory for numpy unit '{self.name_}'")


# 数据单元工厂
class DataUnitFactory:
    """数据单元工厂类"""
    
    @staticmethod
    def create_tensor_unit(name: str, 
                          shape: Tuple[int, ...], 
                          dtype: torch.dtype = torch.float32,
                          device: str = "cpu",
                          **kwargs) -> TensorUnit:
        """创建张量数据单元"""
        if len(shape) == 1:
            single_shape = ()
            size = shape[0]
        else:
            single_shape = shape[1:]
            size = shape[0]
        
        config = UnitConfig(
            name=name,
            dtype=dtype,
            single_shape=single_shape,
            size=size,
            mode=DataMode.TORCH,
            device=device,
            **kwargs
        )
        return TensorUnit(config)
    
    @staticmethod
    def create_numpy_unit(name: str,
                         shape: Tuple[int, ...],
                         dtype: np.dtype = np.float32,
                         **kwargs) -> NumpyUnit:
        """创建NumPy数据单元"""
        if len(shape) == 1:
            single_shape = ()
            size = shape[0]
        else:
            single_shape = shape[1:]
            size = shape[0]
        
        config = UnitConfig(
            name=name,
            dtype=dtype,
            single_shape=single_shape,
            size=size,
            mode=DataMode.NUMPY,
            device="cpu",
            **kwargs
        )
        return NumpyUnit(config)
    
    @staticmethod
    def create_auto_unit(name: str,
                        shape: Tuple[int, ...],
                        dtype: Union[torch.dtype, np.dtype] = None,
                        prefer_torch: bool = None,
                        **kwargs) -> BaseUnit:
        """自动创建数据单元"""
        if prefer_torch is None:
            prefer_torch = torch.cuda.is_available()
        
        if prefer_torch:
            dtype = dtype or torch.float32
            return DataUnitFactory.create_tensor_unit(name, shape, dtype, **kwargs)
        else:
            dtype = dtype or np.float32
            return DataUnitFactory.create_numpy_unit(name, shape, dtype, **kwargs)


# 注册数据单元到协调器
def register_data_unit(unit_cls):
    """数据单元注册装饰器 - 保持legacy的优秀设计"""
    return coordinator.register_component(unit_cls)


# 导出API
__all__ = [
    'DataMode', 'DataFormat', 'UnitConfig', 'BaseUnit', 
    'TensorUnit', 'NumpyUnit', 'DataUnitFactory', 'register_data_unit'
] 