"""AquaML Coordinator

This module provides the main coordinator for the AquaML framework.
"""

from typing import Dict, Any, Optional
from loguru import logger
import torch
from torch.nn import Module

from .device_info import GPUInfo, detect_gpu_devices, get_optimal_device
from .exceptions import AquaMLException
from .tensor_tool import TensorTool
from .managers import (
    ModelManager,
    EnvironmentManager,
    AgentManager,
    DataUnitManager,
    FileSystemManager,
    CommunicatorManager,
    DataManager,
    RunnerManager,
)

# Global device variables
global_device = None
available_devices = []


class AquaMLCoordinator:
    """Main coordinator for AquaML framework

    This class serves as the central hub for managing components,
    configuration, and device management in the AquaML framework.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation"""
        if not cls._instance:
            # Welcome message
            print(
                "\033[1;34m🌊 Welcome to AquaML - Advanced Machine Learning Framework! 🌊\033[0m"
            )
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the coordinator"""
        if self._initialized:
            return

        # Initialize all specialized managers
        self.model_manager = ModelManager()
        self.environment_manager = EnvironmentManager()
        self.agent_manager = AgentManager()
        self.data_unit_manager = DataUnitManager()
        self.file_system_manager = FileSystemManager()
        self.communicator_manager = CommunicatorManager()
        self.data_manager = DataManager()
        self.runner_manager = RunnerManager()

        # Plugin and config managers (optional)
        self._plugin_manager = None
        self._config_manager = None

        self._initialized = True

        self.tensor_tool = TensorTool()
        logger.info("AquaML Coordinator initialized")

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the coordinator with configuration

        Args:
            config: Configuration dictionary
        """
        try:
            # Initialize device management
            self._initialize_device_management(config)

            # Initialize plugin manager
            self._initialize_plugin_manager()

            # Initialize configuration manager
            self._initialize_config_manager(config)

            # Load plugins if configured
            if config and "plugins" in config:
                self._load_plugins(config["plugins"])

            logger.info("AquaML Coordinator fully initialized")

        except Exception as e:
            logger.error(f"Failed to initialize AquaML Coordinator: {e}")
            raise AquaMLException(f"Coordinator initialization failed: {e}")

    def _initialize_device_management(
        self, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize device management system

        Args:
            config: Configuration dictionary that may contain device specifications
        """
        global global_device, available_devices

        # Detect available GPU devices
        available_devices = detect_gpu_devices()

        # Set device based on config or auto-detection
        user_device = None
        if config and "device" in config:
            user_device = config["device"]

        global_device = self._select_device(user_device)
        logger.info(f"Selected device: {global_device}")
        logger.info(f"Available GPU devices: {len(available_devices)}")

    def _select_device(self, user_device: Optional[str] = None) -> str:
        """Select appropriate device based on user preference and availability

        Args:
            user_device: User-specified device (e.g., 'cuda:0', 'cpu')

        Returns:
            Selected device string
        """
        global available_devices

        # If user specified a device, validate and use it
        if user_device:
            if user_device == "cpu":
                logger.info("Using user-specified CPU device")
                return "cpu"

            # Check if user specified GPU exists
            for gpu in available_devices:
                if gpu.device_id == user_device:
                    logger.info(f"Using user-specified device: {user_device}")
                    return user_device

            logger.warning(
                f"User-specified device '{user_device}' not available. Using auto-selection."
            )

        # Auto-selection: prefer GPU 0 if available
        if available_devices:
            optimal_device = get_optimal_device(available_devices)
            logger.info(f"Auto-selecting device: {optimal_device}")
            return optimal_device
        else:
            logger.info("No GPU available, using CPU")
            return "cpu"

    def get_device(self) -> str:
        """Get current device

        Returns:
            Current device string
        """
        global global_device
        if global_device is None:
            global_device = "cpu"
        return global_device

    def get_torch_device(self):
        """Get PyTorch device object

        Returns:
            torch.device object
        """
        return torch.device(self.get_device())

    def set_device(self, device: str) -> bool:
        """Set device for computation

        Args:
            device: Device string (e.g., 'cuda:0', 'cpu')

        Returns:
            True if device was set successfully, False otherwise
        """
        global global_device, available_devices

        if device == "cpu":
            global_device = "cpu"
            logger.info("Device set to: cpu")
            return True

        # Check if GPU device exists
        for gpu in available_devices:
            if gpu.device_id == device:
                global_device = device
                logger.info(f"Device set to: {device}")
                return True

        logger.error(
            f"Device '{device}' not available. Available devices: {self.get_available_devices()}"
        )
        return False

    def get_available_devices(self) -> list:
        """Get list of available devices

        Returns:
            List of available device strings
        """
        global available_devices
        devices = ["cpu"]
        devices.extend([gpu.device_id for gpu in available_devices])
        return devices

    def validate_device(self, device: str) -> bool:
        """Validate device string

        Args:
            device: Device string (e.g., 'cuda:0', 'cpu')

        Returns:
            bool: True if device is valid, False otherwise
        """
        global available_devices

        # CPU is always available
        if device == "cpu":
            return True

        # Check if the device is in the list of available GPU devices
        for gpu in available_devices:
            if gpu.device_id == device:
                return True

        logger.error(
            f"Device '{device}' not available. Available devices: {self.get_available_devices()}"
        )
        return False

    def is_gpu_available(self) -> bool:
        """Check if GPU is available

        Returns:
            True if GPU is available, False otherwise
        """
        global available_devices
        return len(available_devices) > 0

    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information

        Returns:
            Dictionary containing device information
        """
        global global_device, available_devices

        info = {
            "current_device": self.get_device(),
            "available_devices": self.get_available_devices(),
            "gpu_available": self.is_gpu_available(),
            "gpu_count": len(available_devices),
        }

        if available_devices:
            info["gpu_details"] = [gpu.to_dict() for gpu in available_devices]

        return info

    def _initialize_plugin_manager(self) -> None:
        """Initialize plugin manager"""
        try:
            from ..plugins.manager import PluginManager

            self._plugin_manager = PluginManager()
            logger.debug("Plugin manager initialized")
        except ImportError:
            logger.warning("Plugin manager not available")

    def _initialize_config_manager(self, config: Optional[Dict[str, Any]]) -> None:
        """Initialize configuration manager"""
        try:
            from ..config.manager import ConfigManager

            self._config_manager = ConfigManager()
            if config:
                self._config_manager.load_config(config)
            logger.debug("Config manager initialized")
        except ImportError:
            logger.warning("Config manager not available")

    def _load_plugins(self, plugin_configs: Dict[str, Any]) -> None:
        """Load plugins from configuration

        Args:
            plugin_configs: Plugin configuration dictionary
        """
        if not self._plugin_manager:
            logger.warning("Plugin manager not available, skipping plugin loading")
            return

        for plugin_name, plugin_config in plugin_configs.items():
            try:
                self._plugin_manager.load_plugin(
                    plugin_config.get("path", ""), plugin_config.get("config", {})
                )
                logger.info(f"Loaded plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_name}: {e}")

        # ==================== Model Registration ====================

    def registerModel(self, model: Module, model_name: str):
        """将模型注册到模型字典中

        Args:
            model: 模型实例
            model_name: 模型名称
        """
        self.model_manager.register_model(model, model_name)

    def getModel(self, model_name: str) -> Dict[str, Any]:
        """获取模型实例和当前状态

        Args:
            model_name: 模型名称

        Returns:
            模型字典，包含 'model' 和 'status' 键
        """
        return self.model_manager.get_model(model_name)

    # ==================== Environment Registration ====================
        def registerEnv(self, env_cls):
        """注册环境实例，方便集中管理
        
        Args:
            env_cls: 环境类
        """
        return self.environment_manager.register_env(env_cls)

    def getEnv(self):
        """获取环境实例
        
        Returns:
            环境实例
        """
        return self.environment_manager.get_env()

    # ==================== Agent Registration ====================
        def registerAgent(self, agent_cls):
        """注册智能体实例，方便集中管理
        
        Args:
            agent_cls: 智能体类
        """
        return self.agent_manager.register_agent(agent_cls)

    def getAgent(self):
        """获取智能体实例
        
        Returns:
            智能体实例
        """
        return self.agent_manager.get_agent()

    # ==================== Data Unit Registration ====================
        def registerDataUnit(self, data_unit_cls):
        """注册数据单元实例，方便集中管理
        
        Args:
            data_unit_cls: 数据单元类
        """
        return self.data_unit_manager.register_data_unit(data_unit_cls)

    def getDataUnit(self, unit_name: str):
        """获取数据单元实例
        
        Args:
            unit_name: 数据单元名称
            
        Returns:
            数据单元实例
        """
        return self.data_unit_manager.get_data_unit(unit_name)

    # ==================== File System Registration ====================
        def registerFileSystem(self, file_system_cls):
        """注册文件系统实例，方便集中管理
        
        Args:
            file_system_cls: 文件系统类
        """
        return self.file_system_manager.register_file_system(file_system_cls)

    def getFileSystem(self):
        """获取文件系统实例
        
        Returns:
            文件系统实例
        """
        return self.file_system_manager.get_file_system()

    # ==================== Communicator Registration ====================
    def registerCommunicator(self, communicator_cls):
        """注册通信器实例，方便集中管理

        Args:
            communicator_cls: 通信器类
        """

        def wrapper(*args, **kwargs):
            """注册通信器实例"""
            if self.communicator_ is not None:
                logger.error("currently do not support multiple communicators!")
                raise ValueError("communicator already exists!")

            self.communicator_ = communicator_cls(*args, **kwargs)

            comm_name = getattr(self.communicator_, "name", "Unknown")
            logger.info(f"Successfully registered communicator: {comm_name}")

            return self.communicator_

        return wrapper

    def getCommunicator(self):
        """获取通信器实例

        Returns:
            通信器实例
        """
        if self.communicator_ is None:
            logger.error("Communicator not exists!")
            raise ValueError("Communicator not exists!")

        return self.communicator_

    # ==================== Runner Registration ====================
    def registerRunner(self, runner_name: str):
        """注册runner名称，用于记录当前运行的runner名称

        Args:
            runner_name: runner名称
        """
        self.runner_name_ = runner_name

        if self.file_system_ is None:
            logger.warning("file system not exists!")
            logger.warning("do not forget to configure runner in file system!")
        else:
            self.file_system_.configRunner(runner_name)
            logger.info(f"Successfully registered runner: {runner_name}")

    def getRunner(self) -> str:
        """获取runner名称

        Returns:
            runner名称
        """
        if self.runner_name_ is None:
            logger.error("Runner not exists!")
            raise ValueError("Runner not exists!")

        return self.runner_name_

    # ==================== Data Manager Registration ====================
    def register_data_manager(self, data_manager_cls):
        """注册数据管理器类

        Args:
            data_manager_cls: 数据管理器类

        Returns:
            Wrapper function
        """

        def wrapper(*args, **kwargs):
            data_manager_instance = data_manager_cls(*args, **kwargs)
            self._data_manager = data_manager_instance
            logger.info("Successfully registered data manager")
            return data_manager_instance

        return wrapper

    def get_data_manager(self):
        """获取注册的数据管理器"""
        if self._data_manager is None:
            logger.error("Data manager not exists!")
            raise ValueError("Data manager not exists!")
        return self._data_manager

    # ==================== Data Management ====================
    def saveDataUnitInfo(self):
        """保存数据单元信息"""
        if self.runner_name_ is None:
            logger.error("runner name not exists!")
            raise ValueError("runner name not exists!")

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
        if self.file_system_ is not None:
            self.file_system_.saveDataUnit(
                runner_name=self.runner_name_, data_unit_status=save_dict
            )
            logger.info(f"Saved data unit info for {len(save_dict)} units")
        else:
            logger.warning("File system not available, cannot save data unit info")

    # ==================== Plugin and Config Management ====================
    def get_plugin_manager(self):
        """获取插件管理器"""
        return self._plugin_manager

    def get_config_manager(self):
        """获取配置管理器"""
        return self._config_manager

    # ==================== Utility Methods ====================
    def list_components(self) -> Dict[str, int]:
        """列出所有已注册的组件

        Returns:
            组件类型及数量的字典
        """
        components = {
            "models": len(self.models_dict_),
            "data_units": len(self.data_units_),
            "environment": 1 if self.env_ is not None else 0,
            "agent": 1 if self.agent_ is not None else 0,
            "file_system": 1 if self.file_system_ is not None else 0,
            "communicator": 1 if self.communicator_ is not None else 0,
            "data_manager": 1 if self._data_manager is not None else 0,
            "runner": 1 if self.runner_name_ is not None else 0,
        }
        return components

    def get_status(self) -> Dict[str, Any]:
        """获取协调器状态

        Returns:
            状态字典
        """
        return {
            "initialized": self._initialized,
            "components": self.list_components(),
            "device_info": self.get_device_info(),
            "runner_name": self.runner_name_,
        }

    def shutdown(self) -> None:
        """关闭协调器"""
        try:
            # 清空所有组件引用
            self.models_dict_.clear()
            self.data_units_.clear()
            self.env_ = None
            self.agent_ = None
            self.file_system_ = None
            self.communicator_ = None
            self._data_manager = None
            self.runner_name_ = None

            logger.info("AquaML Coordinator shutdown completed")

        except Exception as e:
            logger.error(f"Error during coordinator shutdown: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


# Global coordinator instance
coordinator = AquaMLCoordinator()


def get_coordinator() -> AquaMLCoordinator:
    """Get the global coordinator instance"""
    return coordinator
