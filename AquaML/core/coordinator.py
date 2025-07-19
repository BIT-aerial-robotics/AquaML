"""AquaML Coordinator

This module provides the main coordinator for the AquaML framework.
"""

from typing import Dict, Any, Optional
from loguru import logger
import torch
from torch.nn import Module
import sys
import os

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


def configure_loguru_logging(log_level: str = "INFO", log_file: Optional[str] = None, file_system_instance=None) -> None:
    """Configure loguru logging for AquaML
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, logs to stdout only
        file_system_instance: Optional FileSystem instance for directory management
    """
    # Remove default handler
    logger.remove()
    
    # Console handler with INFO level by default
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File handler if log_file is specified
    if log_file:
        # Create log directory using FileSystem if available
        log_dir = os.path.dirname(log_file)
        if log_dir:
            if file_system_instance:
                try:
                    file_system_instance.ensureDir(log_dir)
                except Exception:
                    # Fallback to direct creation on any error
                    if not os.path.exists(log_dir):
                        os.makedirs(log_dir, exist_ok=True)
            else:
                # Fallback to direct creation if FileSystem is not available
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
            
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
    logger.info(f"Loguru logging configured with level: {log_level}")


# Basic loguru configuration for initial logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
    backtrace=True,
    diagnose=True
)


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

        # Initialize default FileSystem (required component)
        self._initialize_default_file_system()
        
        # Logging configuration
        self._log_level = "INFO"
        self._log_file = None
        
        # Configure loguru logging with FileSystem available
        self._configure_initial_logging()

        self._initialized = True

        self.tensor_tool = TensorTool()
        
        logger.info("AquaML Coordinator initialized")

    def _initialize_default_file_system(self) -> None:
        """Initialize default FileSystem as required component"""
        try:
            from ..utils.file_system import DefaultFileSystem
            
            # Create default workspace directory
            default_workspace = os.path.join(os.getcwd(), "aquaml_workspace")
            file_system = DefaultFileSystem(default_workspace)
            file_system.initFolder()
            
            # Register with manager
            self.file_system_manager.set_file_system(file_system)
            
            logger.info(f"Default FileSystem initialized with workspace: {default_workspace}")
        except Exception as e:
            logger.error(f"Failed to initialize default FileSystem: {e}")
            raise AquaMLException(f"FileSystem initialization failed: {e}")

    def _configure_initial_logging(self) -> None:
        """Configure initial logging with FileSystem available"""
        try:
            file_system = self.file_system_manager.get_file_system() if self.file_system_manager.file_system_exists() else None
            configure_loguru_logging(self._log_level, self._log_file, file_system)
        except Exception as e:
            # Fallback to basic logging configuration
            configure_loguru_logging(self._log_level, self._log_file, None)
            logger.warning(f"Failed to configure logging with FileSystem: {e}")

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

            # Initialize logging configuration
            self._initialize_logging_config(config)

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

    def _initialize_logging_config(self, config: Optional[Dict[str, Any]]) -> None:
        """Initialize logging configuration from config
        
        Args:
            config: Configuration dictionary that may contain logging settings
        """
        if not config:
            return
            
        logging_config = config.get("logging", {})
        
        # Update log level if specified
        if "level" in logging_config:
            self._log_level = logging_config["level"].upper()
            
        # Update log file if specified
        if "file" in logging_config:
            self._log_file = logging_config["file"]
            
        # Reconfigure loguru with new settings using FileSystem
        if logging_config:
            file_system = self.file_system_manager.get_file_system() if self.file_system_manager.file_system_exists() else None
            configure_loguru_logging(self._log_level, self._log_file, file_system)
            logger.info(f"Logging reconfigured from config - Level: {self._log_level}, File: {self._log_file}")

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

    # ==================== Model Management Interface ====================
    def registerModel(self, model: Module, model_name: str) -> None:
        """Register model with the coordinator

        Args:
            model: Model instance
            model_name: Model name
        """
        self.model_manager.register_model(model, model_name)

    def getModel(self, model_name: str) -> Dict[str, Any]:
        """Get model by name

        Args:
            model_name: Model name

        Returns:
            Model dictionary containing model instance and status
        """
        return self.model_manager.get_model(model_name)

    def get_model_manager(self) -> ModelManager:
        """Get model manager instance

        Returns:
            ModelManager instance
        """
        return self.model_manager

    # ==================== Environment Management Interface ====================
    def registerEnv(self, env_cls):
        """Register environment with the coordinator

        Args:
            env_cls: Environment class

        Returns:
            Wrapper function
        """
        return self.environment_manager.register_env(env_cls)

    def getEnv(self):
        """Get environment instance

        Returns:
            Environment instance
        """
        return self.environment_manager.get_env()

    def get_environment_manager(self) -> EnvironmentManager:
        """Get environment manager instance

        Returns:
            EnvironmentManager instance
        """
        return self.environment_manager

    # ==================== Agent Management Interface ====================
    def registerAgent(self, agent_cls):
        """Register agent with the coordinator

        Args:
            agent_cls: Agent class

        Returns:
            Wrapper function
        """
        return self.agent_manager.register_agent(agent_cls)

    def getAgent(self):
        """Get agent instance

        Returns:
            Agent instance
        """
        return self.agent_manager.get_agent()

    def get_agent_manager(self) -> AgentManager:
        """Get agent manager instance

        Returns:
            AgentManager instance
        """
        return self.agent_manager

    # ==================== Data Unit Management Interface ====================
    def registerDataUnit(self, data_unit_cls):
        """Register data unit with the coordinator

        Args:
            data_unit_cls: Data unit class

        Returns:
            Wrapper function
        """
        return self.data_unit_manager.register_data_unit(data_unit_cls)

    def getDataUnit(self, unit_name: str):
        """Get data unit by name

        Args:
            unit_name: Data unit name

        Returns:
            Data unit instance
        """
        return self.data_unit_manager.get_data_unit(unit_name)

    def get_data_unit_manager(self) -> DataUnitManager:
        """Get data unit manager instance

        Returns:
            DataUnitManager instance
        """
        return self.data_unit_manager

    # ==================== File System Management Interface ====================
    def registerFileSystem(self, file_system_cls):
        """Register file system with the coordinator

        Args:
            file_system_cls: File system class

        Returns:
            Wrapper function
        """
        return self.file_system_manager.register_file_system(file_system_cls)

    def getFileSystem(self):
        """Get file system instance

        Returns:
            File system instance
        """
        return self.file_system_manager.get_file_system()

    def get_file_system_manager(self) -> FileSystemManager:
        """Get file system manager instance

        Returns:
            FileSystemManager instance
        """
        return self.file_system_manager

    # ==================== Communicator Management Interface ====================
    def registerCommunicator(self, communicator_cls):
        """Register communicator with the coordinator

        Args:
            communicator_cls: Communicator class

        Returns:
            Wrapper function
        """
        return self.communicator_manager.register_communicator(communicator_cls)

    def getCommunicator(self):
        """Get communicator instance

        Returns:
            Communicator instance
        """
        return self.communicator_manager.get_communicator()

    def get_communicator_manager(self) -> CommunicatorManager:
        """Get communicator manager instance

        Returns:
            CommunicatorManager instance
        """
        return self.communicator_manager

    # ==================== Data Manager Interface ====================
    def register_data_manager(self, data_manager_cls):
        """Register data manager with the coordinator

        Args:
            data_manager_cls: Data manager class

        Returns:
            Wrapper function
        """
        return self.data_manager.register_data_manager(data_manager_cls)

    def get_data_manager(self):
        """Get data manager instance

        Returns:
            Data manager instance
        """
        return self.data_manager.get_data_manager()

    def get_data_manager_manager(self) -> DataManager:
        """Get data manager manager instance

        Returns:
            DataManager instance
        """
        return self.data_manager

    # ==================== Runner Management Interface ====================
    def registerRunner(self, runner_name: Optional[str] = None) -> str:
        """Register runner with the coordinator

        Args:
            runner_name: Runner name, if None will auto-generate with timestamp

        Returns:
            The actual runner name used
        """
        actual_runner_name = self.runner_manager.register_runner(runner_name)

        # Configure runner in file system if available
        if self.file_system_manager.file_system_exists():
            try:
                self.file_system_manager.config_runner(actual_runner_name)
            except Exception as e:
                logger.warning(f"Failed to configure runner in file system: {e}")
        else:
            logger.warning("File system not available for runner configuration")
            
        return actual_runner_name

    def getRunner(self) -> str:
        """Get runner name

        Returns:
            Runner name
        """
        return self.runner_manager.get_runner()

    def get_runner_manager(self) -> RunnerManager:
        """Get runner manager instance

        Returns:
            RunnerManager instance
        """
        return self.runner_manager

    # ==================== Data Management ====================
    def saveDataUnitInfo(self):
        """Save data unit information"""
        try:
            runner_name = self.runner_manager.get_runner()
            self.data_unit_manager.save_data_unit_info(
                runner_name, self.file_system_manager
            )
        except Exception as e:
            logger.error(f"Failed to save data unit info: {e}")

    # ==================== Plugin and Config Management ====================
    def get_plugin_manager(self):
        """Get plugin manager instance"""
        return self._plugin_manager

    def get_config_manager(self):
        """Get config manager instance"""
        return self._config_manager

    # ==================== Logging Management ====================
    def set_log_level(self, level: str) -> None:
        """Set logging level
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self._log_level = level.upper()
        file_system = self.file_system_manager.get_file_system() if self.file_system_manager.file_system_exists() else None
        configure_loguru_logging(self._log_level, self._log_file, file_system)
        logger.info(f"Log level changed to: {self._log_level}")

    def set_log_file(self, file_path: Optional[str]) -> None:
        """Set log file path
        
        Args:
            file_path: Path to log file. If None, disables file logging
        """
        self._log_file = file_path
        file_system = self.file_system_manager.get_file_system() if self.file_system_manager.file_system_exists() else None
        configure_loguru_logging(self._log_level, self._log_file, file_system)
        if file_path:
            logger.info(f"Log file set to: {file_path}")
        else:
            logger.info("File logging disabled")

    def get_log_level(self) -> str:
        """Get current log level
        
        Returns:
            Current log level
        """
        return self._log_level

    def get_log_file(self) -> Optional[str]:
        """Get current log file path
        
        Returns:
            Current log file path or None if file logging is disabled
        """
        return self._log_file

    def configure_logging(self, level: str = "INFO", file_path: Optional[str] = None) -> None:
        """Configure logging settings
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            file_path: Optional log file path
        """
        self._log_level = level.upper()
        self._log_file = file_path
        file_system = self.file_system_manager.get_file_system() if self.file_system_manager.file_system_exists() else None
        configure_loguru_logging(self._log_level, self._log_file, file_system)
        logger.info(f"Logging configured - Level: {self._log_level}, File: {self._log_file}")

    def get_logging_config(self) -> Dict[str, Any]:
        """Get current logging configuration
        
        Returns:
            Dictionary containing current logging configuration
        """
        return {
            "level": self._log_level,
            "file": self._log_file
        }

    # ==================== Utility Methods ====================
    def list_components(self) -> Dict[str, int]:
        """List all registered components

        Returns:
            Dictionary of component types and counts
        """
        components = {
            "models": self.model_manager.get_models_count(),
            "data_units": self.data_unit_manager.get_data_units_count(),
            "environment": 1 if self.environment_manager.env_exists() else 0,
            "agent": 1 if self.agent_manager.agent_exists() else 0,
            "file_system": 1 if self.file_system_manager.file_system_exists() else 0,
            "communicator": 1 if self.communicator_manager.communicator_exists() else 0,
            "data_manager": 1 if self.data_manager.data_manager_exists() else 0,
            "runner": 1 if self.runner_manager.runner_exists() else 0,
        }
        return components

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status

        Returns:
            Status dictionary
        """
        return {
            "initialized": self._initialized,
            "components": self.list_components(),
            "device_info": self.get_device_info(),
            "logging_config": self.get_logging_config(),
            "runner_name": (
                self.runner_manager.get_runner()
                if self.runner_manager.runner_exists()
                else None
            ),
        }

    def get_all_managers_status(self) -> Dict[str, Any]:
        """Get status of all managers

        Returns:
            Dictionary containing status of all managers
        """
        return {
            "model_manager": self.model_manager.get_status(),
            "environment_manager": self.environment_manager.get_status(),
            "agent_manager": self.agent_manager.get_status(),
            "data_unit_manager": self.data_unit_manager.get_status(),
            "file_system_manager": self.file_system_manager.get_status(),
            "communicator_manager": self.communicator_manager.get_status(),
            "data_manager": self.data_manager.get_status(),
            "runner_manager": self.runner_manager.get_status(),
        }

    def shutdown(self) -> None:
        """Shutdown the coordinator"""
        try:
            # Clear all managers
            self.model_manager.clear_models()
            self.environment_manager.remove_env()
            self.agent_manager.remove_agent()
            self.data_unit_manager.clear_data_units()
            self.file_system_manager.remove_file_system()
            self.communicator_manager.remove_communicator()
            self.data_manager.remove_data_manager()
            self.runner_manager.remove_runner()

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
