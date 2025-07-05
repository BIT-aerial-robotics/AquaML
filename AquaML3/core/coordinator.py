"""AquaML Coordinator

This module provides the main coordinator for the AquaML framework.
"""

from typing import Dict, Any, Optional
from loguru import logger

from .registry import ComponentRegistry
from .lifecycle import LifecycleManager
from .exceptions import AquaMLException


class AquaMLCoordinator:
    """Main coordinator for AquaML framework
    
    This class serves as the central hub for managing components,
    configuration, and lifecycle in the AquaML framework.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation"""
        if not cls._instance:
            # Welcome message
            print("\033[1;34m🌊 Welcome to AquaML - Advanced Machine Learning Framework! 🌊\033[0m")
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the coordinator"""
        if self._initialized:
            return
        
        # Core managers
        self.registry = ComponentRegistry()
        self.lifecycle_manager = LifecycleManager()
        
        # Plugin manager will be initialized later
        self._plugin_manager = None
        self._config_manager = None
        
        # Component references
        self._environment = None
        self._agent = None
        self._data_manager = None
        
        self._initialized = True
        logger.info("AquaML Coordinator initialized")
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the coordinator with configuration
        
        Args:
            config: Configuration dictionary
        """
        try:
            # Initialize lifecycle manager
            self.lifecycle_manager.initialize(config)
            
            # Initialize plugin manager
            self._initialize_plugin_manager()
            
            # Initialize configuration manager
            self._initialize_config_manager(config)
            
            # Load plugins if configured
            if config and 'plugins' in config:
                self._load_plugins(config['plugins'])
            
            logger.info("AquaML Coordinator fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize AquaML Coordinator: {e}")
            raise AquaMLException(f"Coordinator initialization failed: {e}")
    
    def shutdown(self) -> None:
        """Shutdown the coordinator"""
        try:
            # Shutdown lifecycle manager
            self.lifecycle_manager.shutdown()
            
            # Clear registry
            self.registry.clear()
            
            # Reset component references
            self._environment = None
            self._agent = None
            self._data_manager = None
            
            logger.info("AquaML Coordinator shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during coordinator shutdown: {e}")
    
    def _initialize_plugin_manager(self) -> None:
        """Initialize plugin manager"""
        try:
            from ..plugins.manager import PluginManager
            self._plugin_manager = PluginManager()
            self.registry.register('plugin_manager', self._plugin_manager)
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
            self.registry.register('config_manager', self._config_manager)
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
                    plugin_config.get('path', ''),
                    plugin_config.get('config', {})
                )
                logger.info(f"Loaded plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_name}: {e}")
    
    # Component registration methods
    def register_environment(self, env_cls):
        """Register environment class
        
        Args:
            env_cls: Environment class
            
        Returns:
            Wrapper function
        """
        def wrapper(*args, **kwargs):
            env_instance = env_cls(*args, **kwargs)
            self._environment = env_instance
            self.registry.register('environment', env_instance)
            self.lifecycle_manager.set_component_state('environment', 'running')
            logger.info(f"Registered environment: {getattr(env_instance, 'name', 'Unknown')}")
            return env_instance
        return wrapper
    
    def register_agent(self, agent_cls):
        """Register agent class
        
        Args:
            agent_cls: Agent class
            
        Returns:
            Wrapper function
        """
        def wrapper(*args, **kwargs):
            if self._agent is not None:
                logger.warning("Agent already registered, replacing...")
            
            agent_instance = agent_cls(*args, **kwargs)
            self._agent = agent_instance
            self.registry.register('agent', agent_instance, replace=True)
            self.lifecycle_manager.set_component_state('agent', 'running')
            logger.info(f"Registered agent: {getattr(agent_instance, 'name', 'Unknown')}")
            return agent_instance
        return wrapper
    
    def register_data_manager(self, data_manager_cls):
        """Register data manager class
        
        Args:
            data_manager_cls: Data manager class
            
        Returns:
            Wrapper function
        """
        def wrapper(*args, **kwargs):
            data_manager_instance = data_manager_cls(*args, **kwargs)
            self._data_manager = data_manager_instance
            self.registry.register('data_manager', data_manager_instance)
            self.lifecycle_manager.set_component_state('data_manager', 'running')
            logger.info("Registered data manager")
            return data_manager_instance
        return wrapper
    
    # Component access methods
    def get_environment(self):
        """Get registered environment"""
        return self._environment or self.registry.get('environment')
    
    def get_agent(self):
        """Get registered agent"""
        return self._agent or self.registry.get('agent')
    
    def get_data_manager(self):
        """Get registered data manager"""
        return self._data_manager or self.registry.get('data_manager')
    
    def get_plugin_manager(self):
        """Get plugin manager"""
        return self._plugin_manager
    
    def get_config_manager(self):
        """Get configuration manager"""
        return self._config_manager
    
    # Utility methods
    def is_component_registered(self, component_name: str) -> bool:
        """Check if component is registered"""
        return self.registry.has(component_name)
    
    def get_component_state(self, component_name: str) -> Optional[str]:
        """Get component state"""
        return self.lifecycle_manager.get_component_state(component_name)
    
    def list_components(self) -> list:
        """List all registered components"""
        return self.registry.list_components()
    
    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status"""
        return {
            'initialized': self._initialized,
            'lifecycle_initialized': self.lifecycle_manager.is_initialized,
            'registered_components': len(self.registry),
            'component_states': self.lifecycle_manager.get_all_component_states(),
            'components': self.registry.list_components()
        }
    
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