"""Plugin Manager

This module provides the plugin manager for AquaML.
"""

from typing import Dict, Any, Type, Optional, List
import importlib
import importlib.util
import inspect
from pathlib import Path
from loguru import logger

from .interface import PluginInterface, PluginInfo, PluginType
from .registry import PluginRegistry
from .loader import PluginLoader
try:
    from ..core.exceptions import PluginError
except ImportError:
    # Fallback for when used as standalone module
    class PluginError(Exception):
        pass


class PluginManager:
    """Plugin manager for AquaML
    
    This class manages the loading, initialization, and lifecycle of plugins.
    """
    
    def __init__(self):
        self.registry = PluginRegistry()
        self.loader = PluginLoader()
        self._initialized_plugins: Dict[str, PluginInterface] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
    
    def register_plugin(self, 
                       plugin_class: Type[PluginInterface], 
                       config: Optional[Dict[str, Any]] = None) -> None:
        """Register a plugin class
        
        Args:
            plugin_class: Plugin class to register
            config: Plugin configuration
        """
        if not issubclass(plugin_class, PluginInterface):
            raise PluginError(f"Plugin class must inherit from PluginInterface")
        
        # Create plugin instance
        plugin_instance = plugin_class()
        plugin_info = plugin_instance.plugin_info
        
        # Check if plugin is already registered
        if self.registry.has_plugin(plugin_info.name):
            logger.warning(f"Plugin {plugin_info.name} is already registered")
            return
        
        # Check dependencies
        if not self._check_dependencies(plugin_info.dependencies):
            raise PluginError(f"Plugin {plugin_info.name} has unmet dependencies")
        
        # Validate configuration
        if config and not plugin_instance.validate_config(config):
            raise PluginError(f"Invalid configuration for plugin {plugin_info.name}")
        
        # Register plugin
        self.registry.register_plugin(plugin_info)
        
        # Initialize plugin
        try:
            plugin_instance.initialize(config or {})
            self._initialized_plugins[plugin_info.name] = plugin_instance
            self._plugin_configs[plugin_info.name] = config or {}
            logger.info(f"Successfully registered plugin: {plugin_info.name}")
        except Exception as e:
            # Clean up on failure
            self.registry.unregister_plugin(plugin_info.name)
            raise PluginError(f"Failed to initialize plugin {plugin_info.name}: {e}")
    
    def load_plugin(self, plugin_path: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Load a plugin from a path
        
        Args:
            plugin_path: Path to plugin module or file
            config: Plugin configuration
        """
        try:
            plugin_class = self.loader.load_plugin_class(plugin_path)
            self.register_plugin(plugin_class, config)
        except Exception as e:
            raise PluginError(f"Failed to load plugin from {plugin_path}: {e}")
    
    def load_plugins_from_directory(self, directory: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Load all plugins from a directory
        
        Args:
            directory: Directory path containing plugins
            config: Default configuration for plugins
        """
        plugin_dir = Path(directory)
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory {directory} does not exist")
            return
        
        # Find all Python files in the directory
        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            
            try:
                plugin_path = str(plugin_file.with_suffix(''))
                self.load_plugin(plugin_path, config)
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_file}: {e}")
    
    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Get a plugin by name
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found
        """
        return self._initialized_plugins.get(name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInterface]:
        """Get plugins by type
        
        Args:
            plugin_type: Plugin type
            
        Returns:
            List of plugin instances
        """
        return [
            plugin for plugin in self._initialized_plugins.values()
            if plugin.plugin_info.plugin_type == plugin_type
        ]
    
    def list_plugins(self, plugin_type: Optional[PluginType] = None) -> List[str]:
        """List all registered plugins
        
        Args:
            plugin_type: Filter by plugin type
            
        Returns:
            List of plugin names
        """
        if plugin_type is None:
            return list(self._initialized_plugins.keys())
        
        return [
            name for name, plugin in self._initialized_plugins.items()
            if plugin.plugin_info.plugin_type == plugin_type
        ]
    
    def unregister_plugin(self, name: str) -> None:
        """Unregister a plugin
        
        Args:
            name: Plugin name
        """
        if name in self._initialized_plugins:
            plugin = self._initialized_plugins[name]
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin {name}: {e}")
            
            del self._initialized_plugins[name]
            del self._plugin_configs[name]
            self.registry.unregister_plugin(name)
            logger.info(f"Unregistered plugin: {name}")
    
    def reload_plugin(self, name: str) -> None:
        """Reload a plugin
        
        Args:
            name: Plugin name
        """
        if name not in self._initialized_plugins:
            raise PluginError(f"Plugin {name} not found")
        
        # Get plugin info and config
        plugin_info = self.registry.get_plugin_info(name)
        config = self._plugin_configs.get(name, {})
        
        # Unregister current plugin
        self.unregister_plugin(name)
        
        # Reload plugin
        self.load_plugin(plugin_info.entry_point, config)
    
    def get_plugin_info(self, name: str) -> Optional[PluginInfo]:
        """Get plugin information
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin info or None if not found
        """
        return self.registry.get_plugin_info(name)
    
    def get_plugin_config(self, name: str) -> Dict[str, Any]:
        """Get plugin configuration
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin configuration
        """
        return self._plugin_configs.get(name, {})
    
    def update_plugin_config(self, name: str, config: Dict[str, Any]) -> None:
        """Update plugin configuration
        
        Args:
            name: Plugin name
            config: New configuration
        """
        if name not in self._initialized_plugins:
            raise PluginError(f"Plugin {name} not found")
        
        plugin = self._initialized_plugins[name]
        if not plugin.validate_config(config):
            raise PluginError(f"Invalid configuration for plugin {name}")
        
        self._plugin_configs[name] = config
        # Note: Plugin needs to be reloaded to apply new config
        logger.info(f"Updated configuration for plugin {name}")
    
    def cleanup_all_plugins(self) -> None:
        """Clean up all plugins"""
        for plugin_name in list(self._initialized_plugins.keys()):
            self.unregister_plugin(plugin_name)
    
    def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if plugin dependencies are met
        
        Args:
            dependencies: List of dependency names
            
        Returns:
            True if all dependencies are met, False otherwise
        """
        for dep in dependencies:
            if dep not in self._initialized_plugins:
                logger.error(f"Dependency {dep} not found")
                return False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get plugin manager status
        
        Returns:
            Status dictionary
        """
        return {
            'total_plugins': len(self._initialized_plugins),
            'plugins_by_type': {
                plugin_type.value: len(self.get_plugins_by_type(plugin_type))
                for plugin_type in PluginType
            },
            'plugin_names': list(self._initialized_plugins.keys())
        }
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup_all_plugins() 