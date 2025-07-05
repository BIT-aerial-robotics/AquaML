"""Plugin Loader

This module provides plugin loading functionality for AquaML.
"""

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Type, List, Optional
from loguru import logger

from .interface import PluginInterface
try:
    from ..core.exceptions import PluginError
except ImportError:
    # Fallback for when used as standalone module
    class PluginError(Exception):
        pass


class PluginLoader:
    """Plugin loader for dynamically loading plugins"""
    
    def __init__(self):
        self._loaded_modules = {}
    
    def load_plugin_class(self, plugin_path: str) -> Type[PluginInterface]:
        """Load a plugin class from a module path
        
        Args:
            plugin_path: Path to the plugin module
            
        Returns:
            Plugin class
            
        Raises:
            PluginError: If plugin cannot be loaded
        """
        try:
            # Try to load as module path first
            if '.' in plugin_path or not plugin_path.endswith('.py'):
                return self._load_from_module_path(plugin_path)
            else:
                # Try to load from file path
                return self._load_from_file_path(plugin_path)
                
        except Exception as e:
            raise PluginError(f"Failed to load plugin from {plugin_path}: {e}")
    
    def _load_from_module_path(self, module_path: str) -> Type[PluginInterface]:
        """Load plugin from module path
        
        Args:
            module_path: Module path (e.g., 'package.module')
            
        Returns:
            Plugin class
        """
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Cache the module
            self._loaded_modules[module_path] = module
            
            # Find plugin class
            plugin_class = self._find_plugin_class(module)
            
            logger.debug(f"Loaded plugin class from module: {module_path}")
            return plugin_class
            
        except ImportError as e:
            raise PluginError(f"Cannot import module {module_path}: {e}")
    
    def _load_from_file_path(self, file_path: str) -> Type[PluginInterface]:
        """Load plugin from file path
        
        Args:
            file_path: File path to the plugin
            
        Returns:
            Plugin class
        """
        path = Path(file_path)
        
        if not path.exists():
            raise PluginError(f"Plugin file not found: {file_path}")
        
        if not path.suffix == '.py':
            raise PluginError(f"Plugin file must be a Python file: {file_path}")
        
        try:
            # Create module spec
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None:
                raise PluginError(f"Cannot create module spec for {file_path}")
            
            # Load module
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Add to sys.modules to make it importable
            sys.modules[path.stem] = module
            
            # Cache the module
            self._loaded_modules[file_path] = module
            
            # Find plugin class
            plugin_class = self._find_plugin_class(module)
            
            logger.debug(f"Loaded plugin class from file: {file_path}")
            return plugin_class
            
        except Exception as e:
            raise PluginError(f"Error loading plugin from {file_path}: {e}")
    
    def _find_plugin_class(self, module) -> Type[PluginInterface]:
        """Find plugin class in a module
        
        Args:
            module: Python module
            
        Returns:
            Plugin class
            
        Raises:
            PluginError: If no plugin class found
        """
        plugin_classes = []
        
        # Inspect all classes in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip the base interface class
            if obj is PluginInterface:
                continue
            
            # Check if it's a plugin class
            if (issubclass(obj, PluginInterface) and 
                obj.__module__ == module.__name__):
                plugin_classes.append(obj)
        
        if not plugin_classes:
            raise PluginError(f"No plugin class found in module {module.__name__}")
        
        if len(plugin_classes) > 1:
            logger.warning(f"Multiple plugin classes found in {module.__name__}, using first one")
        
        return plugin_classes[0]
    
    def load_plugins_from_directory(self, directory: str) -> List[Type[PluginInterface]]:
        """Load all plugins from a directory
        
        Args:
            directory: Directory path
            
        Returns:
            List of plugin classes
        """
        plugin_classes = []
        plugin_dir = Path(directory)
        
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory does not exist: {directory}")
            return plugin_classes
        
        if not plugin_dir.is_dir():
            logger.warning(f"Path is not a directory: {directory}")
            return plugin_classes
        
        # Find all Python files
        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            
            try:
                plugin_class = self.load_plugin_class(str(plugin_file))
                plugin_classes.append(plugin_class)
                logger.debug(f"Loaded plugin from {plugin_file}")
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_file}: {e}")
        
        return plugin_classes
    
    def reload_plugin(self, plugin_path: str) -> Type[PluginInterface]:
        """Reload a plugin
        
        Args:
            plugin_path: Path to the plugin
            
        Returns:
            Reloaded plugin class
        """
        # Remove from cache if exists
        if plugin_path in self._loaded_modules:
            module = self._loaded_modules[plugin_path]
            
            # Remove from sys.modules
            if hasattr(module, '__name__') and module.__name__ in sys.modules:
                del sys.modules[module.__name__]
            
            # Remove from cache
            del self._loaded_modules[plugin_path]
        
        # Load again
        return self.load_plugin_class(plugin_path)
    
    def unload_plugin(self, plugin_path: str) -> None:
        """Unload a plugin
        
        Args:
            plugin_path: Path to the plugin
        """
        if plugin_path in self._loaded_modules:
            module = self._loaded_modules[plugin_path]
            
            # Remove from sys.modules
            if hasattr(module, '__name__') and module.__name__ in sys.modules:
                del sys.modules[module.__name__]
            
            # Remove from cache
            del self._loaded_modules[plugin_path]
            
            logger.debug(f"Unloaded plugin: {plugin_path}")
    
    def get_loaded_modules(self) -> List[str]:
        """Get list of loaded module paths
        
        Returns:
            List of module paths
        """
        return list(self._loaded_modules.keys())
    
    def clear_cache(self) -> None:
        """Clear the module cache"""
        for plugin_path in list(self._loaded_modules.keys()):
            self.unload_plugin(plugin_path)
        
        logger.debug("Cleared plugin loader cache")
    
    def validate_plugin_class(self, plugin_class: Type[PluginInterface]) -> bool:
        """Validate a plugin class
        
        Args:
            plugin_class: Plugin class to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if it's a subclass of PluginInterface
            if not issubclass(plugin_class, PluginInterface):
                logger.error(f"Plugin class {plugin_class.__name__} does not inherit from PluginInterface")
                return False
            
            # Check if it can be instantiated
            instance = plugin_class()
            
            # Check if it has required properties
            if not hasattr(instance, 'plugin_info'):
                logger.error(f"Plugin class {plugin_class.__name__} missing plugin_info property")
                return False
            
            # Try to get plugin info
            plugin_info = instance.plugin_info
            if not plugin_info:
                logger.error(f"Plugin class {plugin_class.__name__} has invalid plugin_info")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating plugin class {plugin_class.__name__}: {e}")
            return False 