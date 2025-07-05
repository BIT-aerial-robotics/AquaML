"""Plugin Registry

This module provides plugin registry functionality for AquaML.
"""

from typing import Dict, List, Optional
from loguru import logger

from .interface import PluginInfo, PluginType
try:
    from ..core.exceptions import PluginError
except ImportError:
    # Fallback for when used as standalone module
    class PluginError(Exception):
        pass


class PluginRegistry:
    """Registry for managing plugin metadata"""
    
    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}
        self._plugins_by_type: Dict[PluginType, List[str]] = {
            plugin_type: [] for plugin_type in PluginType
        }
    
    def register_plugin(self, plugin_info: PluginInfo) -> None:
        """Register a plugin
        
        Args:
            plugin_info: Plugin information
        """
        if plugin_info.name in self._plugins:
            raise PluginError(f"Plugin {plugin_info.name} already registered")
        
        self._plugins[plugin_info.name] = plugin_info
        self._plugins_by_type[plugin_info.plugin_type].append(plugin_info.name)
        
        logger.debug(f"Registered plugin info: {plugin_info.name}")
    
    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a plugin
        
        Args:
            plugin_name: Plugin name
        """
        if plugin_name not in self._plugins:
            logger.warning(f"Plugin {plugin_name} not found in registry")
            return
        
        plugin_info = self._plugins[plugin_name]
        
        # Remove from main registry
        del self._plugins[plugin_name]
        
        # Remove from type-based registry
        if plugin_name in self._plugins_by_type[plugin_info.plugin_type]:
            self._plugins_by_type[plugin_info.plugin_type].remove(plugin_name)
        
        logger.debug(f"Unregistered plugin info: {plugin_name}")
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get plugin information
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            Plugin info or None if not found
        """
        return self._plugins.get(plugin_name)
    
    def has_plugin(self, plugin_name: str) -> bool:
        """Check if plugin is registered
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            True if plugin is registered, False otherwise
        """
        return plugin_name in self._plugins
    
    def list_plugins(self, plugin_type: Optional[PluginType] = None) -> List[str]:
        """List registered plugins
        
        Args:
            plugin_type: Filter by plugin type
            
        Returns:
            List of plugin names
        """
        if plugin_type is None:
            return list(self._plugins.keys())
        
        return self._plugins_by_type.get(plugin_type, []).copy()
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInfo]:
        """Get plugins by type
        
        Args:
            plugin_type: Plugin type
            
        Returns:
            List of plugin info objects
        """
        plugin_names = self._plugins_by_type.get(plugin_type, [])
        return [self._plugins[name] for name in plugin_names]
    
    def search_plugins(self, 
                      name_pattern: Optional[str] = None,
                      plugin_type: Optional[PluginType] = None,
                      author: Optional[str] = None) -> List[PluginInfo]:
        """Search for plugins
        
        Args:
            name_pattern: Pattern to match plugin names
            plugin_type: Filter by plugin type
            author: Filter by author
            
        Returns:
            List of matching plugin info objects
        """
        results = []
        
        for plugin_info in self._plugins.values():
            # Check type filter
            if plugin_type and plugin_info.plugin_type != plugin_type:
                continue
            
            # Check name pattern
            if name_pattern and name_pattern.lower() not in plugin_info.name.lower():
                continue
            
            # Check author filter
            if author and author.lower() not in plugin_info.author.lower():
                continue
            
            results.append(plugin_info)
        
        return results
    
    def get_plugin_dependencies(self, plugin_name: str) -> List[str]:
        """Get plugin dependencies
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            List of dependency names
        """
        plugin_info = self.get_plugin_info(plugin_name)
        if not plugin_info:
            return []
        
        return plugin_info.dependencies.copy()
    
    def get_dependent_plugins(self, plugin_name: str) -> List[str]:
        """Get plugins that depend on the given plugin
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            List of dependent plugin names
        """
        dependents = []
        
        for name, plugin_info in self._plugins.items():
            if plugin_name in plugin_info.dependencies:
                dependents.append(name)
        
        return dependents
    
    def validate_dependencies(self, plugin_name: str) -> bool:
        """Validate plugin dependencies
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            True if all dependencies are met, False otherwise
        """
        plugin_info = self.get_plugin_info(plugin_name)
        if not plugin_info:
            return False
        
        for dep in plugin_info.dependencies:
            if not self.has_plugin(dep):
                logger.error(f"Plugin {plugin_name} depends on {dep} which is not registered")
                return False
        
        return True
    
    def get_registry_stats(self) -> Dict[str, int]:
        """Get registry statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_plugins': len(self._plugins),
            **{f'{plugin_type.value}_plugins': len(plugins) 
               for plugin_type, plugins in self._plugins_by_type.items()}
        }
    
    def clear(self) -> None:
        """Clear all registered plugins"""
        self._plugins.clear()
        for plugin_list in self._plugins_by_type.values():
            plugin_list.clear()
        logger.debug("Cleared plugin registry")
    
    def __len__(self) -> int:
        """Get number of registered plugins"""
        return len(self._plugins)
    
    def __contains__(self, plugin_name: str) -> bool:
        """Check if plugin is registered (using 'in' operator)"""
        return plugin_name in self._plugins
    
    def __iter__(self):
        """Iterate over plugin names"""
        return iter(self._plugins.keys()) 