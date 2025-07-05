"""Component Registry System

This module provides a centralized registry for managing components in the AquaML framework.
"""

from typing import Dict, Any, Type, Optional, List, Callable
from loguru import logger
from .exceptions import RegistryError


class ComponentRegistry:
    """Central registry for managing components in AquaML"""
    
    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._component_types: Dict[str, Type] = {}
        self._component_metadata: Dict[str, Dict[str, Any]] = {}
        self._initialization_callbacks: Dict[str, List[Callable]] = {}
    
    def register(self, 
                 name: str, 
                 component: Any, 
                 component_type: Optional[Type] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 replace: bool = False) -> None:
        """Register a component
        
        Args:
            name: Component name
            component: Component instance
            component_type: Component type class
            metadata: Additional metadata
            replace: Whether to replace existing component
        """
        if name in self._components and not replace:
            raise RegistryError(f"Component '{name}' already registered")
        
        # Store component
        self._components[name] = component
        
        # Store type information
        if component_type:
            self._component_types[name] = component_type
        else:
            self._component_types[name] = type(component)
        
        # Store metadata
        self._component_metadata[name] = metadata or {}
        
        # Execute initialization callbacks
        if name in self._initialization_callbacks:
            for callback in self._initialization_callbacks[name]:
                try:
                    callback(component)
                except Exception as e:
                    logger.error(f"Error executing initialization callback for {name}: {e}")
        
        logger.info(f"Registered component: {name}")
    
    def get(self, name: str, default: Any = None) -> Any:
        """Get a component by name
        
        Args:
            name: Component name
            default: Default value if component not found
            
        Returns:
            Component instance or default value
        """
        return self._components.get(name, default)
    
    def get_strict(self, name: str) -> Any:
        """Get a component by name (strict mode)
        
        Args:
            name: Component name
            
        Returns:
            Component instance
            
        Raises:
            RegistryError: If component not found
        """
        if name not in self._components:
            raise RegistryError(f"Component '{name}' not found")
        return self._components[name]
    
    def unregister(self, name: str) -> None:
        """Unregister a component
        
        Args:
            name: Component name
        """
        if name in self._components:
            del self._components[name]
        if name in self._component_types:
            del self._component_types[name]
        if name in self._component_metadata:
            del self._component_metadata[name]
        if name in self._initialization_callbacks:
            del self._initialization_callbacks[name]
        
        logger.info(f"Unregistered component: {name}")
    
    def has(self, name: str) -> bool:
        """Check if component exists
        
        Args:
            name: Component name
            
        Returns:
            True if component exists, False otherwise
        """
        return name in self._components
    
    def list_components(self, component_type: Optional[Type] = None) -> List[str]:
        """List all registered components
        
        Args:
            component_type: Filter by component type
            
        Returns:
            List of component names
        """
        if component_type is None:
            return list(self._components.keys())
        
        return [name for name, comp_type in self._component_types.items() 
                if issubclass(comp_type, component_type)]
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get component metadata
        
        Args:
            name: Component name
            
        Returns:
            Component metadata
        """
        return self._component_metadata.get(name, {})
    
    def set_metadata(self, name: str, metadata: Dict[str, Any]) -> None:
        """Set component metadata
        
        Args:
            name: Component name
            metadata: Metadata to set
        """
        if name not in self._components:
            raise RegistryError(f"Component '{name}' not found")
        
        self._component_metadata[name] = metadata
    
    def add_initialization_callback(self, component_name: str, callback: Callable) -> None:
        """Add initialization callback for a component
        
        Args:
            component_name: Component name
            callback: Callback function to execute when component is registered
        """
        if component_name not in self._initialization_callbacks:
            self._initialization_callbacks[component_name] = []
        
        self._initialization_callbacks[component_name].append(callback)
    
    def clear(self) -> None:
        """Clear all registered components"""
        self._components.clear()
        self._component_types.clear()
        self._component_metadata.clear()
        self._initialization_callbacks.clear()
        logger.info("Cleared all registered components")
    
    def __len__(self) -> int:
        """Return number of registered components"""
        return len(self._components)
    
    def __contains__(self, name: str) -> bool:
        """Check if component exists (using 'in' operator)"""
        return name in self._components
    
    def __iter__(self):
        """Iterate over component names"""
        return iter(self._components.keys())


# Global registry instance
_global_registry = ComponentRegistry()


def get_global_registry() -> ComponentRegistry:
    """Get the global component registry"""
    return _global_registry


def register_component(name: str, component: Any, **kwargs) -> None:
    """Register a component to the global registry"""
    _global_registry.register(name, component, **kwargs)


def get_component(name: str, default: Any = None) -> Any:
    """Get a component from the global registry"""
    return _global_registry.get(name, default) 