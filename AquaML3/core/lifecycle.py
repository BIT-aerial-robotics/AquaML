"""Lifecycle Management System

This module provides lifecycle management for AquaML components.
"""

from typing import List, Callable, Dict, Any, Optional
from loguru import logger
from .exceptions import LifecycleError


class LifecycleManager:
    """Manages the lifecycle of AquaML components"""
    
    def __init__(self):
        self._initialized: bool = False
        self._startup_callbacks: List[Callable] = []
        self._shutdown_callbacks: List[Callable] = []
        self._component_states: Dict[str, str] = {}
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the lifecycle manager
        
        Args:
            config: Configuration dictionary
        """
        if self._initialized:
            logger.warning("LifecycleManager already initialized")
            return
        
        logger.info("Initializing AquaML Lifecycle Manager")
        
        # Execute startup callbacks
        for callback in self._startup_callbacks:
            try:
                callback(config)
                logger.debug(f"Executed startup callback: {callback.__name__}")
            except Exception as e:
                logger.error(f"Error executing startup callback {callback.__name__}: {e}")
                raise LifecycleError(f"Failed to execute startup callback: {e}")
        
        self._initialized = True
        logger.info("AquaML Lifecycle Manager initialized successfully")
    
    def shutdown(self) -> None:
        """Shutdown the lifecycle manager"""
        if not self._initialized:
            logger.warning("LifecycleManager not initialized")
            return
        
        logger.info("Shutting down AquaML Lifecycle Manager")
        
        # Execute shutdown callbacks in reverse order
        for callback in reversed(self._shutdown_callbacks):
            try:
                callback()
                logger.debug(f"Executed shutdown callback: {callback.__name__}")
            except Exception as e:
                logger.error(f"Error executing shutdown callback {callback.__name__}: {e}")
                # Continue with other shutdown callbacks even if one fails
        
        self._initialized = False
        self._component_states.clear()
        logger.info("AquaML Lifecycle Manager shutdown completed")
    
    def add_startup_callback(self, callback: Callable) -> None:
        """Add a startup callback
        
        Args:
            callback: Function to call during startup
        """
        if not callable(callback):
            raise LifecycleError("Startup callback must be callable")
        
        self._startup_callbacks.append(callback)
        logger.debug(f"Added startup callback: {callback.__name__}")
    
    def add_shutdown_callback(self, callback: Callable) -> None:
        """Add a shutdown callback
        
        Args:
            callback: Function to call during shutdown
        """
        if not callable(callback):
            raise LifecycleError("Shutdown callback must be callable")
        
        self._shutdown_callbacks.append(callback)
        logger.debug(f"Added shutdown callback: {callback.__name__}")
    
    def remove_startup_callback(self, callback: Callable) -> None:
        """Remove a startup callback
        
        Args:
            callback: Function to remove
        """
        if callback in self._startup_callbacks:
            self._startup_callbacks.remove(callback)
            logger.debug(f"Removed startup callback: {callback.__name__}")
    
    def remove_shutdown_callback(self, callback: Callable) -> None:
        """Remove a shutdown callback
        
        Args:
            callback: Function to remove
        """
        if callback in self._shutdown_callbacks:
            self._shutdown_callbacks.remove(callback)
            logger.debug(f"Removed shutdown callback: {callback.__name__}")
    
    def set_component_state(self, component_name: str, state: str) -> None:
        """Set the state of a component
        
        Args:
            component_name: Name of the component
            state: State string (e.g., 'initializing', 'running', 'stopped')
        """
        self._component_states[component_name] = state
        logger.debug(f"Component {component_name} state set to: {state}")
    
    def get_component_state(self, component_name: str) -> Optional[str]:
        """Get the state of a component
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component state or None if not found
        """
        return self._component_states.get(component_name)
    
    def is_component_running(self, component_name: str) -> bool:
        """Check if a component is running
        
        Args:
            component_name: Name of the component
            
        Returns:
            True if component is running, False otherwise
        """
        return self._component_states.get(component_name) == 'running'
    
    def get_all_component_states(self) -> Dict[str, str]:
        """Get all component states
        
        Returns:
            Dictionary of component names to states
        """
        return self._component_states.copy()
    
    @property
    def is_initialized(self) -> bool:
        """Check if lifecycle manager is initialized"""
        return self._initialized
    
    def __enter__(self):
        """Context manager entry"""
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._initialized:
            self.shutdown()


# Global lifecycle manager instance
_global_lifecycle_manager = LifecycleManager()


def get_global_lifecycle_manager() -> LifecycleManager:
    """Get the global lifecycle manager"""
    return _global_lifecycle_manager


def add_startup_callback(callback: Callable) -> None:
    """Add a startup callback to the global lifecycle manager"""
    _global_lifecycle_manager.add_startup_callback(callback)


def add_shutdown_callback(callback: Callable) -> None:
    """Add a shutdown callback to the global lifecycle manager"""
    _global_lifecycle_manager.add_shutdown_callback(callback) 