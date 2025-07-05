"""AquaML Core Framework

This module provides the core infrastructure for the AquaML machine learning library.
"""

from .registry import ComponentRegistry
from .coordinator import AquaMLCoordinator
from .lifecycle import LifecycleManager
from .exceptions import AquaMLException, PluginError, ConfigError

__all__ = [
    'ComponentRegistry',
    'AquaMLCoordinator', 
    'LifecycleManager',
    'AquaMLException',
    'PluginError',
    'ConfigError'
] 