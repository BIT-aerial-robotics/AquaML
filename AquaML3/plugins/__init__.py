"""AquaML Plugin System

This module provides the plugin system for extending AquaML functionality.
"""

from .interface import PluginInterface, PluginInfo, PluginType
from .manager import PluginManager
from .registry import PluginRegistry
from .loader import PluginLoader

__all__ = [
    'PluginInterface',
    'PluginInfo', 
    'PluginType',
    'PluginManager',
    'PluginRegistry',
    'PluginLoader'
] 