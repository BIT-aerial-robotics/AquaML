"""
AquaML - Advanced Machine Learning Framework

A modern, extensible machine learning framework with support for:
- Reinforcement Learning
- Teacher-Student Learning
- Offline Learning
- Multi-Agent Learning
- Plugin System
"""

__version__ = "2.0.0"
__author__ = "AquaML Team"

# Core components
from .core import AquaMLCoordinator, ComponentRegistry, LifecycleManager
from .plugins import PluginManager, PluginInterface
from .learning.base import BaseLearner, BaseTrainer, BaseEvaluator

# Global coordinator instance
from .core.coordinator import coordinator

# Configuration
from .config.manager import ConfigManager

__all__ = [
    'AquaMLCoordinator',
    'ComponentRegistry',
    'LifecycleManager',
    'PluginManager',
    'PluginInterface',
    'BaseLearner',
    'BaseTrainer',
    'BaseEvaluator',
    'ConfigManager',
    'coordinator'
]
