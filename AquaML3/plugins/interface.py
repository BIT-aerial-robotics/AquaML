"""Plugin Interface

This module defines the plugin interface for AquaML.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class PluginType(Enum):
    """Plugin type enumeration"""
    ALGORITHM = "algorithm"
    ENVIRONMENT = "environment"
    MODEL = "model"
    UTILITY = "utility"
    DATA_PROCESSOR = "data_processor"
    VISUALIZATION = "visualization"


@dataclass
class PluginInfo:
    """Plugin information"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str]
    entry_point: str
    min_aquaml_version: str = "0.1.0"
    max_aquaml_version: str = "999.999.999"


class PluginInterface(ABC):
    """Base plugin interface
    
    All plugins must implement this interface to be compatible with AquaML.
    """
    
    @property
    @abstractmethod
    def plugin_info(self) -> PluginInfo:
        """Return plugin information
        
        Returns:
            PluginInfo: Plugin metadata
        """
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin
        
        Args:
            config: Plugin configuration
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up plugin resources"""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration
        
        Args:
            config: Configuration to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return True
    
    def get_capabilities(self) -> List[str]:
        """Get plugin capabilities
        
        Returns:
            List[str]: List of capability names
        """
        return []
    
    def get_requirements(self) -> List[str]:
        """Get plugin requirements
        
        Returns:
            List[str]: List of requirement names
        """
        return []


class AlgorithmPlugin(PluginInterface):
    """Base class for algorithm plugins"""
    
    @abstractmethod
    def get_algorithm_class(self):
        """Get the algorithm class
        
        Returns:
            Algorithm class
        """
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the algorithm
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        pass


class EnvironmentPlugin(PluginInterface):
    """Base class for environment plugins"""
    
    @abstractmethod
    def get_environment_class(self):
        """Get the environment class
        
        Returns:
            Environment class
        """
        pass
    
    @abstractmethod
    def get_supported_environments(self) -> List[str]:
        """Get list of supported environments
        
        Returns:
            List[str]: Environment names
        """
        pass


class ModelPlugin(PluginInterface):
    """Base class for model plugins"""
    
    @abstractmethod
    def get_model_class(self):
        """Get the model class
        
        Returns:
            Model class
        """
        pass
    
    @abstractmethod
    def get_model_architectures(self) -> List[str]:
        """Get supported model architectures
        
        Returns:
            List[str]: Architecture names
        """
        pass


class UtilityPlugin(PluginInterface):
    """Base class for utility plugins"""
    
    @abstractmethod
    def get_utility_functions(self) -> Dict[str, callable]:
        """Get utility functions provided by this plugin
        
        Returns:
            Dict[str, callable]: Function name to function mapping
        """
        pass 