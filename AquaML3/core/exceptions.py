"""AquaML Exception Classes

This module defines custom exception classes for the AquaML framework.
"""


class AquaMLException(Exception):
    """Base exception class for AquaML framework"""
    pass


class PluginError(AquaMLException):
    """Exception raised for plugin-related errors"""
    pass


class ConfigError(AquaMLException):
    """Exception raised for configuration-related errors"""
    pass


class RegistryError(AquaMLException):
    """Exception raised for component registry errors"""
    pass


class LifecycleError(AquaMLException):
    """Exception raised for lifecycle management errors"""
    pass


class EnvironmentError(AquaMLException):
    """Exception raised for environment-related errors"""
    pass


class LearningError(AquaMLException):
    """Exception raised for learning algorithm errors"""
    pass 