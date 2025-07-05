"""Configuration Manager

This module provides configuration management for AquaML.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import json
from loguru import logger

try:
    from ..core.exceptions import ConfigError
except ImportError:
    # Fallback for when used as standalone module
    class ConfigError(Exception):
        pass


class ConfigManager:
    """Configuration manager for AquaML"""
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = {}
        self._validators: Dict[str, callable] = {}
    
    def load_config(self, config: Dict[str, Any]) -> None:
        """Load configuration from dictionary"""
        self._config.update(config)
        self._validate_config()
    
    def load_from_file(self, file_path: str) -> None:
        """Load configuration from file"""
        path = Path(file_path)
        
        if not path.exists():
            raise ConfigError(f"Config file not found: {file_path}")
        
        try:
            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    config = json.load(f)
            else:
                raise ConfigError(f"Unsupported config file format: {path.suffix}")
            
            self.load_config(config)
            logger.info(f"Loaded config from {file_path}")
            
        except Exception as e:
            raise ConfigError(f"Error loading config from {file_path}: {e}")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration"""
        for key, validator in self._validators.items():
            value = self.get_config(key)
            if value is not None and not validator(value):
                raise ConfigError(f"Invalid configuration for {key}")
    
    def register_validator(self, key: str, validator: callable) -> None:
        """Register configuration validator"""
        self._validators[key] = validator
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self._config.copy()
