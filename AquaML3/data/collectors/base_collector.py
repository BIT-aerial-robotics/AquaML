"""Base Data Collector

This module provides the base class for all data collectors in AquaML.
"""

import abc
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from loguru import logger
from pathlib import Path
import pickle
import json
from datetime import datetime

# Safe PyTorch import
try:
    import torch
    HAS_TORCH = True
except RuntimeError:
    # Handle PyTorch docstring issue
    import importlib
    import torch
    importlib.reload(torch)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from ..core_units import UnitConfig, BaseUnit, DataUnitFactory
from ...core.coordinator import coordinator
from ...core.exceptions import AquaMLException


class BaseCollector(abc.ABC):
    """
    Base class for all data collectors.
    
    This class provides the common interface and functionality for collecting
    data from environments and agents for various learning scenarios.
    """
    
    def __init__(self, 
                 name: str,
                 save_path: Optional[Union[str, Path]] = None,
                 auto_save: bool = True,
                 max_buffer_size: int = 100000):
        """
        Initialize the base collector.
        
        Args:
            name: Name of the collector
            save_path: Path to save collected data
            auto_save: Whether to auto-save data when buffer is full
            max_buffer_size: Maximum size of the data buffer
        """
        self.name = name
        self.save_path = Path(save_path) if save_path else None
        self.auto_save = auto_save
        self.max_buffer_size = max_buffer_size
        
        # Data storage
        self.data_buffer: Dict[str, List[Any]] = {}
        self.metadata: Dict[str, Any] = {}
        
        # Collection statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.collection_start_time = None
        
        # Data unit configurations
        self.observation_cfg: Dict[str, UnitConfig] = {}
        self.action_cfg: Dict[str, UnitConfig] = {}
        self.reward_cfg: Dict[str, UnitConfig] = {}
        
        logger.info(f"Initialized collector '{name}' with buffer size {max_buffer_size}")
    
    def initialize_configs(self, env_info: Dict[str, Any]) -> None:
        """
        Initialize data unit configurations from environment info.
        
        Args:
            env_info: Environment information containing observation/action configs
        """
        # Extract observation configuration
        if 'observation_cfg' in env_info:
            for key, cfg_dict in env_info['observation_cfg'].items():
                self.observation_cfg[key] = UnitConfig(**cfg_dict)
        
        # Extract action configuration
        if 'action_cfg' in env_info:
            for key, cfg_dict in env_info['action_cfg'].items():
                self.action_cfg[key] = UnitConfig(**cfg_dict)
        
        # Extract reward configuration (if available)
        if 'reward_cfg' in env_info:
            for key, cfg_dict in env_info['reward_cfg'].items():
                self.reward_cfg[key] = UnitConfig(**cfg_dict)
        
        # Initialize data buffers
        self._initialize_buffers()
        
        logger.info(f"Initialized configs for collector '{self.name}'")
    
    def _initialize_buffers(self) -> None:
        """Initialize data buffers based on configurations."""
        # Initialize observation buffers
        for key in self.observation_cfg:
            self.data_buffer[f'observations_{key}'] = []
        
        # Initialize action buffers
        for key in self.action_cfg:
            self.data_buffer[f'actions_{key}'] = []
        
        # Initialize reward buffers
        for key in self.reward_cfg:
            self.data_buffer[f'rewards_{key}'] = []
        
        # Initialize common buffers
        self.data_buffer['dones'] = []
        self.data_buffer['truncated'] = []
        self.data_buffer['infos'] = []
        self.data_buffer['timestamps'] = []
    
    @abc.abstractmethod
    def collect_step(self, 
                    observation: Dict[str, Any],
                    action: Dict[str, Any],
                    reward: Dict[str, Any],
                    next_observation: Dict[str, Any],
                    done: bool,
                    truncated: bool,
                    info: Dict[str, Any]) -> None:
        """
        Collect data from a single step.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is done
            truncated: Whether episode is truncated
            info: Additional info
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def collect_episode(self, episode_data: Dict[str, Any]) -> None:
        """
        Collect data from a complete episode.
        
        Args:
            episode_data: Complete episode data
        """
        raise NotImplementedError
    
    def start_collection(self) -> None:
        """Start data collection session."""
        self.collection_start_time = datetime.now()
        self.total_steps = 0
        self.total_episodes = 0
        logger.info(f"Started data collection for '{self.name}'")
    
    def end_collection(self) -> None:
        """End data collection session."""
        if self.auto_save and self.save_path:
            self.save_data()
        
        collection_duration = datetime.now() - self.collection_start_time if self.collection_start_time else None
        logger.info(f"Ended data collection for '{self.name}'. "
                   f"Steps: {self.total_steps}, Episodes: {self.total_episodes}, "
                   f"Duration: {collection_duration}")
    
    def save_data(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save collected data to disk.
        
        Args:
            save_path: Path to save data (overrides default)
        """
        target_path = Path(save_path) if save_path else self.save_path
        if not target_path:
            raise ValueError("No save path specified")
        
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Save data buffer
        data_file = target_path / f"{self.name}_data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump(self.data_buffer, f)
        
        # Save metadata
        metadata_file = target_path / f"{self.name}_metadata.json"
        metadata = {
            'name': self.name,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'collection_start_time': self.collection_start_time.isoformat() if self.collection_start_time else None,
            'buffer_size': len(self.data_buffer.get('timestamps', [])),
            'observation_cfg': {k: v.to_dict() for k, v in self.observation_cfg.items()},
            'action_cfg': {k: v.to_dict() for k, v in self.action_cfg.items()},
            'reward_cfg': {k: v.to_dict() for k, v in self.reward_cfg.items()},
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved data for collector '{self.name}' to {target_path}")
    
    def load_data(self, load_path: Union[str, Path]) -> None:
        """
        Load data from disk.
        
        Args:
            load_path: Path to load data from
        """
        load_path = Path(load_path)
        
        # Load data buffer
        data_file = load_path / f"{self.name}_data.pkl"
        if data_file.exists():
            with open(data_file, 'rb') as f:
                self.data_buffer = pickle.load(f)
        
        # Load metadata
        metadata_file = load_path / f"{self.name}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.total_steps = metadata.get('total_steps', 0)
                self.total_episodes = metadata.get('total_episodes', 0)
        
        logger.info(f"Loaded data for collector '{self.name}' from {load_path}")
    
    def clear_buffer(self) -> None:
        """Clear the data buffer."""
        for key in self.data_buffer:
            self.data_buffer[key].clear()
        logger.info(f"Cleared buffer for collector '{self.name}'")
    
    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self.data_buffer.get('timestamps', []))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        buffer_size = self.get_buffer_size()
        return {
            'name': self.name,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'buffer_size': buffer_size,
            'collection_start_time': self.collection_start_time.isoformat() if self.collection_start_time else None,
            'buffer_usage': buffer_size / self.max_buffer_size if self.max_buffer_size > 0 else 0.0
        }
    
    def _should_auto_save(self) -> bool:
        """Check if auto-save should be triggered."""
        return (self.auto_save and 
                self.save_path and 
                self.get_buffer_size() >= self.max_buffer_size)
    
    def _convert_to_numpy(self, data: Any) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        else:
            return np.array([data])
    
    def _validate_data_shapes(self, data: Dict[str, Any], cfg: Dict[str, UnitConfig]) -> None:
        """Validate data shapes against configuration."""
        for key, value in data.items():
            if key in cfg:
                expected_shape = cfg[key].single_shape
                actual_shape = np.array(value).shape
                if len(actual_shape) > 0 and actual_shape[-len(expected_shape):] != expected_shape:
                    logger.warning(f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}") 