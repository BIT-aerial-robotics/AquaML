"""Collector Utilities

This module provides utility classes and functions for data collection.
"""

from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import numpy as np
from loguru import logger
from collections import deque
import threading
from pathlib import Path
import pickle
import h5py
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


class DataBuffer:
    """
    Efficient data buffer for storing collected data.
    
    This buffer provides circular buffer functionality with automatic
    type conversion and memory management.
    """
    
    def __init__(self, 
                 max_size: int = 100000,
                 data_types: Optional[Dict[str, type]] = None,
                 enable_compression: bool = False):
        """
        Initialize the data buffer.
        
        Args:
            max_size: Maximum size of the buffer
            data_types: Expected data types for each key
            enable_compression: Whether to enable data compression
        """
        self.max_size = max_size
        self.data_types = data_types or {}
        self.enable_compression = enable_compression
        
        self.data: Dict[str, deque] = {}
        self.current_size = 0
        self.total_added = 0
        self.lock = threading.Lock()
        
        logger.debug(f"Initialized DataBuffer with max_size={max_size}")
    
    def add(self, key: str, value: Any) -> None:
        """
        Add a value to the buffer.
        
        Args:
            key: Key for the data
            value: Value to add
        """
        with self.lock:
            if key not in self.data:
                self.data[key] = deque(maxlen=self.max_size)
            
            # Convert to expected type if specified
            if key in self.data_types:
                try:
                    if self.data_types[key] == np.ndarray:
                        value = np.array(value)
                    elif self.data_types[key] == torch.Tensor:
                        value = torch.tensor(value)
                except Exception as e:
                    logger.warning(f"Failed to convert {key} to {self.data_types[key]}: {e}")
            
            self.data[key].append(value)
            self.total_added += 1
            
            # Update current size
            self.current_size = len(self.data[key])
    
    def get(self, key: str, start_idx: int = 0, end_idx: Optional[int] = None) -> List[Any]:
        """
        Get data from the buffer.
        
        Args:
            key: Key for the data
            start_idx: Start index
            end_idx: End index (None for all)
            
        Returns:
            List of values
        """
        with self.lock:
            if key not in self.data:
                return []
            
            data_list = list(self.data[key])
            if end_idx is None:
                return data_list[start_idx:]
            else:
                return data_list[start_idx:end_idx]
    
    def get_all_data(self) -> Dict[str, List[Any]]:
        """Get all data from the buffer."""
        with self.lock:
            return {key: list(values) for key, values in self.data.items()}
    
    def clear(self) -> None:
        """Clear all data from the buffer."""
        with self.lock:
            self.data.clear()
            self.current_size = 0
            logger.debug("Cleared DataBuffer")
    
    def size(self) -> int:
        """Get current size of the buffer."""
        return self.current_size
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.current_size >= self.max_size


class TrajectoryBuffer:
    """
    Buffer for storing complete trajectories.
    
    This buffer manages trajectory data with automatic episode segmentation
    and trajectory completion detection.
    """
    
    def __init__(self, 
                 max_trajectory_length: int = 1000,
                 max_trajectories: int = 1000):
        """
        Initialize the trajectory buffer.
        
        Args:
            max_trajectory_length: Maximum length of a single trajectory
            max_trajectories: Maximum number of trajectories to store
        """
        self.max_trajectory_length = max_trajectory_length
        self.max_trajectories = max_trajectories
        
        self.trajectories: deque = deque(maxlen=max_trajectories)
        self.current_trajectory: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
        logger.debug(f"Initialized TrajectoryBuffer with max_length={max_trajectory_length}, "
                    f"max_trajectories={max_trajectories}")
    
    def add_step(self, step_data: Dict[str, Any]) -> None:
        """
        Add a step to the current trajectory.
        
        Args:
            step_data: Step data including observation, action, reward, etc.
        """
        with self.lock:
            self.current_trajectory.append(step_data)
            
            # Check if trajectory is complete
            if (step_data.get('done', False) or 
                step_data.get('truncated', False) or
                len(self.current_trajectory) >= self.max_trajectory_length):
                self.finalize_trajectory()
    
    def finalize_trajectory(self) -> None:
        """Finalize the current trajectory."""
        with self.lock:
            if self.current_trajectory:
                # Create trajectory dictionary
                trajectory = {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'next_observations': [],
                    'dones': [],
                    'truncated': [],
                    'infos': [],
                    'length': len(self.current_trajectory)
                }
                
                # Extract data from steps
                for step in self.current_trajectory:
                    trajectory['observations'].append(step['observation'])
                    trajectory['actions'].append(step['action'])
                    trajectory['rewards'].append(step['reward'])
                    trajectory['next_observations'].append(step.get('next_observation', {}))
                    trajectory['dones'].append(step.get('done', False))
                    trajectory['truncated'].append(step.get('truncated', False))
                    trajectory['infos'].append(step.get('info', {}))
                
                # Add to trajectories
                self.trajectories.append(trajectory)
                
                # Clear current trajectory
                self.current_trajectory.clear()
                
                logger.debug(f"Finalized trajectory with {trajectory['length']} steps")
    
    def get_trajectories(self) -> List[Dict[str, Any]]:
        """Get all completed trajectories."""
        with self.lock:
            return list(self.trajectories)
    
    def get_latest_trajectory(self) -> Optional[Dict[str, Any]]:
        """Get the latest completed trajectory."""
        with self.lock:
            return self.trajectories[-1] if self.trajectories else None
    
    def clear(self) -> None:
        """Clear all trajectories."""
        with self.lock:
            self.trajectories.clear()
            self.current_trajectory.clear()
            logger.debug("Cleared TrajectoryBuffer")
    
    def size(self) -> int:
        """Get number of completed trajectories."""
        return len(self.trajectories)


class CollectorUtils:
    """
    Utility functions for data collection.
    """
    
    @staticmethod
    def convert_data_format(data: Dict[str, Any], 
                           target_format: str = "numpy") -> Dict[str, Any]:
        """
        Convert data to target format.
        
        Args:
            data: Input data
            target_format: Target format ("numpy" or "torch")
            
        Returns:
            Converted data
        """
        converted = {}
        
        for key, value in data.items():
            if target_format == "numpy":
                if isinstance(value, torch.Tensor):
                    converted[key] = value.detach().cpu().numpy()
                elif isinstance(value, np.ndarray):
                    converted[key] = value
                else:
                    converted[key] = np.array(value)
            elif target_format == "torch":
                if isinstance(value, np.ndarray):
                    converted[key] = torch.from_numpy(value)
                elif isinstance(value, torch.Tensor):
                    converted[key] = value
                else:
                    converted[key] = torch.tensor(value)
            else:
                converted[key] = value
        
        return converted
    
    @staticmethod
    def validate_data_consistency(data: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Validate data consistency across different keys.
        
        Args:
            data: Data to validate
            
        Returns:
            Validation report
        """
        report = {
            'valid': True,
            'issues': [],
            'lengths': {},
            'dtypes': {}
        }
        
        # Check lengths
        lengths = {}
        for key, values in data.items():
            lengths[key] = len(values)
            report['lengths'][key] = lengths[key]
        
        # Check if all lengths are the same
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            report['valid'] = False
            report['issues'].append(f"Inconsistent lengths: {lengths}")
        
        # Check data types
        for key, values in data.items():
            if values:
                dtype = type(values[0])
                report['dtypes'][key] = dtype.__name__
                
                # Check if all values have the same type
                if not all(isinstance(v, dtype) for v in values):
                    report['valid'] = False
                    report['issues'].append(f"Inconsistent types in {key}")
        
        return report
    
    @staticmethod
    def save_data_hdf5(data: Dict[str, Any], 
                      filepath: Union[str, Path],
                      compression: str = "gzip") -> None:
        """
        Save data to HDF5 format.
        
        Args:
            data: Data to save
            filepath: Path to save file
            compression: Compression method
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            for key, value in data.items():
                if isinstance(value, (list, tuple)):
                    # Convert to numpy array
                    arr = np.array(value)
                    f.create_dataset(key, data=arr, compression=compression)
                elif isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression=compression)
                elif isinstance(value, torch.Tensor):
                    arr = value.detach().cpu().numpy()
                    f.create_dataset(key, data=arr, compression=compression)
                else:
                    # Store as string
                    f.attrs[key] = str(value)
        
        logger.info(f"Saved data to HDF5 file: {filepath}")
    
    @staticmethod
    def load_data_hdf5(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data from HDF5 format.
        
        Args:
            filepath: Path to load file
            
        Returns:
            Loaded data
        """
        filepath = Path(filepath)
        data = {}
        
        with h5py.File(filepath, 'r') as f:
            # Load datasets
            for key in f.keys():
                data[key] = f[key][:]
            
            # Load attributes
            for key, value in f.attrs.items():
                data[key] = value
        
        logger.info(f"Loaded data from HDF5 file: {filepath}")
        return data
    
    @staticmethod
    def compute_statistics(data: Union[np.ndarray, List[float]]) -> Dict[str, float]:
        """
        Compute basic statistics for data.
        
        Args:
            data: Input data
            
        Returns:
            Statistics dictionary
        """
        if isinstance(data, list):
            data = np.array(data)
        
        if len(data) == 0:
            return {}
        
        stats = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            'count': len(data)
        }
        
        # Add percentiles
        percentiles = [25, 75, 90, 95, 99]
        for p in percentiles:
            stats[f'p{p}'] = np.percentile(data, p)
        
        return stats
    
    @staticmethod
    def filter_data(data: Dict[str, List[Any]], 
                   filter_fn: Callable[[Dict[str, Any]], bool]) -> Dict[str, List[Any]]:
        """
        Filter data based on a filter function.
        
        Args:
            data: Input data
            filter_fn: Filter function that takes a row and returns bool
            
        Returns:
            Filtered data
        """
        if not data:
            return {}
        
        # Get the length of data
        first_key = next(iter(data.keys()))
        data_length = len(data[first_key])
        
        # Create filtered data
        filtered_data = {key: [] for key in data.keys()}
        
        for i in range(data_length):
            # Create row
            row = {key: values[i] for key, values in data.items()}
            
            # Apply filter
            if filter_fn(row):
                for key, value in row.items():
                    filtered_data[key].append(value)
        
        return filtered_data
    
    @staticmethod
    def sample_data(data: Dict[str, List[Any]], 
                   sample_size: int,
                   random_seed: Optional[int] = None) -> Dict[str, List[Any]]:
        """
        Sample data randomly.
        
        Args:
            data: Input data
            sample_size: Number of samples to take
            random_seed: Random seed for reproducibility
            
        Returns:
            Sampled data
        """
        if not data:
            return {}
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Get the length of data
        first_key = next(iter(data.keys()))
        data_length = len(data[first_key])
        
        # Sample indices
        sample_size = min(sample_size, data_length)
        indices = np.random.choice(data_length, sample_size, replace=False)
        
        # Create sampled data
        sampled_data = {}
        for key, values in data.items():
            sampled_data[key] = [values[i] for i in indices]
        
        return sampled_data
    
    @staticmethod
    def merge_datasets(datasets: List[Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
        """
        Merge multiple datasets.
        
        Args:
            datasets: List of datasets to merge
            
        Returns:
            Merged dataset
        """
        if not datasets:
            return {}
        
        # Get all keys
        all_keys = set()
        for dataset in datasets:
            all_keys.update(dataset.keys())
        
        # Merge data
        merged_data = {key: [] for key in all_keys}
        
        for dataset in datasets:
            for key in all_keys:
                if key in dataset:
                    merged_data[key].extend(dataset[key])
                else:
                    # Fill with None or empty values
                    merged_data[key].extend([None] * len(dataset[next(iter(dataset.keys()))]))
        
        return merged_data 