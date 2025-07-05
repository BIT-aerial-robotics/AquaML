"""Trajectory Data Collector

This module provides a trajectory-focused data collector for collecting
complete episode trajectories with advanced trajectory management features.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from loguru import logger
from pathlib import Path
from datetime import datetime
import json
import pickle

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

from .base_collector import BaseCollector
from .utils import TrajectoryBuffer, CollectorUtils
from ..core_units import UnitConfig


class TrajectoryCollector(BaseCollector):
    """
    Trajectory Data Collector.
    
    This collector specializes in collecting complete trajectories/episodes
    with advanced features for trajectory analysis and management.
    """
    
    def __init__(self,
                 name: str = "trajectory_collector",
                 save_path: Optional[Union[str, Path]] = None,
                 auto_save: bool = True,
                 max_buffer_size: int = 10000,  # Number of trajectories
                 max_trajectory_length: int = 1000,
                 min_trajectory_length: int = 1,
                 enable_trajectory_filtering: bool = True,
                 trajectory_filter_criteria: Optional[Dict[str, Any]] = None):
        """
        Initialize the trajectory collector.
        
        Args:
            name: Name of the collector
            save_path: Path to save collected data
            auto_save: Whether to auto-save data when buffer is full
            max_buffer_size: Maximum number of trajectories to store
            max_trajectory_length: Maximum length of a single trajectory
            min_trajectory_length: Minimum length of a trajectory to be stored
            enable_trajectory_filtering: Whether to enable trajectory filtering
            trajectory_filter_criteria: Criteria for filtering trajectories
        """
        super().__init__(name, save_path, auto_save, max_buffer_size)
        
        self.max_trajectory_length = max_trajectory_length
        self.min_trajectory_length = min_trajectory_length
        self.enable_trajectory_filtering = enable_trajectory_filtering
        self.trajectory_filter_criteria = trajectory_filter_criteria or {}
        
        # Trajectory storage
        self.trajectories: List[Dict[str, Any]] = []
        self.current_trajectory: Dict[str, List[Any]] = {}
        self.trajectory_metadata: List[Dict[str, Any]] = []
        
        # Trajectory statistics
        self.trajectory_stats = {
            'lengths': [],
            'rewards': [],
            'completion_types': [],  # 'done', 'truncated', 'max_length'
            'start_times': [],
            'durations': []
        }
        
        self.current_trajectory_start_time = None
        
        logger.info(f"Initialized trajectory collector '{name}' with max_length={max_trajectory_length}")
    
    def collect_step(self,
                    observation: Dict[str, Any],
                    action: Dict[str, Any],
                    reward: Dict[str, Any],
                    next_observation: Dict[str, Any],
                    done: bool,
                    truncated: bool,
                    info: Dict[str, Any]) -> None:
        """
        Collect data from a single step and add to current trajectory.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is done
            truncated: Whether episode is truncated
            info: Additional info
        """
        # Initialize current trajectory if needed
        if not self.current_trajectory:
            self._start_new_trajectory()
        
        # Validate data shapes
        self._validate_data_shapes(observation, self.observation_cfg)
        self._validate_data_shapes(action, self.action_cfg)
        self._validate_data_shapes(reward, self.reward_cfg)
        
        # Convert to numpy arrays
        observation_np = {k: self._convert_to_numpy(v) for k, v in observation.items()}
        action_np = {k: self._convert_to_numpy(v) for k, v in action.items()}
        reward_np = {k: self._convert_to_numpy(v) for k, v in reward.items()}
        next_observation_np = {k: self._convert_to_numpy(v) for k, v in next_observation.items()}
        
        # Add to current trajectory
        if 'observations' not in self.current_trajectory:
            self.current_trajectory['observations'] = {k: [] for k in observation_np.keys()}
        if 'actions' not in self.current_trajectory:
            self.current_trajectory['actions'] = {k: [] for k in action_np.keys()}
        if 'rewards' not in self.current_trajectory:
            self.current_trajectory['rewards'] = {k: [] for k in reward_np.keys()}
        if 'next_observations' not in self.current_trajectory:
            self.current_trajectory['next_observations'] = {k: [] for k in next_observation_np.keys()}
        
        # Initialize other trajectory components
        for key in ['dones', 'truncated', 'infos', 'timestamps']:
            if key not in self.current_trajectory:
                self.current_trajectory[key] = []
        
        # Store step data
        for key, value in observation_np.items():
            self.current_trajectory['observations'][key].append(value)
        
        for key, value in action_np.items():
            self.current_trajectory['actions'][key].append(value)
        
        for key, value in reward_np.items():
            self.current_trajectory['rewards'][key].append(value)
        
        for key, value in next_observation_np.items():
            self.current_trajectory['next_observations'][key].append(value)
        
        self.current_trajectory['dones'].append(done)
        self.current_trajectory['truncated'].append(truncated)
        self.current_trajectory['infos'].append(info)
        self.current_trajectory['timestamps'].append(datetime.now())
        
        # Update step counter
        self.total_steps += 1
        
        # Check if trajectory should be finalized
        trajectory_length = len(self.current_trajectory['dones'])
        should_finalize = (
            done or 
            truncated or 
            trajectory_length >= self.max_trajectory_length
        )
        
        if should_finalize:
            completion_type = 'done' if done else ('truncated' if truncated else 'max_length')
            self._finalize_trajectory(completion_type)
    
    def _start_new_trajectory(self) -> None:
        """Start a new trajectory."""
        self.current_trajectory = {}
        self.current_trajectory_start_time = datetime.now()
        logger.debug("Started new trajectory")
    
    def _finalize_trajectory(self, completion_type: str) -> None:
        """
        Finalize the current trajectory.
        
        Args:
            completion_type: How the trajectory was completed ('done', 'truncated', 'max_length')
        """
        if not self.current_trajectory:
            return
        
        trajectory_length = len(self.current_trajectory['dones'])
        
        # Check minimum length requirement
        if trajectory_length < self.min_trajectory_length:
            logger.debug(f"Trajectory too short ({trajectory_length} < {self.min_trajectory_length}), skipping")
            self.current_trajectory = {}
            return
        
        # Calculate trajectory statistics
        total_reward = self._calculate_trajectory_reward()
        duration = datetime.now() - self.current_trajectory_start_time if self.current_trajectory_start_time else None
        
        # Create trajectory metadata
        metadata = {
            'length': trajectory_length,
            'total_reward': total_reward,
            'completion_type': completion_type,
            'start_time': self.current_trajectory_start_time.isoformat() if self.current_trajectory_start_time else None,
            'duration': duration.total_seconds() if duration else None,
            'trajectory_id': len(self.trajectories)
        }
        
        # Apply trajectory filtering if enabled
        if self.enable_trajectory_filtering and not self._should_keep_trajectory(metadata):
            logger.debug(f"Trajectory filtered out: {metadata}")
            self.current_trajectory = {}
            return
        
        # Store trajectory
        trajectory_copy = self._copy_trajectory(self.current_trajectory)
        self.trajectories.append(trajectory_copy)
        self.trajectory_metadata.append(metadata)
        
        # Update statistics
        self.trajectory_stats['lengths'].append(trajectory_length)
        self.trajectory_stats['rewards'].append(total_reward)
        self.trajectory_stats['completion_types'].append(completion_type)
        self.trajectory_stats['start_times'].append(self.current_trajectory_start_time)
        self.trajectory_stats['durations'].append(duration.total_seconds() if duration else 0)
        
        # Update counters
        self.total_episodes += 1
        
        # Clear current trajectory
        self.current_trajectory = {}
        self.current_trajectory_start_time = None
        
        logger.debug(f"Finalized trajectory {len(self.trajectories)}: length={trajectory_length}, "
                    f"reward={total_reward}, completion={completion_type}")
        
        # Auto-save if needed
        if self._should_auto_save():
            self.save_data()
            if self.max_buffer_size > 0:
                self._trim_buffer()
    
    def _calculate_trajectory_reward(self) -> float:
        """Calculate total reward for current trajectory."""
        if 'rewards' not in self.current_trajectory:
            return 0.0
        
        total_reward = 0.0
        for reward_key, reward_list in self.current_trajectory['rewards'].items():
            for reward_value in reward_list:
                if isinstance(reward_value, (int, float)):
                    total_reward += reward_value
                elif isinstance(reward_value, np.ndarray):
                    total_reward += np.sum(reward_value)
        
        return total_reward
    
    def _should_keep_trajectory(self, metadata: Dict[str, Any]) -> bool:
        """
        Check if trajectory should be kept based on filtering criteria.
        
        Args:
            metadata: Trajectory metadata
            
        Returns:
            Whether to keep the trajectory
        """
        if not self.trajectory_filter_criteria:
            return True
        
        # Check minimum reward
        if 'min_reward' in self.trajectory_filter_criteria:
            if metadata['total_reward'] < self.trajectory_filter_criteria['min_reward']:
                return False
        
        # Check maximum reward
        if 'max_reward' in self.trajectory_filter_criteria:
            if metadata['total_reward'] > self.trajectory_filter_criteria['max_reward']:
                return False
        
        # Check completion type
        if 'allowed_completion_types' in self.trajectory_filter_criteria:
            if metadata['completion_type'] not in self.trajectory_filter_criteria['allowed_completion_types']:
                return False
        
        # Check minimum length (additional check)
        if 'min_length' in self.trajectory_filter_criteria:
            if metadata['length'] < self.trajectory_filter_criteria['min_length']:
                return False
        
        return True
    
    def _copy_trajectory(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of trajectory data."""
        copied = {}
        
        for key, value in trajectory.items():
            if isinstance(value, dict):
                copied[key] = {k: np.array(v) if isinstance(v, list) else v for k, v in value.items()}
            elif isinstance(value, list):
                copied[key] = value.copy()
            else:
                copied[key] = value
        
        return copied
    
    def _trim_buffer(self) -> None:
        """Trim buffer to maximum size."""
        if len(self.trajectories) > self.max_buffer_size:
            # Remove oldest trajectories
            num_to_remove = len(self.trajectories) - self.max_buffer_size
            self.trajectories = self.trajectories[num_to_remove:]
            self.trajectory_metadata = self.trajectory_metadata[num_to_remove:]
            
            # Update statistics
            for key in self.trajectory_stats:
                if isinstance(self.trajectory_stats[key], list):
                    self.trajectory_stats[key] = self.trajectory_stats[key][num_to_remove:]
            
            logger.debug(f"Trimmed buffer, removed {num_to_remove} trajectories")
    
    def collect_episode(self, episode_data: Dict[str, Any]) -> None:
        """
        Collect data from a complete episode.
        
        Args:
            episode_data: Complete episode data
        """
        # This method is used when receiving complete episode data
        # Convert episode data to trajectory format
        if 'observations' in episode_data and episode_data['observations']:
            trajectory_length = len(episode_data['observations'])
            
            # Check minimum length
            if trajectory_length < self.min_trajectory_length:
                logger.debug(f"Episode too short ({trajectory_length} < {self.min_trajectory_length}), skipping")
                return
            
            # Calculate total reward
            total_reward = 0.0
            if 'rewards' in episode_data:
                for reward in episode_data['rewards']:
                    if isinstance(reward, dict):
                        total_reward += sum(reward.values())
                    else:
                        total_reward += reward
            
            # Create metadata
            metadata = {
                'length': trajectory_length,
                'total_reward': total_reward,
                'completion_type': 'episode',
                'start_time': datetime.now().isoformat(),
                'duration': None,
                'trajectory_id': len(self.trajectories)
            }
            
            # Apply filtering
            if self.enable_trajectory_filtering and not self._should_keep_trajectory(metadata):
                logger.debug(f"Episode filtered out: {metadata}")
                return
            
            # Store trajectory
            self.trajectories.append(episode_data.copy())
            self.trajectory_metadata.append(metadata)
            
            # Update statistics
            self.trajectory_stats['lengths'].append(trajectory_length)
            self.trajectory_stats['rewards'].append(total_reward)
            self.trajectory_stats['completion_types'].append('episode')
            self.trajectory_stats['start_times'].append(datetime.now())
            self.trajectory_stats['durations'].append(None)
            
            self.total_episodes += 1
            self.total_steps += trajectory_length
            
            logger.debug(f"Collected episode: length={trajectory_length}, reward={total_reward}")
    
    def get_trajectories(self, 
                        filter_criteria: Optional[Dict[str, Any]] = None,
                        sort_by: Optional[str] = None,
                        reverse: bool = False) -> List[Dict[str, Any]]:
        """
        Get trajectories with optional filtering and sorting.
        
        Args:
            filter_criteria: Criteria for filtering trajectories
            sort_by: Key to sort by ('length', 'reward', 'completion_type')
            reverse: Whether to reverse sort order
            
        Returns:
            List of trajectories
        """
        trajectories = self.trajectories.copy()
        metadata = self.trajectory_metadata.copy()
        
        # Apply filtering
        if filter_criteria:
            filtered_trajectories = []
            filtered_metadata = []
            
            for traj, meta in zip(trajectories, metadata):
                if self._matches_filter_criteria(meta, filter_criteria):
                    filtered_trajectories.append(traj)
                    filtered_metadata.append(meta)
            
            trajectories = filtered_trajectories
            metadata = filtered_metadata
        
        # Apply sorting
        if sort_by and sort_by in ['length', 'total_reward', 'completion_type']:
            combined = list(zip(trajectories, metadata))
            combined.sort(key=lambda x: x[1][sort_by], reverse=reverse)
            trajectories = [traj for traj, _ in combined]
        
        return trajectories
    
    def _matches_filter_criteria(self, metadata: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if trajectory metadata matches filter criteria."""
        for key, value in criteria.items():
            if key == 'min_reward' and metadata['total_reward'] < value:
                return False
            elif key == 'max_reward' and metadata['total_reward'] > value:
                return False
            elif key == 'min_length' and metadata['length'] < value:
                return False
            elif key == 'max_length' and metadata['length'] > value:
                return False
            elif key == 'completion_type' and metadata['completion_type'] != value:
                return False
        
        return True
    
    def get_trajectory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trajectory statistics."""
        if not self.trajectory_stats['lengths']:
            return {'message': 'No trajectories collected yet'}
        
        stats = {
            'total_trajectories': len(self.trajectories),
            'total_steps': sum(self.trajectory_stats['lengths']),
            'trajectory_lengths': CollectorUtils.compute_statistics(self.trajectory_stats['lengths']),
            'trajectory_rewards': CollectorUtils.compute_statistics(self.trajectory_stats['rewards']),
            'completion_types': {}
        }
        
        # Completion type distribution
        for completion_type in self.trajectory_stats['completion_types']:
            stats['completion_types'][completion_type] = stats['completion_types'].get(completion_type, 0) + 1
        
        # Duration statistics (if available)
        durations = [d for d in self.trajectory_stats['durations'] if d is not None]
        if durations:
            stats['trajectory_durations'] = CollectorUtils.compute_statistics(durations)
        
        return stats
    
    def export_trajectories(self, 
                           export_path: Union[str, Path],
                           format: str = "pickle",
                           include_metadata: bool = True) -> None:
        """
        Export trajectories to file.
        
        Args:
            export_path: Path to export file
            format: Export format ("pickle", "json", "hdf5")
            include_metadata: Whether to include metadata
        """
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            'trajectories': self.trajectories,
            'statistics': self.get_trajectory_statistics()
        }
        
        if include_metadata:
            export_data['metadata'] = self.trajectory_metadata
        
        if format == "pickle":
            with open(export_path, 'wb') as f:
                pickle.dump(export_data, f)
        elif format == "json":
            # Convert numpy arrays to lists for JSON serialization
            json_data = self._convert_for_json(export_data)
            with open(export_path, 'w') as f:
                json.dump(json_data, f, indent=2)
        elif format == "hdf5":
            CollectorUtils.save_data_hdf5(export_data, export_path)
        
        logger.info(f"Exported {len(self.trajectories)} trajectories to {export_path}")
    
    def _convert_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert data for JSON serialization."""
        if isinstance(data, dict):
            return {k: self._convert_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.float64, np.float32)):
            return float(data)
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data
    
    def get_buffer_size(self) -> int:
        """Get current buffer size (number of trajectories)."""
        return len(self.trajectories)
    
    def clear_buffer(self) -> None:
        """Clear all trajectories."""
        self.trajectories.clear()
        self.trajectory_metadata.clear()
        self.current_trajectory = {}
        
        # Clear statistics
        for key in self.trajectory_stats:
            if isinstance(self.trajectory_stats[key], list):
                self.trajectory_stats[key].clear()
        
        logger.info(f"Cleared trajectory buffer for collector '{self.name}'") 