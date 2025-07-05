"""Reinforcement Learning Data Collector

This module provides a specialized data collector for reinforcement learning,
supporting various RL algorithms and training scenarios.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from loguru import logger
from datetime import datetime
import threading
import queue
from pathlib import Path

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
from .utils import DataBuffer, TrajectoryBuffer
from ..core_units import UnitConfig


class RLCollector(BaseCollector):
    """
    Reinforcement Learning Data Collector.
    
    This collector is specifically designed for RL data collection,
    supporting online learning, offline data generation, and 
    teacher-student learning scenarios.
    """
    
    def __init__(self,
                 name: str = "rl_collector",
                 save_path: Optional[Union[str, Path]] = None,
                 auto_save: bool = True,
                 max_buffer_size: int = 100000,
                 collect_mode: str = "step",  # "step" or "episode"
                 enable_trajectory_buffer: bool = True,
                 max_trajectory_length: int = 1000,
                 async_collection: bool = False):
        """
        Initialize the RL collector.
        
        Args:
            name: Name of the collector
            save_path: Path to save collected data
            auto_save: Whether to auto-save data when buffer is full
            max_buffer_size: Maximum size of the data buffer
            collect_mode: Collection mode ("step" or "episode")
            enable_trajectory_buffer: Whether to enable trajectory buffering
            max_trajectory_length: Maximum length of trajectories
            async_collection: Whether to use asynchronous collection
        """
        super().__init__(name, save_path, auto_save, max_buffer_size)
        
        self.collect_mode = collect_mode
        self.enable_trajectory_buffer = enable_trajectory_buffer
        self.max_trajectory_length = max_trajectory_length
        self.async_collection = async_collection
        
        # Trajectory management
        self.current_trajectory = []
        self.trajectory_buffer = TrajectoryBuffer(max_trajectory_length) if enable_trajectory_buffer else None
        
        # Async collection
        self.collection_queue = queue.Queue() if async_collection else None
        self.collection_thread = None
        self.stop_collection = False
        
        # Episode management
        self.current_episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'truncated': [],
            'infos': []
        }
        
        logger.info(f"Initialized RL collector '{name}' with mode '{collect_mode}'")
    
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
        if self.async_collection:
            # Add to queue for async processing
            step_data = {
                'observation': observation,
                'action': action,
                'reward': reward,
                'next_observation': next_observation,
                'done': done,
                'truncated': truncated,
                'info': info,
                'timestamp': datetime.now()
            }
            self.collection_queue.put(step_data)
        else:
            # Process immediately
            self._process_step(observation, action, reward, next_observation, done, truncated, info)
    
    def _process_step(self,
                     observation: Dict[str, Any],
                     action: Dict[str, Any],
                     reward: Dict[str, Any],
                     next_observation: Dict[str, Any],
                     done: bool,
                     truncated: bool,
                     info: Dict[str, Any]) -> None:
        """Process a single step of data collection."""
        # Validate data shapes
        self._validate_data_shapes(observation, self.observation_cfg)
        self._validate_data_shapes(action, self.action_cfg)
        self._validate_data_shapes(reward, self.reward_cfg)
        
        # Convert to numpy arrays
        observation_np = {k: self._convert_to_numpy(v) for k, v in observation.items()}
        action_np = {k: self._convert_to_numpy(v) for k, v in action.items()}
        reward_np = {k: self._convert_to_numpy(v) for k, v in reward.items()}
        next_observation_np = {k: self._convert_to_numpy(v) for k, v in next_observation.items()}
        
        # Store in buffers
        for key, value in observation_np.items():
            self.data_buffer[f'observations_{key}'].append(value)
        
        for key, value in action_np.items():
            self.data_buffer[f'actions_{key}'].append(value)
        
        for key, value in reward_np.items():
            self.data_buffer[f'rewards_{key}'].append(value)
        
        for key, value in next_observation_np.items():
            self.data_buffer[f'next_observations_{key}'].append(value)
        
        self.data_buffer['dones'].append(done)
        self.data_buffer['truncated'].append(truncated)
        self.data_buffer['infos'].append(info)
        self.data_buffer['timestamps'].append(datetime.now())
        
        # Update trajectory buffer
        if self.trajectory_buffer:
            step_data = {
                'observation': observation_np,
                'action': action_np,
                'reward': reward_np,
                'next_observation': next_observation_np,
                'done': done,
                'truncated': truncated,
                'info': info
            }
            self.trajectory_buffer.add_step(step_data)
        
        # Add to current episode
        self.current_episode_data['observations'].append(observation_np)
        self.current_episode_data['actions'].append(action_np)
        self.current_episode_data['rewards'].append(reward_np)
        self.current_episode_data['dones'].append(done)
        self.current_episode_data['truncated'].append(truncated)
        self.current_episode_data['infos'].append(info)
        
        # Update counters
        self.total_steps += 1
        
        # Check if episode is done
        if done or truncated:
            self._finalize_episode()
        
        # Auto-save if needed
        if self._should_auto_save():
            self.save_data()
            if self.max_buffer_size > 0:
                self.clear_buffer()
    
    def _finalize_episode(self) -> None:
        """Finalize the current episode."""
        # The trajectory buffer will have already finalized the trajectory
        # in add_step() when done=True, so we don't need to call it again
        
        # Process episode data if needed
        if self.collect_mode == "episode":
            self.collect_episode(self.current_episode_data)
        
        # Clear episode data
        for key in self.current_episode_data:
            self.current_episode_data[key].clear()
        
        self.total_episodes += 1
        logger.debug(f"Finalized episode {self.total_episodes}")
    
    def collect_episode(self, episode_data: Dict[str, Any]) -> None:
        """
        Collect data from a complete episode.
        
        Args:
            episode_data: Complete episode data
        """
        # Store episode-level statistics
        episode_length = len(episode_data['observations'])
        total_reward = 0.0
        for r in episode_data['rewards']:
            if isinstance(r, dict):
                for value in r.values():
                    if isinstance(value, (int, float)):
                        total_reward += value
                    elif hasattr(value, 'item'):  # numpy scalar
                        total_reward += value.item()
                    elif hasattr(value, 'sum'):   # numpy array
                        total_reward += value.sum()
                    else:
                        total_reward += float(value)
            else:
                if hasattr(r, 'item'):  # numpy scalar
                    total_reward += r.item()
                elif hasattr(r, 'sum'):   # numpy array
                    total_reward += r.sum()
                else:
                    total_reward += float(r)
        
        # Store in episode buffer
        if 'episode_lengths' not in self.data_buffer:
            self.data_buffer['episode_lengths'] = []
        if 'episode_rewards' not in self.data_buffer:
            self.data_buffer['episode_rewards'] = []
        
        self.data_buffer['episode_lengths'].append(episode_length)
        self.data_buffer['episode_rewards'].append(total_reward)
        
        logger.debug(f"Collected episode: length={episode_length}, reward={total_reward}")
    
    def start_async_collection(self) -> None:
        """Start asynchronous data collection."""
        if not self.async_collection:
            logger.warning("Async collection is not enabled")
            return
        
        if self.collection_thread and self.collection_thread.is_alive():
            logger.warning("Async collection is already running")
            return
        
        self.stop_collection = False
        self.collection_thread = threading.Thread(target=self._async_collection_worker)
        self.collection_thread.start()
        logger.info("Started async data collection")
    
    def stop_async_collection(self) -> None:
        """Stop asynchronous data collection."""
        if not self.async_collection:
            return
        
        self.stop_collection = True
        if self.collection_thread:
            self.collection_thread.join()
        logger.info("Stopped async data collection")
    
    def _async_collection_worker(self) -> None:
        """Worker function for async data collection."""
        while not self.stop_collection:
            try:
                # Get data from queue with timeout
                step_data = self.collection_queue.get(timeout=1.0)
                
                # Process the step
                self._process_step(
                    step_data['observation'],
                    step_data['action'],
                    step_data['reward'],
                    step_data['next_observation'],
                    step_data['done'],
                    step_data['truncated'],
                    step_data['info']
                )
                
                self.collection_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in async collection worker: {e}")
    
    def get_trajectory_data(self) -> List[Dict[str, Any]]:
        """Get trajectory data from trajectory buffer."""
        if not self.trajectory_buffer:
            logger.warning("Trajectory buffer is not enabled")
            return []
        
        return self.trajectory_buffer.get_trajectories()
    
    def get_latest_trajectory(self) -> Optional[Dict[str, Any]]:
        """Get the latest completed trajectory."""
        if not self.trajectory_buffer:
            return None
        
        trajectories = self.trajectory_buffer.get_trajectories()
        return trajectories[-1] if trajectories else None
    
    def export_for_offline_rl(self, export_path: Union[str, Path]) -> None:
        """
        Export collected data for offline RL training.
        
        Args:
            export_path: Path to export data
        """
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for offline RL
        offline_data = {
            'observations': {},
            'actions': {},
            'rewards': {},
            'next_observations': {},
            'terminals': np.array(self.data_buffer['dones']),
            'timeouts': np.array(self.data_buffer['truncated']),
            'infos': self.data_buffer['infos']
        }
        
        # Organize observations
        for key in self.observation_cfg:
            obs_key = f'observations_{key}'
            if obs_key in self.data_buffer:
                offline_data['observations'][key] = np.array(self.data_buffer[obs_key])
        
        # Organize actions
        for key in self.action_cfg:
            act_key = f'actions_{key}'
            if act_key in self.data_buffer:
                offline_data['actions'][key] = np.array(self.data_buffer[act_key])
        
        # Organize rewards
        for key in self.reward_cfg:
            rew_key = f'rewards_{key}'
            if rew_key in self.data_buffer:
                offline_data['rewards'][key] = np.array(self.data_buffer[rew_key])
        
        # Organize next observations
        for key in self.observation_cfg:
            next_obs_key = f'next_observations_{key}'
            if next_obs_key in self.data_buffer:
                offline_data['next_observations'][key] = np.array(self.data_buffer[next_obs_key])
        
        # Save data
        data_file = export_path / "offline_rl_data.pkl"
        import pickle
        with open(data_file, 'wb') as f:
            pickle.dump(offline_data, f)
        
        logger.info(f"Exported offline RL data to {data_file}")
    
    def export_for_teacher_student(self, export_path: Union[str, Path]) -> None:
        """
        Export collected data for teacher-student learning.
        
        Args:
            export_path: Path to export data
        """
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Get trajectory data
        trajectories = self.get_trajectory_data()
        
        # Prepare teacher demonstrations
        teacher_data = {
            'demonstrations': trajectories,
            'metadata': {
                'total_trajectories': len(trajectories),
                'total_steps': self.total_steps,
                'collection_time': self.collection_start_time.isoformat() if self.collection_start_time else None
            }
        }
        
        # Save data
        data_file = export_path / "teacher_student_data.pkl"
        import pickle
        with open(data_file, 'wb') as f:
            pickle.dump(teacher_data, f)
        
        logger.info(f"Exported teacher-student data to {data_file}")
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        stats = self.get_statistics()
        
        # Add RL-specific statistics
        if 'episode_lengths' in self.data_buffer:
            episode_lengths = self.data_buffer['episode_lengths']
            stats['avg_episode_length'] = np.mean(episode_lengths)
            stats['std_episode_length'] = np.std(episode_lengths)
            stats['min_episode_length'] = np.min(episode_lengths)
            stats['max_episode_length'] = np.max(episode_lengths)
        
        if 'episode_rewards' in self.data_buffer:
            episode_rewards = self.data_buffer['episode_rewards']
            stats['avg_episode_reward'] = np.mean(episode_rewards)
            stats['std_episode_reward'] = np.std(episode_rewards)
            stats['min_episode_reward'] = np.min(episode_rewards)
            stats['max_episode_reward'] = np.max(episode_rewards)
        
        # Add trajectory statistics
        if self.trajectory_buffer:
            trajectories = self.trajectory_buffer.get_trajectories()
            stats['total_trajectories'] = len(trajectories)
            if trajectories:
                trajectory_lengths = [len(traj['observations']) for traj in trajectories]
                stats['avg_trajectory_length'] = np.mean(trajectory_lengths)
                stats['std_trajectory_length'] = np.std(trajectory_lengths)
        
        return stats
    
    def _initialize_buffers(self) -> None:
        """Initialize data buffers for RL collection."""
        super()._initialize_buffers()
        
        # Add next observation buffers
        for key in self.observation_cfg:
            self.data_buffer[f'next_observations_{key}'] = []
        
        # Add episode-level buffers
        self.data_buffer['episode_lengths'] = []
        self.data_buffer['episode_rewards'] = [] 