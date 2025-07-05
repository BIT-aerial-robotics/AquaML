"""
Tests for TrajectoryCollector

This module contains unit tests for the TrajectoryCollector class,
testing trajectory-specific functionality and filtering capabilities.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
from unittest.mock import Mock, patch

from AquaML.data.collectors.trajectory_collector import TrajectoryCollector


class TestTrajectoryCollector:
    """Test class for TrajectoryCollector."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_env_info(self):
        """Sample environment info for testing."""
        return {
            'observation_cfg': {
                'state': {
                    'name': 'state',
                    'dtype': 'float32',
                    'single_shape': (4,),
                    'size': 1000
                }
            },
            'action_cfg': {
                'action': {
                    'name': 'action',
                    'dtype': 'float32',
                    'single_shape': (1,),
                    'size': 1000
                }
            },
            'reward_cfg': {
                'reward': {
                    'name': 'reward',
                    'dtype': 'float32',
                    'single_shape': (1,),
                    'size': 1000
                }
            }
        }
    
    @pytest.fixture
    def trajectory_collector(self, temp_dir):
        """Create a TrajectoryCollector instance for testing."""
        return TrajectoryCollector(
            name="test_trajectory_collector",
            save_path=temp_dir,
            max_buffer_size=10,
            max_trajectory_length=20,
            min_trajectory_length=3
        )
    
    def test_trajectory_collector_initialization(self, temp_dir):
        """Test TrajectoryCollector initialization."""
        filter_criteria = {
            'min_reward': 10.0,
            'min_length': 5
        }
        
        collector = TrajectoryCollector(
            name="test_traj",
            save_path=temp_dir,
            max_buffer_size=50,
            max_trajectory_length=100,
            min_trajectory_length=5,
            enable_trajectory_filtering=True,
            trajectory_filter_criteria=filter_criteria
        )
        
        assert collector.name == "test_traj"
        assert collector.max_trajectory_length == 100
        assert collector.min_trajectory_length == 5
        assert collector.enable_trajectory_filtering is True
        assert collector.trajectory_filter_criteria == filter_criteria
        assert isinstance(collector.trajectories, list)
        assert isinstance(collector.trajectory_metadata, list)
    
    def test_collect_single_trajectory(self, trajectory_collector, sample_env_info):
        """Test collecting a single complete trajectory."""
        trajectory_collector.initialize_configs(sample_env_info)
        trajectory_collector.start_collection()
        
        # Collect trajectory steps
        trajectory_length = 5
        for step in range(trajectory_length):
            observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
            reward = {'reward': np.array([[[1.0]]]).astype(np.float32)}
            next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            
            done = (step == trajectory_length - 1)
            
            trajectory_collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
                truncated=False,
                info={'step': step}
            )
        
        assert trajectory_collector.total_steps == trajectory_length
        assert trajectory_collector.total_episodes == 1
        assert len(trajectory_collector.trajectories) == 1
        assert len(trajectory_collector.trajectory_metadata) == 1
        
        # Check trajectory data
        trajectory = trajectory_collector.trajectories[0]
        assert len(trajectory['observations']['state']) == trajectory_length
        assert len(trajectory['actions']['action']) == trajectory_length
        assert len(trajectory['rewards']['reward']) == trajectory_length
    
    def test_trajectory_filtering_by_length(self, temp_dir, sample_env_info):
        """Test trajectory filtering by minimum length."""
        collector = TrajectoryCollector(
            name="length_filter_collector",
            save_path=temp_dir,
            min_trajectory_length=5
        )
        collector.initialize_configs(sample_env_info)
        collector.start_collection()
        
        # Collect short trajectory (should be filtered out)
        for step in range(3):
            observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
            reward = {'reward': np.array([[[1.0]]]).astype(np.float32)}
            next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            
            collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=(step == 2),
                truncated=False,
                info={'step': step}
            )
        
        # Should have no trajectories due to filtering
        assert len(collector.trajectories) == 0
    
    def test_trajectory_filtering_by_reward(self, temp_dir, sample_env_info):
        """Test trajectory filtering by reward criteria."""
        filter_criteria = {'min_reward': 5.0}
        
        collector = TrajectoryCollector(
            name="reward_filter_collector",
            save_path=temp_dir,
            enable_trajectory_filtering=True,
            trajectory_filter_criteria=filter_criteria
        )
        collector.initialize_configs(sample_env_info)
        collector.start_collection()
        
        # Collect low-reward trajectory (should be filtered out)
        for step in range(5):
            observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
            reward = {'reward': np.array([[[0.5]]]).astype(np.float32)}  # Low reward
            next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            
            collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=(step == 4),
                truncated=False,
                info={'step': step}
            )
        
        # Should have no trajectories due to reward filtering
        assert len(collector.trajectories) == 0
    
    def test_get_trajectories_with_filtering(self, trajectory_collector, sample_env_info):
        """Test getting trajectories with runtime filtering."""
        trajectory_collector.initialize_configs(sample_env_info)
        trajectory_collector.start_collection()
        
        # Collect multiple trajectories with different rewards
        rewards = [10.0, 5.0, 15.0, 3.0]
        for i, total_reward in enumerate(rewards):
            traj_length = 5
            reward_per_step = total_reward / traj_length
            
            for step in range(traj_length):
                observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
                action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
                reward = {'reward': np.array([[[reward_per_step]]]).astype(np.float32)}
                next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
                
                trajectory_collector.collect_step(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    done=(step == traj_length - 1),
                    truncated=False,
                    info={'step': step, 'trajectory': i}
                )
        
        # Get all trajectories
        all_trajectories = trajectory_collector.get_trajectories()
        assert len(all_trajectories) == 4
        
        # Get trajectories with minimum reward filter
        high_reward_trajectories = trajectory_collector.get_trajectories(
            filter_criteria={'min_reward': 8.0}
        )
        assert len(high_reward_trajectories) == 2  # 10.0 and 15.0 reward trajectories
    
    def test_trajectory_statistics(self, trajectory_collector, sample_env_info):
        """Test trajectory statistics calculation."""
        trajectory_collector.initialize_configs(sample_env_info)
        trajectory_collector.start_collection()
        
        # Collect trajectories
        trajectory_lengths = [5, 8, 6]
        for length in trajectory_lengths:
            for step in range(length):
                observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
                action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
                reward = {'reward': np.array([[[1.0]]]).astype(np.float32)}
                next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
                
                trajectory_collector.collect_step(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    done=(step == length - 1),
                    truncated=False,
                    info={'step': step}
                )
        
        stats = trajectory_collector.get_trajectory_statistics()
        
        assert stats['total_trajectories'] == 3
        assert stats['total_steps'] == sum(trajectory_lengths)
        assert 'trajectory_lengths' in stats
        assert 'trajectory_rewards' in stats
        assert 'completion_types' in stats
    
    def test_export_trajectories(self, trajectory_collector, sample_env_info, temp_dir):
        """Test exporting trajectories to file."""
        trajectory_collector.initialize_configs(sample_env_info)
        trajectory_collector.start_collection()
        
        # Collect a trajectory
        for step in range(5):
            observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
            reward = {'reward': np.array([[[1.0]]]).astype(np.float32)}
            next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            
            trajectory_collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=(step == 4),
                truncated=False,
                info={'step': step}
            )
        
        # Export trajectories
        export_file = temp_dir / "trajectories.pkl"
        trajectory_collector.export_trajectories(
            export_file,
            format="pickle",
            include_metadata=True
        )
        
        assert export_file.exists()
        
        # Load and verify exported data
        import pickle
        with open(export_file, 'rb') as f:
            exported_data = pickle.load(f)
        
        assert 'trajectories' in exported_data
        assert 'statistics' in exported_data
        assert 'metadata' in exported_data
        assert len(exported_data['trajectories']) == 1
    
    def test_collect_episode_method(self, trajectory_collector, sample_env_info):
        """Test collecting complete episode data."""
        trajectory_collector.initialize_configs(sample_env_info)
        trajectory_collector.start_collection()
        
        # Create episode data
        episode_data = {
            'observations': [
                {'state': np.random.randn(1, 1, 4).astype(np.float32)} for _ in range(8)
            ],
            'actions': [
                {'action': np.random.randn(1, 1, 1).astype(np.float32)} for _ in range(8)
            ],
            'rewards': [
                {'reward': np.array([[[2.0]]]).astype(np.float32)} for _ in range(8)
            ],
            'dones': [False] * 7 + [True],
            'truncated': [False] * 8,
            'infos': [{'step': i} for i in range(8)]
        }
        
        trajectory_collector.collect_episode(episode_data)
        
        assert len(trajectory_collector.trajectories) == 1
        assert trajectory_collector.total_episodes == 1
        assert trajectory_collector.total_steps == 8
    
    def test_buffer_trimming(self, temp_dir, sample_env_info):
        """Test buffer trimming when max size is exceeded."""
        collector = TrajectoryCollector(
            name="trim_test_collector",
            save_path=temp_dir,
            max_buffer_size=2  # Small buffer size
        )
        collector.initialize_configs(sample_env_info)
        collector.start_collection()
        
        # Collect 3 trajectories (exceeds buffer size)
        for traj in range(3):
            for step in range(5):
                observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
                action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
                reward = {'reward': np.array([[[1.0]]]).astype(np.float32)}
                next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
                
                collector.collect_step(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    done=(step == 4),
                    truncated=False,
                    info={'step': step, 'trajectory': traj}
                )
        
        # Should only keep the last 2 trajectories
        assert len(collector.trajectories) == 2
        assert len(collector.trajectory_metadata) == 2 