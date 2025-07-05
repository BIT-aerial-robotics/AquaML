"""
Tests for RLCollector

This module contains unit tests for the RLCollector class,
testing RL-specific functionality, async collection, and data export.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import time
import threading
from unittest.mock import Mock, patch

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

from AquaML.data.collectors.rl_collector import RLCollector
from AquaML.data.collectors.utils import TrajectoryBuffer


class TestRLCollector:
    """Test class for RLCollector."""
    
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
    def rl_collector(self, temp_dir):
        """Create an RLCollector instance for testing."""
        return RLCollector(
            name="test_rl_collector",
            save_path=temp_dir,
            max_buffer_size=100,
            collect_mode="step",
            enable_trajectory_buffer=True,
            max_trajectory_length=50
        )
    
    def test_rl_collector_initialization(self, temp_dir):
        """Test RLCollector initialization."""
        collector = RLCollector(
            name="test_rl",
            save_path=temp_dir,
            max_buffer_size=500,
            collect_mode="episode",
            enable_trajectory_buffer=True,
            max_trajectory_length=200,
            async_collection=True
        )
        
        try:
            assert collector.name == "test_rl"
            assert collector.collect_mode == "episode"
            assert collector.enable_trajectory_buffer is True
            assert collector.max_trajectory_length == 200
            assert collector.async_collection is True
            assert isinstance(collector.trajectory_buffer, TrajectoryBuffer)
            assert collector.collection_queue is not None
        finally:
            # Clean up any async threads
            if collector.async_collection:
                collector.stop_async_collection()
    
    def test_collect_step_sync(self, rl_collector, sample_env_info):
        """Test synchronous step collection."""
        rl_collector.initialize_configs(sample_env_info)
        rl_collector.start_collection()
        
        # Generate sample data
        observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
        action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
        reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
        next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
        
        # Collect step
        rl_collector.collect_step(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=False,
            truncated=False,
            info={'step': 1}
        )
        
        assert rl_collector.total_steps == 1
        assert len(rl_collector.data_buffer['observations_state']) == 1
        assert len(rl_collector.data_buffer['actions_action']) == 1
        assert len(rl_collector.data_buffer['rewards_reward']) == 1
        assert len(rl_collector.data_buffer['next_observations_state']) == 1
        assert len(rl_collector.data_buffer['dones']) == 1
        assert len(rl_collector.data_buffer['truncated']) == 1
    
    def test_collect_step_episode_completion(self, rl_collector, sample_env_info):
        """Test episode completion during step collection."""
        # Disable trajectory buffer for debugging
        rl_collector.trajectory_buffer = None
        rl_collector.enable_trajectory_buffer = False
        
        rl_collector.initialize_configs(sample_env_info)
        rl_collector.start_collection()
        
        # Collect several steps
        for step in range(5):
            observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
            reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
            next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            
            done = (step == 4)  # Last step completes episode
            
            rl_collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
                truncated=False,
                info={'step': step}
            )
        
        assert rl_collector.total_steps == 5
        assert rl_collector.total_episodes == 1
    
    def test_collect_episode(self, rl_collector, sample_env_info):
        """Test episode collection."""
        rl_collector.initialize_configs(sample_env_info)
        rl_collector.start_collection()
        
        # Create episode data
        episode_data = {
            'observations': [
                {'state': np.random.randn(1, 1, 4).astype(np.float32)} for _ in range(10)
            ],
            'actions': [
                {'action': np.random.randn(1, 1, 1).astype(np.float32)} for _ in range(10)
            ],
            'rewards': [
                {'reward': np.random.randn(1, 1, 1).astype(np.float32)} for _ in range(10)
            ],
            'dones': [False] * 9 + [True],
            'truncated': [False] * 10,
            'infos': [{'step': i} for i in range(10)]
        }
        
        rl_collector.collect_episode(episode_data)
        
        assert 'episode_lengths' in rl_collector.data_buffer
        assert 'episode_rewards' in rl_collector.data_buffer
        assert len(rl_collector.data_buffer['episode_lengths']) == 1
        assert rl_collector.data_buffer['episode_lengths'][0] == 10
    
    def test_async_collection(self, rl_collector, sample_env_info):
        """Test asynchronous collection."""
        rl_collector.async_collection = True
        rl_collector.collection_queue = Mock()
        rl_collector.initialize_configs(sample_env_info)
        rl_collector.start_collection()
        
        try:
            observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
            reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
            next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            
            rl_collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=False,
                truncated=False,
                info={'step': 1}
            )
            
            # Verify data was added to queue
            rl_collector.collection_queue.put.assert_called_once()
        finally:
            # Clean up
            rl_collector.async_collection = False
    
    def test_start_stop_async_collection(self, rl_collector):
        """Test starting and stopping async collection."""
        rl_collector.async_collection = True
        rl_collector.collection_queue = Mock()
        
        try:
            # Start async collection
            with patch('threading.Thread') as mock_thread:
                rl_collector.start_async_collection()
                mock_thread.assert_called_once()
                assert rl_collector.stop_collection is False
            
            # Stop async collection
            rl_collector.collection_thread = Mock()
            rl_collector.stop_async_collection()
            assert rl_collector.stop_collection is True
            rl_collector.collection_thread.join.assert_called_once()
        finally:
            # Clean up
            rl_collector.async_collection = False
            rl_collector.stop_collection = True
    
    def test_trajectory_buffer_functionality(self, rl_collector, sample_env_info):
        """Test trajectory buffer functionality."""
        rl_collector.initialize_configs(sample_env_info)
        rl_collector.start_collection()
        
        # Collect a complete trajectory
        for step in range(10):
            observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
            reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
            next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            
            done = (step == 9)
            
            rl_collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
                truncated=False,
                info={'step': step}
            )
        
        # Check trajectory buffer
        trajectories = rl_collector.get_trajectory_data()
        assert len(trajectories) == 1
        
        latest_trajectory = rl_collector.get_latest_trajectory()
        assert latest_trajectory is not None
        assert latest_trajectory['length'] == 10
    
    def test_export_for_offline_rl(self, rl_collector, sample_env_info, temp_dir):
        """Test exporting data for offline RL."""
        rl_collector.initialize_configs(sample_env_info)
        rl_collector.start_collection()
        
        # Collect some data
        for step in range(20):
            observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
            reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
            next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            
            rl_collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=(step % 10 == 9),
                truncated=False,
                info={'step': step}
            )
        
        # Export for offline RL
        export_path = temp_dir / "offline_rl"
        rl_collector.export_for_offline_rl(export_path)
        
        # Check export file exists
        data_file = export_path / "offline_rl_data.pkl"
        assert data_file.exists()
        
        # Load and verify data structure
        import pickle
        with open(data_file, 'rb') as f:
            offline_data = pickle.load(f)
        
        assert 'observations' in offline_data
        assert 'actions' in offline_data
        assert 'rewards' in offline_data
        assert 'next_observations' in offline_data
        assert 'terminals' in offline_data
        assert 'timeouts' in offline_data
    
    def test_export_for_teacher_student(self, rl_collector, sample_env_info, temp_dir):
        """Test exporting data for teacher-student learning."""
        rl_collector.initialize_configs(sample_env_info)
        rl_collector.start_collection()
        
        # Collect trajectory data
        for step in range(15):
            observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
            reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
            next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            
            rl_collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=(step == 14),
                truncated=False,
                info={'step': step}
            )
        
        # Export for teacher-student learning
        export_path = temp_dir / "teacher_student"
        rl_collector.export_for_teacher_student(export_path)
        
        # Check export file exists
        data_file = export_path / "teacher_student_data.pkl"
        assert data_file.exists()
        
        # Load and verify data structure
        import pickle
        with open(data_file, 'rb') as f:
            teacher_data = pickle.load(f)
        
        assert 'demonstrations' in teacher_data
        assert 'metadata' in teacher_data
        assert teacher_data['metadata']['total_trajectories'] > 0
    
    def test_get_dataset_statistics(self, rl_collector, sample_env_info):
        """Test dataset statistics retrieval."""
        rl_collector.initialize_configs(sample_env_info)
        rl_collector.start_collection()
        
        # Collect data for multiple episodes
        episode_lengths = [10, 15, 8, 12]
        for ep_len in episode_lengths:
            for step in range(ep_len):
                observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
                action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
                reward = {'reward': np.array([[[1.0]]]).astype(np.float32)}
                next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
                
                rl_collector.collect_step(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    done=(step == ep_len - 1),
                    truncated=False,
                    info={'step': step}
                )
        
        stats = rl_collector.get_dataset_statistics()
        
        assert 'total_episodes' in stats
        assert 'total_steps' in stats
        assert 'avg_episode_length' in stats
        assert 'avg_episode_reward' in stats
        assert stats['total_episodes'] == len(episode_lengths)
        assert stats['total_steps'] == sum(episode_lengths)
    
    def test_buffer_initialization(self, rl_collector, sample_env_info):
        """Test buffer initialization for RL collector."""
        rl_collector.initialize_configs(sample_env_info)
        
        # Check next observation buffers are created
        assert 'next_observations_state' in rl_collector.data_buffer
        
        # Check episode-level buffers are created
        assert 'episode_lengths' in rl_collector.data_buffer
        assert 'episode_rewards' in rl_collector.data_buffer
    
    def test_collect_mode_step(self, temp_dir, sample_env_info):
        """Test collection in step mode."""
        collector = RLCollector(
            name="step_mode_collector",
            save_path=temp_dir,
            collect_mode="step"
        )
        collector.initialize_configs(sample_env_info)
        collector.start_collection()
        
        # Collect steps
        observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
        action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
        reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
        next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
        
        collector.collect_step(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=True,
            truncated=False,
            info={'step': 1}
        )
        
        assert collector.total_steps == 1
        assert collector.total_episodes == 1
    
    def test_collect_mode_episode(self, temp_dir, sample_env_info):
        """Test collection in episode mode."""
        collector = RLCollector(
            name="episode_mode_collector",
            save_path=temp_dir,
            collect_mode="episode"
        )
        collector.initialize_configs(sample_env_info)
        collector.start_collection()
        
        # Collect episode
        observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
        action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
        reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
        next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
        
        collector.collect_step(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=True,
            truncated=False,
            info={'step': 1}
        )
        
        assert collector.total_steps == 1
        assert collector.total_episodes == 1
    
    def test_auto_save_functionality(self, temp_dir, sample_env_info):
        """Test auto-save functionality."""
        collector = RLCollector(
            name="auto_save_collector",
            save_path=temp_dir,
            max_buffer_size=5,  # Small buffer to trigger auto-save quickly
            auto_save=True
        )
        collector.initialize_configs(sample_env_info)
        collector.start_collection()
        
        # Mock save_data to track calls
        with patch.object(collector, 'save_data') as mock_save:
            # Collect enough data to trigger auto-save
            for step in range(10):
                observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
                action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
                reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
                next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
                
                collector.collect_step(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    done=False,
                    truncated=False,
                    info={'step': step}
                )
            
            # Auto-save should have been triggered
            assert mock_save.call_count > 0 