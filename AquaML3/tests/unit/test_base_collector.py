"""
Tests for BaseCollector

This module contains unit tests for the BaseCollector class,
testing core functionality, configuration management, and data validation.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
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

from AquaML.data.collectors.base_collector import BaseCollector
from AquaML.data.core_units import UnitConfig


class MockCollector(BaseCollector):
    """Mock implementation of BaseCollector for testing."""
    
    def collect_step(self, observation, action, reward, next_observation, done, truncated, info):
        """Mock implementation of collect_step."""
        pass
    
    def collect_episode(self, episode_data):
        """Mock implementation of collect_episode."""
        pass


class TestBaseCollector:
    """Test class for BaseCollector."""
    
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
                },
                'image': {
                    'name': 'image',
                    'dtype': 'float32',
                    'single_shape': (64, 64, 3),
                    'size': 1000
                }
            },
            'action_cfg': {
                'action': {
                    'name': 'action',
                    'dtype': 'float32',
                    'single_shape': (2,),
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
    def collector(self, temp_dir):
        """Create a MockCollector instance for testing."""
        return MockCollector(
            name="test_collector",
            save_path=temp_dir,
            auto_save=True,
            max_buffer_size=1000
        )
    
    def test_collector_initialization(self, temp_dir):
        """Test collector initialization."""
        collector = MockCollector(
            name="test_collector",
            save_path=temp_dir,
            auto_save=True,
            max_buffer_size=1000
        )
        
        assert collector.name == "test_collector"
        assert collector.save_path == temp_dir
        assert collector.auto_save is True
        assert collector.max_buffer_size == 1000
        assert collector.total_steps == 0
        assert collector.total_episodes == 0
        assert isinstance(collector.data_buffer, dict)
        assert isinstance(collector.metadata, dict)
    
    def test_initialize_configs(self, collector, sample_env_info):
        """Test configuration initialization."""
        collector.initialize_configs(sample_env_info)
        
        # Check observation configs
        assert 'state' in collector.observation_cfg
        assert 'image' in collector.observation_cfg
        assert isinstance(collector.observation_cfg['state'], UnitConfig)
        assert collector.observation_cfg['state'].name == 'state'
        assert collector.observation_cfg['state'].single_shape == (4,)
        
        # Check action configs
        assert 'action' in collector.action_cfg
        assert isinstance(collector.action_cfg['action'], UnitConfig)
        
        # Check reward configs
        assert 'reward' in collector.reward_cfg
        assert isinstance(collector.reward_cfg['reward'], UnitConfig)
        
        # Check data buffers are initialized
        assert 'observations_state' in collector.data_buffer
        assert 'observations_image' in collector.data_buffer
        assert 'actions_action' in collector.data_buffer
        assert 'rewards_reward' in collector.data_buffer
        assert 'dones' in collector.data_buffer
        assert 'truncated' in collector.data_buffer
        assert 'infos' in collector.data_buffer
        assert 'timestamps' in collector.data_buffer
    
    def test_start_collection(self, collector):
        """Test starting collection."""
        collector.start_collection()
        
        assert collector.collection_start_time is not None
        assert isinstance(collector.collection_start_time, datetime)
        assert collector.total_steps == 0
        assert collector.total_episodes == 0
    
    def test_end_collection(self, collector, temp_dir):
        """Test ending collection."""
        collector.start_collection()
        
        # Mock save_data to avoid actual file operations
        with patch.object(collector, 'save_data') as mock_save:
            collector.end_collection()
            mock_save.assert_called_once()
    
    def test_save_data(self, collector, sample_env_info, temp_dir):
        """Test data saving functionality."""
        collector.initialize_configs(sample_env_info)
        collector.start_collection()
        
        # Add some mock data
        collector.data_buffer['timestamps'] = [datetime.now()]
        collector.total_steps = 10
        collector.total_episodes = 2
        
        # Save data
        collector.save_data()
        
        # Check files are created
        data_file = temp_dir / f"{collector.name}_data.pkl"
        metadata_file = temp_dir / f"{collector.name}_metadata.json"
        
        assert data_file.exists()
        assert metadata_file.exists()
    
    def test_load_data(self, collector, sample_env_info, temp_dir):
        """Test data loading functionality."""
        collector.initialize_configs(sample_env_info)
        collector.start_collection()
        
        # Add and save data
        collector.data_buffer['timestamps'] = [datetime.now()]
        collector.total_steps = 10
        collector.total_episodes = 2
        collector.save_data()
        
        # Create new collector and load data
        new_collector = MockCollector(
            name="test_collector",
            save_path=temp_dir
        )
        new_collector.load_data(temp_dir)
        
        assert new_collector.total_steps == 10
        assert new_collector.total_episodes == 2
        assert 'timestamps' in new_collector.data_buffer
    
    def test_clear_buffer(self, collector, sample_env_info):
        """Test buffer clearing."""
        collector.initialize_configs(sample_env_info)
        
        # Add some data
        collector.data_buffer['observations_state'].append(np.array([1, 2, 3, 4]))
        collector.data_buffer['actions_action'].append(np.array([0.5, -0.5]))
        
        assert len(collector.data_buffer['observations_state']) == 1
        assert len(collector.data_buffer['actions_action']) == 1
        
        # Clear buffer
        collector.clear_buffer()
        
        assert len(collector.data_buffer['observations_state']) == 0
        assert len(collector.data_buffer['actions_action']) == 0
    
    def test_get_buffer_size(self, collector, sample_env_info):
        """Test buffer size calculation."""
        collector.initialize_configs(sample_env_info)
        
        assert collector.get_buffer_size() == 0
        
        # Add timestamps to simulate data
        collector.data_buffer['timestamps'].extend([datetime.now()] * 5)
        
        assert collector.get_buffer_size() == 5
    
    def test_get_statistics(self, collector):
        """Test statistics retrieval."""
        collector.start_collection()
        collector.total_steps = 100
        collector.total_episodes = 10
        
        stats = collector.get_statistics()
        
        assert stats['name'] == collector.name
        assert stats['total_steps'] == 100
        assert stats['total_episodes'] == 10
        assert stats['buffer_size'] == 0
        assert 'collection_start_time' in stats
        assert 'buffer_usage' in stats
    
    def test_convert_to_numpy(self, collector):
        """Test data conversion to numpy."""
        # Test torch tensor conversion
        torch_tensor = torch.tensor([1.0, 2.0, 3.0])
        np_result = collector._convert_to_numpy(torch_tensor)
        assert isinstance(np_result, np.ndarray)
        np.testing.assert_array_equal(np_result, [1.0, 2.0, 3.0])
        
        # Test numpy array (should remain unchanged)
        np_array = np.array([4.0, 5.0, 6.0])
        np_result = collector._convert_to_numpy(np_array)
        assert isinstance(np_result, np.ndarray)
        np.testing.assert_array_equal(np_result, [4.0, 5.0, 6.0])
        
        # Test list conversion
        list_data = [7.0, 8.0, 9.0]
        np_result = collector._convert_to_numpy(list_data)
        assert isinstance(np_result, np.ndarray)
        np.testing.assert_array_equal(np_result, [7.0, 8.0, 9.0])
        
        # Test scalar conversion
        scalar = 10.0
        np_result = collector._convert_to_numpy(scalar)
        assert isinstance(np_result, np.ndarray)
        assert np_result.shape == (1,)
        assert np_result[0] == 10.0
    
    def test_validate_data_shapes(self, collector, sample_env_info):
        """Test data shape validation."""
        collector.initialize_configs(sample_env_info)
        
        # Valid data
        valid_data = {
            'state': np.random.randn(1, 1, 4),
            'image': np.random.randn(1, 1, 64, 64, 3)
        }
        
        # This should not raise any exception
        collector._validate_data_shapes(valid_data, collector.observation_cfg)
        
        # Invalid data (wrong shape for state)
        invalid_data = {
            'state': np.random.randn(1, 1, 5),  # Should be (1, 1, 4)
        }
        
        # This should log a warning but not raise an exception
        with patch('AquaML.data.collectors.base_collector.logger') as mock_logger:
            collector._validate_data_shapes(invalid_data, collector.observation_cfg)
            mock_logger.warning.assert_called()
    
    def test_should_auto_save(self, collector):
        """Test auto-save condition checking."""
        # Initially should not auto-save (no data)
        assert not collector._should_auto_save()
        
        # Add data but not enough
        collector.data_buffer['timestamps'] = [datetime.now()] * 500
        assert not collector._should_auto_save()
        
        # Add enough data to trigger auto-save
        collector.data_buffer['timestamps'] = [datetime.now()] * 1000
        assert collector._should_auto_save()
    
    def test_collector_without_save_path(self):
        """Test collector without save path."""
        collector = MockCollector(
            name="no_save_collector",
            save_path=None,
            auto_save=False
        )
        
        assert collector.save_path is None
        assert collector.auto_save is False
        
        # Should not auto-save without save path
        collector.data_buffer['timestamps'] = [datetime.now()] * 2000
        assert not collector._should_auto_save()
    
    def test_invalid_save_path_raises_error(self, collector):
        """Test that invalid save path raises error."""
        collector.save_path = None
        
        with pytest.raises(ValueError, match="No save path specified"):
            collector.save_data()
    
    def test_collector_name_in_metadata(self, collector, sample_env_info, temp_dir):
        """Test that collector name is properly included in metadata."""
        collector.initialize_configs(sample_env_info)
        collector.start_collection()
        collector.save_data()
        
        # Load and check metadata
        import json
        metadata_file = temp_dir / f"{collector.name}_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['name'] == collector.name 