"""
Test configuration and shared fixtures for AquaML data collectors.

This module provides common fixtures and configuration used across all tests.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a session-scoped temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="aquaml_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for individual tests."""
    temp_dir = tempfile.mkdtemp(prefix="aquaml_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def simple_env_info():
    """Basic environment configuration for testing."""
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
def complex_env_info():
    """Complex environment configuration with multiple observation types."""
    return {
        'observation_cfg': {
            'state': {
                'name': 'state',
                'dtype': 'float32',
                'single_shape': (8,),
                'size': 10000
            },
            'image': {
                'name': 'image',
                'dtype': 'float32',
                'single_shape': (64, 64, 3),
                'size': 1000
            },
            'goal': {
                'name': 'goal',
                'dtype': 'float32',
                'single_shape': (3,),
                'size': 10000
            }
        },
        'action_cfg': {
            'continuous': {
                'name': 'continuous',
                'dtype': 'float32',
                'single_shape': (4,),
                'size': 10000
            },
            'discrete': {
                'name': 'discrete',
                'dtype': 'int32',
                'single_shape': (1,),
                'size': 10000
            }
        },
        'reward_cfg': {
            'task_reward': {
                'name': 'task_reward',
                'dtype': 'float32',
                'single_shape': (1,),
                'size': 10000
            },
            'sparse_reward': {
                'name': 'sparse_reward',
                'dtype': 'float32',
                'single_shape': (1,),
                'size': 10000
            }
        }
    }


@pytest.fixture
def sample_step_data():
    """Generate sample step data for testing."""
    def _generate_step_data(obs_shape=(4,), action_shape=(1,), batch_size=1):
        return {
            'observation': {
                'state': np.random.randn(batch_size, 1, *obs_shape).astype(np.float32)
            },
            'action': {
                'action': np.random.randn(batch_size, 1, *action_shape).astype(np.float32)
            },
            'reward': {
                'reward': np.random.randn(batch_size, 1, 1).astype(np.float32)
            },
            'next_observation': {
                'state': np.random.randn(batch_size, 1, *obs_shape).astype(np.float32)
            },
            'done': False,
            'truncated': False,
            'info': {'step': 0}
        }
    return _generate_step_data


@pytest.fixture
def sample_episode_data():
    """Generate sample episode data for testing."""
    def _generate_episode_data(episode_length=10, obs_shape=(4,), action_shape=(1,)):
        return {
            'observations': [
                {'state': np.random.randn(1, 1, *obs_shape).astype(np.float32)} 
                for _ in range(episode_length)
            ],
            'actions': [
                {'action': np.random.randn(1, 1, *action_shape).astype(np.float32)} 
                for _ in range(episode_length)
            ],
            'rewards': [
                {'reward': np.random.randn(1, 1, 1).astype(np.float32)} 
                for _ in range(episode_length)
            ],
            'dones': [False] * (episode_length - 1) + [True],
            'truncated': [False] * episode_length,
            'infos': [{'step': i} for i in range(episode_length)]
        }
    return _generate_episode_data


@pytest.fixture
def mock_environment():
    """Create a mock environment for testing."""
    class MockEnv:
        def __init__(self, obs_shape=(4,), action_shape=(1,), max_steps=100):
            self.obs_shape = obs_shape
            self.action_shape = action_shape
            self.max_steps = max_steps
            self.step_count = 0
            
        def reset(self):
            self.step_count = 0
            obs = np.random.randn(*self.obs_shape).astype(np.float32)
            return obs, {}
            
        def step(self, action):
            self.step_count += 1
            obs = np.random.randn(*self.obs_shape).astype(np.float32)
            reward = np.random.randn()
            done = self.step_count >= self.max_steps
            truncated = False
            info = {'step': self.step_count}
            return obs, reward, done, truncated, info
    
    return MockEnv


@pytest.fixture
def collector_test_config():
    """Common configuration for collector tests."""
    return {
        'max_buffer_size': 1000,
        'auto_save': True,
        'save_interval': 100,
        'data_format': 'numpy'
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add slow marker to performance tests
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)


# Custom assertions for data collectors
def assert_data_consistency(data_buffer, expected_length=None):
    """Assert that data buffer has consistent lengths across all keys."""
    if not data_buffer:
        return
    
    lengths = []
    for key, value in data_buffer.items():
        if isinstance(value, list):
            lengths.append(len(value))
    
    if lengths:
        assert all(length == lengths[0] for length in lengths), \
            f"Inconsistent data lengths: {dict(zip(data_buffer.keys(), lengths))}"
        
        if expected_length is not None:
            assert lengths[0] == expected_length, \
                f"Expected length {expected_length}, got {lengths[0]}"


def assert_collector_state(collector, expected_steps=None, expected_episodes=None):
    """Assert collector state matches expectations."""
    if expected_steps is not None:
        assert collector.total_steps == expected_steps, \
            f"Expected {expected_steps} steps, got {collector.total_steps}"
    
    if expected_episodes is not None:
        assert collector.total_episodes == expected_episodes, \
            f"Expected {expected_episodes} episodes, got {collector.total_episodes}"


def assert_export_files_exist(export_path, expected_files):
    """Assert that expected export files exist."""
    export_path = Path(export_path)
    for filename in expected_files:
        file_path = export_path / filename
        assert file_path.exists(), f"Expected export file {file_path} does not exist"


# Performance test helpers
@pytest.fixture
def performance_threshold():
    """Performance thresholds for different operations."""
    return {
        'step_collection_rate': 100,  # steps per second
        'episode_collection_rate': 10,  # episodes per second
        'memory_usage_mb': 500,  # MB
        'save_time_seconds': 5,  # seconds
        'load_time_seconds': 3,  # seconds
        'export_time_seconds': 10  # seconds
    }


@pytest.fixture
def memory_monitor():
    """Fixture to monitor memory usage during tests."""
    import psutil
    import os
    
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.baseline = self.get_memory_mb()
            self.peak = self.baseline
            
        def get_memory_mb(self):
            return self.process.memory_info().rss / 1024 / 1024
            
        def update_peak(self):
            current = self.get_memory_mb()
            if current > self.peak:
                self.peak = current
                
        def get_usage_delta(self):
            return self.get_memory_mb() - self.baseline
            
        def get_peak_delta(self):
            return self.peak - self.baseline
    
    return MemoryMonitor()


# Test data generators
@pytest.fixture
def data_generator():
    """Fixture to generate test data of various types and sizes."""
    
    class DataGenerator:
        @staticmethod
        def generate_observations(shape, num_samples, dtype='float32'):
            return np.random.randn(num_samples, *shape).astype(dtype)
            
        @staticmethod
        def generate_actions(shape, num_samples, dtype='float32'):
            if dtype == 'float32':
                return np.random.randn(num_samples, *shape).astype(dtype)
            elif dtype == 'int32':
                return np.random.randint(0, 10, size=(num_samples, *shape)).astype(dtype)
                
        @staticmethod
        def generate_rewards(num_samples, reward_range=(-1, 1), dtype='float32'):
            return np.random.uniform(
                reward_range[0], reward_range[1], 
                size=(num_samples, 1)
            ).astype(dtype)
            
        @staticmethod
        def generate_episode(length, obs_shape, action_shape, reward_range=(-1, 1)):
            return {
                'observations': DataGenerator.generate_observations(obs_shape, length),
                'actions': DataGenerator.generate_actions(action_shape, length),
                'rewards': DataGenerator.generate_rewards(length, reward_range),
                'dones': [False] * (length - 1) + [True],
                'truncated': [False] * length,
                'infos': [{'step': i} for i in range(length)]
            }
    
    return DataGenerator() 