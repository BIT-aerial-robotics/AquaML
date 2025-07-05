"""
Integration Tests for Data Collectors

This module contains integration tests for data collectors,
testing their interaction with environment wrappers and real usage scenarios.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
import gymnasium as gym
from unittest.mock import Mock, patch

from AquaML.data.collectors.rl_collector import RLCollector
from AquaML.data.collectors.trajectory_collector import TrajectoryCollector
from AquaML.data.collectors.buffer_collector import BufferCollector


class MockEnvironment:
    """Mock environment for testing."""
    
    def __init__(self, obs_space_shape=(4,), action_space_shape=(2,)):
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_space_shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=action_space_shape, dtype=np.float32
        )
        self.step_count = 0
        self.max_episode_steps = 10
    
    def reset(self):
        self.step_count = 0
        obs = self.observation_space.sample()
        return obs, {}
    
    def step(self, action):
        self.step_count += 1
        obs = self.observation_space.sample()
        reward = np.random.random()
        done = self.step_count >= self.max_episode_steps
        truncated = False
        info = {'step': self.step_count}
        return obs, reward, done, truncated, info


class TestCollectorIntegration:
    """Integration tests for data collectors."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_env(self):
        """Create a mock environment."""
        return MockEnvironment()
    
    @pytest.fixture
    def env_info(self):
        """Create environment info for collectors."""
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
    
    def test_rl_collector_with_environment(self, temp_dir, mock_env, env_info):
        """Test RLCollector integration with environment."""
        collector = RLCollector(
            name="integration_rl_collector",
            save_path=temp_dir,
            max_buffer_size=50,
            collect_mode="step"
        )
        collector.initialize_configs(env_info)
        collector.start_collection()
        
        # Run environment episodes
        for episode in range(3):
            obs, info = mock_env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Random action
                action = mock_env.action_space.sample()
                next_obs, reward, done, truncated, step_info = mock_env.step(action)
                
                # Collect step data
                collector.collect_step(
                    observation={'state': obs.reshape(1, 1, -1)},
                    action={'action': action.reshape(1, 1, -1)},
                    reward={'reward': np.array([[[reward]]]).astype(np.float32)},
                    next_observation={'state': next_obs.reshape(1, 1, -1)},
                    done=done,
                    truncated=truncated,
                    info=step_info
                )
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
                
                if done or truncated:
                    break
        
        # Verify collection
        assert collector.total_episodes == 3
        assert collector.total_steps == 3 * mock_env.max_episode_steps
        
        # Export data
        export_path = temp_dir / "offline_rl_export"
        collector.export_for_offline_rl(export_path)
        
        # Verify export
        data_file = export_path / "offline_rl_data.pkl"
        assert data_file.exists()
        
        # Load and verify exported data
        import pickle
        with open(data_file, 'rb') as f:
            exported_data = pickle.load(f)
        
        assert 'observations' in exported_data
        assert 'actions' in exported_data
        assert 'rewards' in exported_data
        assert len(exported_data['observations']) == collector.total_steps
    
    def test_trajectory_collector_with_environment(self, temp_dir, mock_env, env_info):
        """Test TrajectoryCollector integration with environment."""
        collector = TrajectoryCollector(
            name="integration_trajectory_collector",
            save_path=temp_dir,
            max_buffer_size=5,
            min_trajectory_length=5,
            enable_trajectory_filtering=True,
            trajectory_filter_criteria={'min_reward': 3.0}
        )
        collector.initialize_configs(env_info)
        collector.start_collection()
        
        # Run environment episodes
        for episode in range(4):
            obs, info = mock_env.reset()
            
            while True:
                action = mock_env.action_space.sample()
                next_obs, reward, done, truncated, step_info = mock_env.step(action)
                
                collector.collect_step(
                    observation={'state': obs.reshape(1, 1, -1)},
                    action={'action': action.reshape(1, 1, -1)},
                    reward={'reward': np.array([[[reward]]]).astype(np.float32)},
                    next_observation={'state': next_obs.reshape(1, 1, -1)},
                    done=done,
                    truncated=truncated,
                    info=step_info
                )
                
                obs = next_obs
                if done or truncated:
                    break
        
        # Verify trajectory collection
        assert collector.total_episodes == 4
        trajectories = collector.get_trajectories()
        
        # Should have collected some trajectories
        assert len(trajectories) > 0
        
        # Export trajectories
        export_file = temp_dir / "trajectories_export.pkl"
        collector.export_trajectories(export_file, format="pickle")
        
        assert export_file.exists()
    
    def test_buffer_collector_with_memory_management(self, temp_dir, mock_env, env_info):
        """Test BufferCollector with memory management during environment interaction."""
        collector = BufferCollector(
            name="integration_buffer_collector",
            save_path=temp_dir,
            max_memory_mb=10,  # Small limit to trigger flushes
            auto_flush=True,
            memory_check_interval=5
        )
        collector.initialize_configs(env_info)
        collector.start_collection()
        
        # Mock memory usage to trigger flushes
        with patch.object(collector, 'get_memory_usage') as mock_usage:
            mock_usage.return_value = {
                'current_mb': 12,
                'max_mb': 10,
                'usage_percentage': 120
            }
            
            # Run environment episodes
            for episode in range(3):
                obs, info = mock_env.reset()
                
                while True:
                    action = mock_env.action_space.sample()
                    next_obs, reward, done, truncated, step_info = mock_env.step(action)
                    
                    collector.collect_step(
                        observation={'state': obs.reshape(1, 1, -1)},
                        action={'action': action.reshape(1, 1, -1)},
                        reward={'reward': np.array([[[reward]]]).astype(np.float32)},
                        next_observation={'state': next_obs.reshape(1, 1, -1)},
                        done=done,
                        truncated=truncated,
                        info=step_info
                    )
                    
                    obs = next_obs
                    if done or truncated:
                        break
        
        # Should have triggered some flushes
        assert collector.flush_counter > 0
        
        # Check flush files exist
        flush_files = list(temp_dir.glob(f"{collector.name}_flush_*.pkl"))
        assert len(flush_files) > 0
    
    def test_multi_collector_workflow(self, temp_dir, mock_env, env_info):
        """Test using multiple collectors in the same workflow."""
        rl_collector = RLCollector(
            name="multi_rl_collector",
            save_path=temp_dir,
            collect_mode="step"
        )
        
        trajectory_collector = TrajectoryCollector(
            name="multi_trajectory_collector",
            save_path=temp_dir,
            min_trajectory_length=3
        )
        
        collectors = [rl_collector, trajectory_collector]
        
        # Initialize all collectors
        for collector in collectors:
            collector.initialize_configs(env_info)
            collector.start_collection()
        
        # Run environment episodes
        for episode in range(2):
            obs, info = mock_env.reset()
            
            while True:
                action = mock_env.action_space.sample()
                next_obs, reward, done, truncated, step_info = mock_env.step(action)
                
                # Collect data with all collectors
                for collector in collectors:
                    collector.collect_step(
                        observation={'state': obs.reshape(1, 1, -1)},
                        action={'action': action.reshape(1, 1, -1)},
                        reward={'reward': np.array([[[reward]]]).astype(np.float32)},
                        next_observation={'state': next_obs.reshape(1, 1, -1)},
                        done=done,
                        truncated=truncated,
                        info=step_info
                    )
                
                obs = next_obs
                if done or truncated:
                    break
        
        # Verify all collectors collected data
        for collector in collectors:
            assert collector.total_episodes == 2
            assert collector.total_steps == 2 * mock_env.max_episode_steps
        
        # End collection for all collectors
        for collector in collectors:
            collector.end_collection()
        
        # Verify data files were created
        rl_data_file = temp_dir / f"{rl_collector.name}_data.pkl"
        trajectory_data_file = temp_dir / f"{trajectory_collector.name}_data.pkl"
        
        assert rl_data_file.exists()
        assert trajectory_data_file.exists()
    
    def test_collector_with_dict_observations(self, temp_dir, env_info):
        """Test collector with complex dictionary observations."""
        # Extended environment info with multiple observation types
        complex_env_info = {
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
                    'single_shape': (32, 32, 3),
                    'size': 100
                },
                'goal': {
                    'name': 'goal',
                    'dtype': 'float32',
                    'single_shape': (2,),
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
        
        collector = RLCollector(
            name="complex_obs_collector",
            save_path=temp_dir,
            collect_mode="step"
        )
        collector.initialize_configs(complex_env_info)
        collector.start_collection()
        
        # Collect data with complex observations
        for step in range(10):
            complex_obs = {
                'state': np.random.randn(1, 1, 4).astype(np.float32),
                'image': np.random.randn(1, 1, 32, 32, 3).astype(np.float32),
                'goal': np.random.randn(1, 1, 2).astype(np.float32)
            }
            
            action = {'action': np.random.randn(1, 1, 2).astype(np.float32)}
            reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
            
            collector.collect_step(
                observation=complex_obs,
                action=action,
                reward=reward,
                next_observation=complex_obs,
                done=(step == 9),
                truncated=False,
                info={'step': step}
            )
        
        # Verify collection
        assert collector.total_steps == 10
        assert collector.total_episodes == 1
        
        # Check that all observation types were collected
        assert 'observations_state' in collector.data_buffer
        assert 'observations_image' in collector.data_buffer
        assert 'observations_goal' in collector.data_buffer
        
        assert len(collector.data_buffer['observations_state']) == 10
        assert len(collector.data_buffer['observations_image']) == 10
        assert len(collector.data_buffer['observations_goal']) == 10
    
    def test_collector_data_consistency(self, temp_dir, mock_env, env_info):
        """Test data consistency across collection sessions."""
        collector = RLCollector(
            name="consistency_test_collector",
            save_path=temp_dir,
            collect_mode="step"
        )
        collector.initialize_configs(env_info)
        collector.start_collection()
        
        # First collection session
        obs, info = mock_env.reset()
        for step in range(5):
            action = mock_env.action_space.sample()
            next_obs, reward, done, truncated, step_info = mock_env.step(action)
            
            collector.collect_step(
                observation={'state': obs.reshape(1, 1, -1)},
                action={'action': action.reshape(1, 1, -1)},
                reward={'reward': np.array([[[reward]]]).astype(np.float32)},
                next_observation={'state': next_obs.reshape(1, 1, -1)},
                done=done,
                truncated=truncated,
                info=step_info
            )
            
            obs = next_obs
            if done or truncated:
                break
        
        # Save data
        collector.save_data()
        first_session_steps = collector.total_steps
        
        # Second collection session with same collector
        obs, info = mock_env.reset()
        for step in range(5):
            action = mock_env.action_space.sample()
            next_obs, reward, done, truncated, step_info = mock_env.step(action)
            
            collector.collect_step(
                observation={'state': obs.reshape(1, 1, -1)},
                action={'action': action.reshape(1, 1, -1)},
                reward={'reward': np.array([[[reward]]]).astype(np.float32)},
                next_observation={'state': next_obs.reshape(1, 1, -1)},
                done=done,
                truncated=truncated,
                info=step_info
            )
            
            obs = next_obs
            if done or truncated:
                break
        
        # Verify data consistency
        assert collector.total_steps == first_session_steps + 5
        
        # Check that all data arrays have consistent lengths
        buffer_sizes = [len(collector.data_buffer[key]) for key in collector.data_buffer.keys() if isinstance(collector.data_buffer[key], list)]
        assert all(size == collector.total_steps for size in buffer_sizes if size > 0) 