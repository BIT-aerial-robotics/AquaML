"""
Tests for BufferCollector

This module contains unit tests for the BufferCollector class,
testing memory-optimized collection and buffer management.
"""

import pytest
import numpy as np
import psutil
import time
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch

from AquaML.data.collectors.buffer_collector import BufferCollector


class TestBufferCollector:
    """Test class for BufferCollector."""
    
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
    def buffer_collector(self, temp_dir):
        """Create a BufferCollector instance for testing."""
        return BufferCollector(
            name="test_buffer_collector",
            save_path=temp_dir,
            memory_limit_mb=100,
            enable_memory_monitoring=True,
            max_buffer_size=1000
        )
    
    def test_buffer_collector_initialization(self, temp_dir):
        """Test BufferCollector initialization."""
        collector = BufferCollector(
            name="test_buffer",
            save_path=temp_dir,
            memory_limit_mb=500,
            enable_memory_monitoring=True,
            buffer_batch_size=1000,
            compression_enabled=True
        )
        
        assert collector.name == "test_buffer"
        assert collector.memory_limit_mb == 500
        assert collector.enable_memory_monitoring is True
        assert collector.buffer_batch_size == 1000
        assert collector.compression_enabled is True
        assert collector.async_enabled is False
    
    def test_collect_step_with_memory_monitoring(self, buffer_collector, sample_env_info):
        """Test step collection with memory monitoring."""
        buffer_collector.initialize_configs(sample_env_info)
        buffer_collector.start_collection()
        
        # Mock memory monitoring
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 50 * 1024 * 1024  # 50MB
            
            # Collect steps
            for step in range(15):
                observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
                action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
                reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
                next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
                
                buffer_collector.collect_step(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    done=(step % 5 == 4),
                    truncated=False,
                    info={'step': step}
                )
        
        assert buffer_collector.total_steps == 15
        assert buffer_collector.total_episodes == 3
    
    def test_memory_check_and_auto_flush(self, buffer_collector, sample_env_info):
        """Test memory checking and auto-flush functionality."""
        buffer_collector.initialize_configs(sample_env_info)
        buffer_collector.start_collection()
        
        # Mock high memory usage to trigger flush
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 120 * 1024 * 1024  # 120MB (exceeds 100MB limit)
            
            # Mock _save_and_flush to track calls
            with patch.object(buffer_collector, '_save_and_flush') as mock_flush:
                # Collect enough steps to trigger memory check
                for step in range(15):
                    observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
                    action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
                    reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
                    next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
                    
                    buffer_collector.collect_step(
                        observation=observation,
                        action=action,
                        reward=reward,
                        next_observation=next_observation,
                        done=False,
                        truncated=False,
                        info={'step': step}
                    )
                
                # Should have triggered flush due to high memory usage
                assert mock_flush.call_count > 0
    
    def test_buffer_data_functionality(self, buffer_collector, sample_env_info, temp_dir):
        """Test buffer data functionality."""
        buffer_collector.initialize_configs(sample_env_info)
        buffer_collector.start_collection()
        
        # Collect some data
        for step in range(10):
            observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
            reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
            next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            
            buffer_collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=False,
                truncated=False,
                info={'step': step}
            )
        
        # Check that data was collected
        assert buffer_collector.total_steps == 10
        assert buffer_collector.get_buffer_size() > 0
        
        # Test get_buffer_data
        buffer_data = buffer_collector.get_buffer_data()
        assert isinstance(buffer_data, dict)
    
    def test_buffer_statistics(self, buffer_collector, sample_env_info, temp_dir):
        """Test buffer statistics functionality."""
        buffer_collector.initialize_configs(sample_env_info)
        buffer_collector.start_collection()
        
        # Collect some data
        for step in range(15):
            observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
            reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
            next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            
            buffer_collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=(step % 5 == 4),
                truncated=False,
                info={'step': step}
            )
        
        # Test get_buffer_statistics
        stats = buffer_collector.get_buffer_statistics()
        assert isinstance(stats, dict)
        assert 'items_added' in stats
        assert stats['items_added'] > 0
        
        # Test basic buffer operations
        assert buffer_collector.total_steps == 15
        assert buffer_collector.total_episodes == 3
    
    def test_memory_monitoring_disabled(self, temp_dir, sample_env_info):
        """Test collector with memory monitoring disabled."""
        collector = BufferCollector(
            name="no_memory_monitor",
            save_path=temp_dir,
            enable_memory_monitoring=False  # Disable memory monitoring
        )
        collector.initialize_configs(sample_env_info)
        collector.start_collection()
        
        # Should not check memory or auto-flush
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 1000 * 1024 * 1024  # 1GB
            
            # Collect many steps
            for step in range(20):
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
            
            # Should not have called memory monitoring
            assert not mock_process.called
    
    def test_compression_functionality(self, temp_dir, sample_env_info):
        """Test data compression functionality."""
        collector = BufferCollector(
            name="compression_test",
            save_path=temp_dir,
            compression_enabled=True,  # Enable compression
            auto_save=False
        )
        collector.initialize_configs(sample_env_info)
        collector.start_collection()
        
        # Collect data
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
        
        # Test basic functionality with compression enabled
        assert collector.total_steps == 10
        assert collector.compression_enabled is True
    
    def test_collect_episode_with_buffering(self, buffer_collector, sample_env_info):
        """Test episode collection with buffering."""
        buffer_collector.initialize_configs(sample_env_info)
        buffer_collector.start_collection()
        
        # Create episode data
        episode_data = {
            'observations': [
                {'state': np.random.randn(1, 1, 4).astype(np.float32)} for _ in range(12)
            ],
            'actions': [
                {'action': np.random.randn(1, 1, 1).astype(np.float32)} for _ in range(12)
            ],
            'rewards': [
                {'reward': np.random.randn(1, 1, 1).astype(np.float32)} for _ in range(12)
            ],
            'dones': [False] * 11 + [True],
            'truncated': [False] * 12,
            'infos': [{'step': i} for i in range(12)]
        }
        
        buffer_collector.collect_episode(episode_data)
        
        assert buffer_collector.total_episodes == 1
        assert buffer_collector.total_steps == 12
    
    def test_get_memory_usage(self, buffer_collector):
        """Test memory usage reporting."""
        if buffer_collector.memory_monitor:
            usage = buffer_collector.memory_monitor.get_memory_usage()
            
            assert 'rss_mb' in usage
            assert 'vms_mb' in usage
            assert 'percent' in usage
            assert isinstance(usage['rss_mb'], (int, float))
            assert isinstance(usage['vms_mb'], (int, float))
            assert isinstance(usage['percent'], (int, float))
    
    def test_memory_monitoring_functionality(self, buffer_collector, sample_env_info):
        """Test memory monitoring functionality."""
        buffer_collector.initialize_configs(sample_env_info)
        buffer_collector.start_collection()
        
        # Test memory monitoring when enabled
        if buffer_collector.memory_monitor:
            # Mock memory check
            with patch.object(buffer_collector.memory_monitor, 'check_memory_usage') as mock_check:
                mock_check.return_value = True
                
                # This should trigger memory warning handling
                buffer_collector._handle_memory_warning()
                
                # Verify memory warning was recorded
                assert buffer_collector.buffer_stats['memory_warnings'] > 0
    
    def test_buffer_optimization(self, buffer_collector, sample_env_info):
        """Test buffer optimization functionality."""
        buffer_collector.initialize_configs(sample_env_info)
        buffer_collector.start_collection()
        
        # Collect some data
        for step in range(8):
            observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            action = {'action': np.random.randn(1, 1, 1).astype(np.float32)}
            reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
            next_observation = {'state': np.random.randn(1, 1, 4).astype(np.float32)}
            
            buffer_collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=False,
                truncated=False,
                info={'step': step}
            )
        
        # Test buffer optimization
        if buffer_collector.buffer_optimization:
            buffer_collector.optimize_buffer()
            
        # Test buffer clear functionality
        buffer_collector.clear_buffer()
        assert buffer_collector.get_buffer_size() == 0 