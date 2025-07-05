"""
Performance Tests for Data Collectors

This module contains performance tests for data collectors,
testing memory usage, throughput, and scalability.
"""

import pytest
import numpy as np
import time
import psutil
import os
from pathlib import Path
import tempfile
import shutil

from AquaML.data.collectors.rl_collector import RLCollector
from AquaML.data.collectors.trajectory_collector import TrajectoryCollector
from AquaML.data.collectors.buffer_collector import BufferCollector


class TestCollectorPerformance:
    """Performance tests for data collectors."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def large_env_info(self):
        """Environment info with large observation spaces."""
        return {
            'observation_cfg': {
                'state': {
                    'name': 'state',
                    'dtype': 'float32',
                    'single_shape': (100,),
                    'size': 10000
                },
                'image': {
                    'name': 'image',
                    'dtype': 'float32',
                    'single_shape': (84, 84, 3),
                    'size': 1000
                }
            },
            'action_cfg': {
                'action': {
                    'name': 'action',
                    'dtype': 'float32',
                    'single_shape': (10,),
                    'size': 10000
                }
            },
            'reward_cfg': {
                'reward': {
                    'name': 'reward',
                    'dtype': 'float32',
                    'single_shape': (1,),
                    'size': 10000
                }
            }
        }
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def test_rl_collector_throughput(self, temp_dir, large_env_info):
        """Test RLCollector throughput with large data."""
        collector = RLCollector(
            name="performance_rl_collector",
            save_path=temp_dir,
            max_buffer_size=10000,
            collect_mode="step"
        )
        collector.initialize_configs(large_env_info)
        collector.start_collection()
        
        # Test parameters
        num_steps = 1000
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # Collect data
        for step in range(num_steps):
            observation = {
                'state': np.random.randn(1, 1, 100).astype(np.float32),
                'image': np.random.randn(1, 1, 84, 84, 3).astype(np.float32)
            }
            action = {'action': np.random.randn(1, 1, 10).astype(np.float32)}
            reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
            next_observation = {
                'state': np.random.randn(1, 1, 100).astype(np.float32),
                'image': np.random.randn(1, 1, 84, 84, 3).astype(np.float32)
            }
            
            collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=(step % 100 == 99),
                truncated=False,
                info={'step': step}
            )
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        # Calculate performance metrics
        duration = end_time - start_time
        throughput = num_steps / duration
        memory_usage = end_memory - start_memory
        
        print(f"RLCollector Performance:")
        print(f"  Steps: {num_steps}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} steps/s")
        print(f"  Memory Usage: {memory_usage:.2f} MB")
        
        # Performance assertions
        assert throughput > 100, f"Throughput too low: {throughput:.2f} steps/s"
        assert memory_usage < 500, f"Memory usage too high: {memory_usage:.2f} MB"
        
        # Verify data integrity
        assert collector.total_steps == num_steps
        assert collector.total_episodes == 10  # 100 steps per episode
    
    def test_trajectory_collector_scalability(self, temp_dir, large_env_info):
        """Test TrajectoryCollector scalability with many trajectories."""
        collector = TrajectoryCollector(
            name="performance_trajectory_collector",
            save_path=temp_dir,
            max_buffer_size=100,
            max_trajectory_length=50
        )
        collector.initialize_configs(large_env_info)
        collector.start_collection()
        
        # Test parameters
        num_trajectories = 50
        trajectory_length = 50
        total_steps = num_trajectories * trajectory_length
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # Collect trajectories
        for traj in range(num_trajectories):
            for step in range(trajectory_length):
                observation = {
                    'state': np.random.randn(1, 1, 100).astype(np.float32),
                    'image': np.random.randn(1, 1, 84, 84, 3).astype(np.float32)
                }
                action = {'action': np.random.randn(1, 1, 10).astype(np.float32)}
                reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
                next_observation = {
                    'state': np.random.randn(1, 1, 100).astype(np.float32),
                    'image': np.random.randn(1, 1, 84, 84, 3).astype(np.float32)
                }
                
                collector.collect_step(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    done=(step == trajectory_length - 1),
                    truncated=False,
                    info={'step': step, 'trajectory': traj}
                )
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        # Calculate performance metrics
        duration = end_time - start_time
        throughput = total_steps / duration
        memory_usage = end_memory - start_memory
        
        print(f"TrajectoryCollector Performance:")
        print(f"  Trajectories: {num_trajectories}")
        print(f"  Total Steps: {total_steps}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} steps/s")
        print(f"  Memory Usage: {memory_usage:.2f} MB")
        
        # Performance assertions
        assert throughput > 50, f"Throughput too low: {throughput:.2f} steps/s"
        assert memory_usage < 1000, f"Memory usage too high: {memory_usage:.2f} MB"
        
        # Verify trajectory collection
        assert collector.total_trajectories == num_trajectories
        assert collector.total_steps == total_steps
    
    def test_buffer_collector_memory_efficiency(self, temp_dir, large_env_info):
        """Test BufferCollector memory efficiency with auto-flushing."""
        collector = BufferCollector(
            name="performance_buffer_collector",
            save_path=temp_dir,
            max_memory_mb=200,  # 200MB limit
            auto_flush=True,
            memory_check_interval=10
        )
        collector.initialize_configs(large_env_info)
        collector.start_collection()
        
        # Test parameters
        num_steps = 2000
        start_time = time.time()
        start_memory = self.get_memory_usage()
        max_memory = start_memory
        
        # Collect data with memory monitoring
        for step in range(num_steps):
            observation = {
                'state': np.random.randn(1, 1, 100).astype(np.float32),
                'image': np.random.randn(1, 1, 84, 84, 3).astype(np.float32)
            }
            action = {'action': np.random.randn(1, 1, 10).astype(np.float32)}
            reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
            next_observation = {
                'state': np.random.randn(1, 1, 100).astype(np.float32),
                'image': np.random.randn(1, 1, 84, 84, 3).astype(np.float32)
            }
            
            collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=(step % 100 == 99),
                truncated=False,
                info={'step': step}
            )
            
            # Track peak memory usage
            current_memory = self.get_memory_usage()
            if current_memory > max_memory:
                max_memory = current_memory
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        # Calculate performance metrics
        duration = end_time - start_time
        throughput = num_steps / duration
        peak_memory_usage = max_memory - start_memory
        
        print(f"BufferCollector Performance:")
        print(f"  Steps: {num_steps}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} steps/s")
        print(f"  Peak Memory Usage: {peak_memory_usage:.2f} MB")
        print(f"  Flush Count: {collector.flush_counter}")
        
        # Performance assertions
        assert throughput > 50, f"Throughput too low: {throughput:.2f} steps/s"
        assert peak_memory_usage < 400, f"Peak memory usage too high: {peak_memory_usage:.2f} MB"
        
        # Verify auto-flushing occurred
        assert collector.flush_counter > 0, "Auto-flush should have been triggered"
        
        # Check flush files exist
        flush_files = list(temp_dir.glob(f"{collector.name}_flush_*.pkl"))
        assert len(flush_files) > 0, "Flush files should exist"
    
    def test_concurrent_collectors_performance(self, temp_dir, large_env_info):
        """Test performance with multiple collectors running concurrently."""
        # Create multiple collectors
        collectors = []
        for i in range(3):
            collector = RLCollector(
                name=f"concurrent_collector_{i}",
                save_path=temp_dir,
                max_buffer_size=1000,
                collect_mode="step"
            )
            collector.initialize_configs(large_env_info)
            collector.start_collection()
            collectors.append(collector)
        
        # Test parameters
        num_steps = 500
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # Collect data with all collectors
        for step in range(num_steps):
            observation = {
                'state': np.random.randn(1, 1, 100).astype(np.float32),
                'image': np.random.randn(1, 1, 84, 84, 3).astype(np.float32)
            }
            action = {'action': np.random.randn(1, 1, 10).astype(np.float32)}
            reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
            next_observation = {
                'state': np.random.randn(1, 1, 100).astype(np.float32),
                'image': np.random.randn(1, 1, 84, 84, 3).astype(np.float32)
            }
            
            # Collect with all collectors
            for collector in collectors:
                collector.collect_step(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    done=(step % 50 == 49),
                    truncated=False,
                    info={'step': step}
                )
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        # Calculate performance metrics
        duration = end_time - start_time
        total_collections = num_steps * len(collectors)
        throughput = total_collections / duration
        memory_usage = end_memory - start_memory
        
        print(f"Concurrent Collectors Performance:")
        print(f"  Collectors: {len(collectors)}")
        print(f"  Steps per collector: {num_steps}")
        print(f"  Total collections: {total_collections}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} collections/s")
        print(f"  Memory Usage: {memory_usage:.2f} MB")
        
        # Performance assertions
        assert throughput > 100, f"Throughput too low: {throughput:.2f} collections/s"
        assert memory_usage < 1000, f"Memory usage too high: {memory_usage:.2f} MB"
        
        # Verify all collectors worked
        for collector in collectors:
            assert collector.total_steps == num_steps
            assert collector.total_episodes == 10  # 50 steps per episode
    
    def test_data_export_performance(self, temp_dir, large_env_info):
        """Test performance of data export operations."""
        collector = RLCollector(
            name="export_performance_collector",
            save_path=temp_dir,
            max_buffer_size=5000,
            collect_mode="step"
        )
        collector.initialize_configs(large_env_info)
        collector.start_collection()
        
        # Collect substantial amount of data
        num_steps = 1000
        for step in range(num_steps):
            observation = {
                'state': np.random.randn(1, 1, 100).astype(np.float32),
                'image': np.random.randn(1, 1, 84, 84, 3).astype(np.float32)
            }
            action = {'action': np.random.randn(1, 1, 10).astype(np.float32)}
            reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
            next_observation = {
                'state': np.random.randn(1, 1, 100).astype(np.float32),
                'image': np.random.randn(1, 1, 84, 84, 3).astype(np.float32)
            }
            
            collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=(step % 100 == 99),
                truncated=False,
                info={'step': step}
            )
        
        # Test export performance
        export_path = temp_dir / "export_test"
        
        # Test offline RL export
        start_time = time.time()
        collector.export_for_offline_rl(export_path)
        offline_rl_duration = time.time() - start_time
        
        # Test teacher-student export
        start_time = time.time()
        collector.export_for_teacher_student(export_path)
        teacher_student_duration = time.time() - start_time
        
        # Test regular save
        start_time = time.time()
        collector.save_data()
        save_duration = time.time() - start_time
        
        print(f"Data Export Performance:")
        print(f"  Data Size: {num_steps} steps")
        print(f"  Offline RL Export: {offline_rl_duration:.2f}s")
        print(f"  Teacher-Student Export: {teacher_student_duration:.2f}s")
        print(f"  Regular Save: {save_duration:.2f}s")
        
        # Performance assertions
        assert offline_rl_duration < 10, f"Offline RL export too slow: {offline_rl_duration:.2f}s"
        assert teacher_student_duration < 10, f"Teacher-student export too slow: {teacher_student_duration:.2f}s"
        assert save_duration < 5, f"Regular save too slow: {save_duration:.2f}s"
        
        # Verify exports exist
        offline_file = export_path / "offline_rl_data.pkl"
        teacher_file = export_path / "teacher_student_data.pkl"
        regular_file = temp_dir / f"{collector.name}_data.pkl"
        
        assert offline_file.exists()
        assert teacher_file.exists()
        assert regular_file.exists()
    
    @pytest.mark.slow
    def test_large_scale_collection(self, temp_dir, large_env_info):
        """Test large-scale data collection (marked as slow test)."""
        collector = RLCollector(
            name="large_scale_collector",
            save_path=temp_dir,
            max_buffer_size=50000,
            collect_mode="step"
        )
        collector.initialize_configs(large_env_info)
        collector.start_collection()
        
        # Test parameters for large scale
        num_steps = 10000
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # Collect large amount of data
        for step in range(num_steps):
            observation = {
                'state': np.random.randn(1, 1, 100).astype(np.float32),
                'image': np.random.randn(1, 1, 84, 84, 3).astype(np.float32)
            }
            action = {'action': np.random.randn(1, 1, 10).astype(np.float32)}
            reward = {'reward': np.random.randn(1, 1, 1).astype(np.float32)}
            next_observation = {
                'state': np.random.randn(1, 1, 100).astype(np.float32),
                'image': np.random.randn(1, 1, 84, 84, 3).astype(np.float32)
            }
            
            collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=(step % 1000 == 999),
                truncated=False,
                info={'step': step}
            )
            
            # Progress reporting
            if step % 1000 == 0:
                current_memory = self.get_memory_usage()
                print(f"  Progress: {step}/{num_steps}, Memory: {current_memory:.2f} MB")
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        # Calculate performance metrics
        duration = end_time - start_time
        throughput = num_steps / duration
        memory_usage = end_memory - start_memory
        
        print(f"Large Scale Collection Performance:")
        print(f"  Steps: {num_steps}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} steps/s")
        print(f"  Memory Usage: {memory_usage:.2f} MB")
        
        # Performance assertions for large scale
        assert throughput > 50, f"Large scale throughput too low: {throughput:.2f} steps/s"
        assert memory_usage < 2000, f"Large scale memory usage too high: {memory_usage:.2f} MB"
        
        # Verify data integrity
        assert collector.total_steps == num_steps
        assert collector.total_episodes == 10  # 1000 steps per episode