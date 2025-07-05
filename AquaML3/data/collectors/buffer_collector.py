"""Buffer Data Collector

This module provides a buffer-focused data collector with advanced
buffer management, memory optimization, and batch processing capabilities.
"""

from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import numpy as np
from loguru import logger
from pathlib import Path
from datetime import datetime
import threading
import queue
import time
from collections import deque
import psutil
import gc

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
from .utils import DataBuffer, CollectorUtils
from ..core_units import UnitConfig


class BufferCollector(BaseCollector):
    """
    Buffer Data Collector.
    
    This collector focuses on efficient buffer management with features like
    memory monitoring, batch processing, and intelligent buffer optimization.
    """
    
    def __init__(self,
                 name: str = "buffer_collector",
                 save_path: Optional[Union[str, Path]] = None,
                 auto_save: bool = True,
                 max_buffer_size: int = 1000000,
                 buffer_batch_size: int = 10000,
                 enable_memory_monitoring: bool = True,
                 memory_limit_mb: int = 1024,
                 compression_enabled: bool = True,
                 buffer_optimization: bool = True,
                 prefetch_enabled: bool = True,
                 prefetch_size: int = 1000):
        """
        Initialize the buffer collector.
        
        Args:
            name: Name of the collector
            save_path: Path to save collected data
            auto_save: Whether to auto-save data when buffer is full
            max_buffer_size: Maximum size of the data buffer
            buffer_batch_size: Size of batches for processing
            enable_memory_monitoring: Whether to monitor memory usage
            memory_limit_mb: Memory limit in MB
            compression_enabled: Whether to enable data compression
            buffer_optimization: Whether to enable buffer optimization
            prefetch_enabled: Whether to enable data prefetching
            prefetch_size: Size of prefetch buffer
        """
        super().__init__(name, save_path, auto_save, max_buffer_size)
        
        self.buffer_batch_size = buffer_batch_size
        self.enable_memory_monitoring = enable_memory_monitoring
        self.memory_limit_mb = memory_limit_mb
        self.compression_enabled = compression_enabled
        self.buffer_optimization = buffer_optimization
        self.prefetch_enabled = prefetch_enabled
        self.prefetch_size = prefetch_size
        
        # Advanced buffer management
        self.buffer_manager = DataBuffer(
            max_size=max_buffer_size,
            enable_compression=compression_enabled
        )
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor(memory_limit_mb) if enable_memory_monitoring else None
        
        # Batch processing
        self.batch_queue = queue.Queue() if prefetch_enabled else None
        self.batch_processor = BatchProcessor(self.buffer_batch_size) if buffer_optimization else None
        
        # Buffer statistics
        self.buffer_stats = {
            'total_items_added': 0,
            'total_items_removed': 0,
            'buffer_flushes': 0,
            'memory_warnings': 0,
            'compression_ratio': 0.0,
            'avg_batch_size': 0.0,
            'processing_times': deque(maxlen=1000)
        }
        
        # Threading for async operations
        self.async_enabled = False
        self.async_thread = None
        self.stop_async = False
        
        logger.info(f"Initialized buffer collector '{name}' with buffer_size={max_buffer_size}, "
                   f"batch_size={buffer_batch_size}")
    
    def collect_step(self,
                    observation: Dict[str, Any],
                    action: Dict[str, Any],
                    reward: Dict[str, Any],
                    next_observation: Dict[str, Any],
                    done: bool,
                    truncated: bool,
                    info: Dict[str, Any]) -> None:
        """
        Collect data from a single step with buffer optimization.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is done
            truncated: Whether episode is truncated
            info: Additional info
        """
        start_time = time.time()
        
        # Memory check
        if self.memory_monitor and self.memory_monitor.check_memory_usage():
            self._handle_memory_warning()
        
        # Validate and convert data
        self._validate_data_shapes(observation, self.observation_cfg)
        self._validate_data_shapes(action, self.action_cfg)
        self._validate_data_shapes(reward, self.reward_cfg)
        
        # Convert to numpy arrays
        observation_np = {k: self._convert_to_numpy(v) for k, v in observation.items()}
        action_np = {k: self._convert_to_numpy(v) for k, v in action.items()}
        reward_np = {k: self._convert_to_numpy(v) for k, v in reward.items()}
        next_observation_np = {k: self._convert_to_numpy(v) for k, v in next_observation.items()}
        
        # Create step data
        step_data = {
            'observation': observation_np,
            'action': action_np,
            'reward': reward_np,
            'next_observation': next_observation_np,
            'done': done,
            'truncated': truncated,
            'info': info,
            'timestamp': datetime.now()
        }
        
        # Add to buffer
        self._add_to_buffer(step_data)
        
        # Update statistics
        self.total_steps += 1
        self.buffer_stats['total_items_added'] += 1
        
        # Process batch if needed
        if self.batch_processor and self.buffer_manager.size() >= self.buffer_batch_size:
            self._process_batch()
        
        # Auto-save check
        if self._should_auto_save():
            self._save_and_flush()
        
        # Update processing time
        processing_time = time.time() - start_time
        self.buffer_stats['processing_times'].append(processing_time)
        
        # Episode completion
        if done or truncated:
            self.total_episodes += 1
    
    def _add_to_buffer(self, step_data: Dict[str, Any]) -> None:
        """Add step data to buffer with optimization."""
        # Add observations
        for key, value in step_data['observation'].items():
            self.buffer_manager.add(f'observations_{key}', value)
        
        # Add actions
        for key, value in step_data['action'].items():
            self.buffer_manager.add(f'actions_{key}', value)
        
        # Add rewards
        for key, value in step_data['reward'].items():
            self.buffer_manager.add(f'rewards_{key}', value)
        
        # Add next observations
        for key, value in step_data['next_observation'].items():
            self.buffer_manager.add(f'next_observations_{key}', value)
        
        # Add other data
        self.buffer_manager.add('dones', step_data['done'])
        self.buffer_manager.add('truncated', step_data['truncated'])
        self.buffer_manager.add('infos', step_data['info'])
        self.buffer_manager.add('timestamps', step_data['timestamp'])
    
    def _process_batch(self) -> None:
        """Process a batch of data."""
        if not self.batch_processor:
            return
        
        # Get batch data
        batch_data = self._get_batch_data()
        
        # Process batch
        processed_batch = self.batch_processor.process_batch(batch_data)
        
        # Store processed batch if needed
        if processed_batch:
            self._store_processed_batch(processed_batch)
        
        logger.debug(f"Processed batch of size {len(batch_data)}")
    
    def _get_batch_data(self) -> Dict[str, List[Any]]:
        """Get batch data from buffer."""
        batch_data = {}
        
        # Get data from buffer manager
        all_data = self.buffer_manager.get_all_data()
        
        # Take last batch_size items
        for key, values in all_data.items():
            if len(values) >= self.buffer_batch_size:
                batch_data[key] = values[-self.buffer_batch_size:]
        
        return batch_data
    
    def _store_processed_batch(self, processed_batch: Dict[str, Any]) -> None:
        """Store processed batch data."""
        # Store in data buffer for compatibility
        for key, value in processed_batch.items():
            if key not in self.data_buffer:
                self.data_buffer[key] = []
            self.data_buffer[key].append(value)
    
    def _handle_memory_warning(self) -> None:
        """Handle memory warning by reducing buffer size."""
        self.buffer_stats['memory_warnings'] += 1
        logger.warning(f"Memory usage high, flushing buffer for collector '{self.name}'")
        
        # Save and flush buffer
        self._save_and_flush()
        
        # Force garbage collection
        gc.collect()
    
    def _save_and_flush(self) -> None:
        """Save data and flush buffer."""
        if self.save_path:
            self.save_data()
        
        # Flush buffer
        self.buffer_manager.clear()
        self.buffer_stats['buffer_flushes'] += 1
        
        logger.debug(f"Flushed buffer for collector '{self.name}'")
    
    def collect_episode(self, episode_data: Dict[str, Any]) -> None:
        """
        Collect data from a complete episode.
        
        Args:
            episode_data: Complete episode data
        """
        if not episode_data:
            return
        
        # Process episode data in batches
        episode_length = len(episode_data.get('observations', []))
        
        for i in range(0, episode_length, self.buffer_batch_size):
            batch_end = min(i + self.buffer_batch_size, episode_length)
            
            # Extract batch
            batch_data = {}
            for key, values in episode_data.items():
                if isinstance(values, list) and len(values) == episode_length:
                    batch_data[key] = values[i:batch_end]
            
            # Process batch
            if batch_data:
                self._process_episode_batch(batch_data)
        
        self.total_episodes += 1
        logger.debug(f"Collected episode with {episode_length} steps")
    
    def _process_episode_batch(self, batch_data: Dict[str, List[Any]]) -> None:
        """Process a batch of episode data."""
        batch_size = len(batch_data.get('observations', []))
        
        for i in range(batch_size):
            step_data = {}
            for key, values in batch_data.items():
                if i < len(values):
                    step_data[key] = values[i]
            
            if step_data:
                self._add_episode_step_to_buffer(step_data)
    
    def _add_episode_step_to_buffer(self, step_data: Dict[str, Any]) -> None:
        """Add a step from episode data to buffer."""
        # Handle different data formats
        if 'observations' in step_data:
            for key, value in step_data['observations'].items():
                self.buffer_manager.add(f'observations_{key}', value)
        
        if 'actions' in step_data:
            for key, value in step_data['actions'].items():
                self.buffer_manager.add(f'actions_{key}', value)
        
        if 'rewards' in step_data:
            for key, value in step_data['rewards'].items():
                self.buffer_manager.add(f'rewards_{key}', value)
        
        # Add other data
        for key in ['dones', 'truncated', 'infos']:
            if key in step_data:
                self.buffer_manager.add(key, step_data[key])
        
        # Update step count
        self.total_steps += 1
        self.buffer_stats['total_items_added'] += 1
    
    def get_buffer_data(self, 
                       batch_size: Optional[int] = None,
                       format: str = "numpy") -> Dict[str, Any]:
        """
        Get data from buffer.
        
        Args:
            batch_size: Size of batch to return (None for all)
            format: Data format ("numpy" or "torch")
            
        Returns:
            Buffer data
        """
        all_data = self.buffer_manager.get_all_data()
        
        if batch_size:
            # Return last batch_size items
            batched_data = {}
            for key, values in all_data.items():
                if len(values) >= batch_size:
                    batched_data[key] = values[-batch_size:]
                else:
                    batched_data[key] = values
            all_data = batched_data
        
        # Convert format if needed
        if format == "torch":
            all_data = CollectorUtils.convert_data_format(all_data, "torch")
        
        return all_data
    
    def enable_async_processing(self) -> None:
        """Enable asynchronous processing."""
        if self.async_enabled:
            logger.warning("Async processing already enabled")
            return
        
        self.async_enabled = True
        self.stop_async = False
        self.async_thread = threading.Thread(target=self._async_processor)
        self.async_thread.start()
        
        logger.info("Enabled async processing")
    
    def disable_async_processing(self) -> None:
        """Disable asynchronous processing."""
        if not self.async_enabled:
            return
        
        self.stop_async = True
        if self.async_thread:
            self.async_thread.join()
        
        self.async_enabled = False
        logger.info("Disabled async processing")
    
    def _async_processor(self) -> None:
        """Async processor worker."""
        while not self.stop_async:
            try:
                # Process buffer in background
                if self.buffer_manager.size() >= self.buffer_batch_size:
                    self._process_batch()
                
                # Memory monitoring
                if self.memory_monitor and self.memory_monitor.check_memory_usage():
                    self._handle_memory_warning()
                
                time.sleep(0.1)  # Small delay to prevent CPU overuse
                
            except Exception as e:
                logger.error(f"Error in async processor: {e}")
    
    def get_buffer_statistics(self) -> Dict[str, Any]:
        """Get comprehensive buffer statistics."""
        base_stats = self.get_statistics()
        
        # Add buffer-specific statistics
        buffer_stats = {
            'buffer_size': self.buffer_manager.size(),
            'buffer_capacity': self.max_buffer_size,
            'buffer_usage': self.buffer_manager.size() / self.max_buffer_size if self.max_buffer_size > 0 else 0,
            'items_added': self.buffer_stats['total_items_added'],
            'items_removed': self.buffer_stats['total_items_removed'],
            'buffer_flushes': self.buffer_stats['buffer_flushes'],
            'memory_warnings': self.buffer_stats['memory_warnings'],
        }
        
        # Processing time statistics
        if self.buffer_stats['processing_times']:
            processing_times = list(self.buffer_stats['processing_times'])
            buffer_stats['avg_processing_time'] = np.mean(processing_times)
            buffer_stats['max_processing_time'] = np.max(processing_times)
            buffer_stats['min_processing_time'] = np.min(processing_times)
        
        # Memory statistics
        if self.memory_monitor:
            buffer_stats['memory_usage'] = self.memory_monitor.get_memory_usage()
        
        # Merge with base stats
        base_stats.update(buffer_stats)
        
        return base_stats
    
    def optimize_buffer(self) -> None:
        """Optimize buffer performance."""
        if not self.buffer_optimization:
            return
        
        logger.info("Optimizing buffer performance")
        
        # Clear unnecessary data
        gc.collect()
        
        # Optimize data types
        self._optimize_data_types()
        
        # Compress data if enabled
        if self.compression_enabled:
            self._compress_buffer_data()
        
        logger.info("Buffer optimization completed")
    
    def _optimize_data_types(self) -> None:
        """Optimize data types in buffer."""
        # This would implement data type optimization
        # For now, just log the action
        logger.debug("Optimizing data types in buffer")
    
    def _compress_buffer_data(self) -> None:
        """Compress buffer data."""
        # This would implement data compression
        # For now, just log the action
        logger.debug("Compressing buffer data")
    
    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return self.buffer_manager.size()
    
    def clear_buffer(self) -> None:
        """Clear the buffer."""
        self.buffer_manager.clear()
        self.buffer_stats['total_items_removed'] += self.buffer_stats['total_items_added']
        self.buffer_stats['total_items_added'] = 0
        
        logger.info(f"Cleared buffer for collector '{self.name}'")


class MemoryMonitor:
    """Memory monitoring utility."""
    
    def __init__(self, memory_limit_mb: int):
        self.memory_limit_mb = memory_limit_mb
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage exceeds limit."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss > self.memory_limit_bytes
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent()
        }


class BatchProcessor:
    """Batch processing utility."""
    
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.processing_functions: List[Callable] = []
    
    def add_processing_function(self, func: Callable) -> None:
        """Add a processing function."""
        self.processing_functions.append(func)
    
    def process_batch(self, batch_data: Dict[str, List[Any]]) -> Optional[Dict[str, Any]]:
        """Process a batch of data."""
        if not batch_data:
            return None
        
        processed_data = batch_data.copy()
        
        # Apply processing functions
        for func in self.processing_functions:
            try:
                processed_data = func(processed_data)
            except Exception as e:
                logger.error(f"Error in batch processing function: {e}")
        
        return processed_data 