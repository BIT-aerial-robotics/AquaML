# AquaML Data Collection System

A comprehensive data collection system for reinforcement learning, designed to support multiple environments, learning scenarios, and data formats.

## Features

### 🎯 **Multiple Collector Types**
- **RLCollector**: Specialized for reinforcement learning data collection
- **TrajectoryCollector**: Focused on complete trajectory/episode collection
- **BufferCollector**: Optimized for buffer management and memory efficiency

### 🌍 **Environment Support**
- **Gymnasium**: Single and vectorized environments
- **Isaac Lab**: Support for robotics simulation environments
- **Custom Environments**: Extensible wrapper system

### 📊 **Data Formats**
- **Dict-based**: Structured data with multiple observation/action/reward units
- **Numpy/Torch**: Seamless conversion between formats
- **Compression**: HDF5 and pickle support with compression

### 🎓 **Learning Scenarios**
- **Online RL**: Real-time data collection during training
- **Offline RL**: Dataset generation for offline learning
- **Teacher-Student**: Demonstration data for imitation learning
- **Multi-Agent**: Support for multi-agent environments

## Quick Start

### Basic Usage

```python
from AquaML.data.collectors import RLCollector
from AquaML.environment.wrappers.gymnasium_envs import GymnasiumWrapper

# Create environment
env = GymnasiumWrapper("CartPole-v1")
env_info = env.getEnvInfo()

# Create collector
collector = RLCollector(
    name="my_collector",
    save_path="./data",
    max_buffer_size=100000,
    enable_trajectory_buffer=True
)

# Initialize with environment info
collector.initialize_configs(env_info)
collector.start_collection()

# Collect data
observation, info = env.reset()
for step in range(1000):
    action = {'action': np.random.randint(0, 2, size=(1, 1, 1))}
    next_observation, reward, done, truncated, info = env.step(action)
    
    collector.collect_step(
        observation=observation,
        action=action,
        reward=reward,
        next_observation=next_observation,
        done=done,
        truncated=truncated,
        info=info
    )
    
    observation = next_observation
    if done or truncated:
        observation, info = env.reset()

# End collection and save
collector.end_collection()
```

### Advanced Features

#### Trajectory Collection with Filtering
```python
from AquaML.data.collectors import TrajectoryCollector

# Create collector with filtering
trajectory_filter = {
    'min_reward': 100.0,
    'min_length': 10,
    'allowed_completion_types': ['done']
}

collector = TrajectoryCollector(
    name="filtered_trajectories",
    save_path="./trajectories",
    enable_trajectory_filtering=True,
    trajectory_filter_criteria=trajectory_filter
)
```

#### Buffer Management
```python
from AquaML.data.collectors import BufferCollector

# Create buffer collector with memory monitoring
collector = BufferCollector(
    name="efficient_buffer",
    save_path="./buffers",
    max_buffer_size=1000000,
    enable_memory_monitoring=True,
    memory_limit_mb=2048,
    compression_enabled=True
)

# Enable async processing
collector.enable_async_processing()
```

## Data Export

### Offline RL
```python
# Export data for offline RL algorithms
collector.export_for_offline_rl("./offline_data")
```

### Teacher-Student Learning
```python
# Export demonstrations for imitation learning
collector.export_for_teacher_student("./demonstrations")
```

### Custom Export
```python
# Export trajectories in different formats
trajectory_collector.export_trajectories(
    "./trajectories.pkl",
    format="pickle",
    include_metadata=True
)
```

## Architecture

### Core Components

#### BaseCollector
- Abstract base class for all collectors
- Handles configuration management
- Provides data validation and conversion
- Manages save/load operations

#### RLCollector
- Specialized for RL data collection
- Supports step-by-step and episode-based collection
- Async collection capabilities
- Built-in trajectory buffer

#### TrajectoryCollector
- Focuses on complete trajectory collection
- Advanced filtering and sorting capabilities
- Trajectory statistics and analysis
- Multiple export formats

#### BufferCollector
- Optimized for memory efficiency
- Batch processing capabilities
- Memory monitoring and management
- Compression support

### Data Units

The system uses AquaML's `UnitConfig` system for structured data:

```python
# Example configuration
observation_cfg = {
    'state': UnitConfig(
        name='state',
        dtype=np.float32,
        single_shape=(4,),
        size=10000
    )
}
```

## Integration Examples

### Isaac Lab Integration
```python
# Conceptual Isaac Lab integration
isaac_config = {
    'observation_cfg': {
        'proprioceptive': {
            'name': 'proprioceptive',
            'dtype': 'float32',
            'single_shape': (48,),
            'size': 10000
        },
        'visual': {
            'name': 'visual',
            'dtype': 'float32',
            'single_shape': (84, 84, 3),
            'size': 10000
        }
    },
    'action_cfg': {
        'joint_actions': {
            'name': 'joint_actions',
            'dtype': 'float32',
            'single_shape': (12,),
            'size': 10000
        }
    }
}

collector = RLCollector(
    name="isaac_lab_collector",
    save_path="./isaac_data",
    max_buffer_size=200000,
    async_collection=True
)
```

### Vector Environment Support
```python
from AquaML.environment.wrappers.gymnasium_vector_envs import GymnasiumVectorWrapper

# Create vector environment
env = GymnasiumVectorWrapper("CartPole-v1", num_envs=8)

# Use buffer collector for efficient parallel collection
collector = BufferCollector(
    name="vector_collector",
    buffer_batch_size=1000,
    enable_memory_monitoring=True
)
```

## Performance Features

### Memory Management
- Automatic memory monitoring
- Configurable memory limits
- Intelligent buffer flushing
- Garbage collection optimization

### Async Processing
- Non-blocking data collection
- Background processing threads
- Queue-based data handling
- Parallel environment support

### Data Compression
- HDF5 compression support
- Pickle optimization
- Configurable compression levels
- Memory-efficient storage

## Statistics and Analysis

### Collection Statistics
```python
# Get comprehensive statistics
stats = collector.get_dataset_statistics()
print(f"Total steps: {stats['total_steps']}")
print(f"Total episodes: {stats['total_episodes']}")
print(f"Average episode length: {stats['avg_episode_length']}")
print(f"Average episode reward: {stats['avg_episode_reward']}")
```

### Trajectory Analysis
```python
# Get trajectory statistics
traj_stats = trajectory_collector.get_trajectory_statistics()
print(f"Total trajectories: {traj_stats['total_trajectories']}")
print(f"Trajectory length stats: {traj_stats['trajectory_lengths']}")
print(f"Trajectory reward stats: {traj_stats['trajectory_rewards']}")
```

## Error Handling

The system includes comprehensive error handling:
- Data validation with shape checking
- Type conversion with fallback handling
- Memory overflow protection
- Graceful degradation on errors

## Extensibility

The modular design allows for easy extension:
- Custom collectors by inheriting from `BaseCollector`
- Environment-specific wrappers
- Custom data processing functions
- Pluggable storage backends

## Best Practices

1. **Memory Management**: Monitor memory usage in long-running collections
2. **Batch Processing**: Use appropriate batch sizes for your hardware
3. **Data Validation**: Always validate data shapes and types
4. **Async Collection**: Use async processing for high-throughput scenarios
5. **Compression**: Enable compression for large datasets

## Dependencies

- numpy
- torch
- gymnasium
- loguru
- h5py
- psutil
- pathlib

## License

This system is part of the AquaML framework and follows the same licensing terms. 