"""Data Collector Demo

This module demonstrates how to use the AquaML data collection system
with different environment wrappers for various learning scenarios.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import time
from loguru import logger

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

from .rl_collector import RLCollector
from .trajectory_collector import TrajectoryCollector
from .buffer_collector import BufferCollector
from ..core_units import UnitConfig
from ...environment.wrappers.gymnasium_envs import GymnasiumWrapper
from ...environment.wrappers.gymnasium_vector_envs import GymnasiumVectorWrapper


class CollectorDemo:
    """
    Demonstration class for the AquaML data collection system.
    
    This class shows how to use different collectors with various
    environments and configurations.
    """
    
    def __init__(self, save_path: Optional[str] = None):
        """
        Initialize the demo.
        
        Args:
            save_path: Path to save collected data
        """
        self.save_path = Path(save_path) if save_path else Path("./collected_data")
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize collectors
        self.rl_collector = None
        self.trajectory_collector = None
        self.buffer_collector = None
        
        logger.info("Initialized CollectorDemo")
    
    def demo_gymnasium_collection(self, env_name: str = "CartPole-v1", num_episodes: int = 10) -> None:
        """
        Demonstrate data collection with Gymnasium environment.
        
        Args:
            env_name: Name of the Gymnasium environment
            num_episodes: Number of episodes to collect
        """
        logger.info(f"Starting Gymnasium collection demo with {env_name}")
        
        # Create environment wrapper
        env = GymnasiumWrapper(env_name)
        
        # Get environment info
        env_info = env.getEnvInfo()
        
        # Initialize RL collector
        self.rl_collector = RLCollector(
            name=f"gym_{env_name}_collector",
            save_path=self.save_path / "gymnasium",
            max_buffer_size=50000,
            enable_trajectory_buffer=True
        )
        
        # Initialize collector with environment info
        self.rl_collector.initialize_configs(env_info)
        self.rl_collector.start_collection()
        
        # Collect data
        for episode in range(num_episodes):
            observation, info = env.reset()
            episode_reward = 0
            
            while True:
                # Random action for demo
                action_space_shape = env.action_cfg_['action'].single_shape
                if len(action_space_shape) == 0:  # Discrete action
                    action = {'action': np.array([np.random.randint(0, 2)])}
                else:  # Continuous action
                    action = {'action': np.random.uniform(-1, 1, action_space_shape)}
                
                # Step environment
                next_observation, reward, done, truncated, info = env.step(action)
                
                # Collect step data
                self.rl_collector.collect_step(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    done=done,
                    truncated=truncated,
                    info=info
                )
                
                episode_reward += reward['reward'][0, 0]
                observation = next_observation
                
                if done or truncated:
                    break
            
            logger.info(f"Episode {episode + 1}/{num_episodes} completed, reward: {episode_reward:.2f}")
        
        # End collection
        self.rl_collector.end_collection()
        
        # Print statistics
        stats = self.rl_collector.get_dataset_statistics()
        logger.info(f"Collection completed. Statistics: {stats}")
        
        # Export for different use cases
        self._export_collected_data(self.rl_collector, env_name)
    
    def demo_vector_environment_collection(self, env_name: str = "CartPole-v1", 
                                         num_envs: int = 4, num_episodes: int = 20) -> None:
        """
        Demonstrate data collection with vectorized environments.
        
        Args:
            env_name: Name of the environment
            num_envs: Number of parallel environments
            num_episodes: Total number of episodes to collect
        """
        logger.info(f"Starting vector environment collection demo with {num_envs} parallel {env_name}")
        
        # Create vector environment wrapper
        env = GymnasiumVectorWrapper(env_name, num_envs)
        
        # Get environment info
        env_info = env.getEnvInfo()
        
        # Initialize buffer collector for efficient parallel collection
        self.buffer_collector = BufferCollector(
            name=f"vector_{env_name}_collector",
            save_path=self.save_path / "vector",
            max_buffer_size=100000,
            buffer_batch_size=1000,
            enable_memory_monitoring=True
        )
        
        # Initialize collector with environment info
        self.buffer_collector.initialize_configs(env_info)
        self.buffer_collector.start_collection()
        
        # Collect data
        episodes_completed = 0
        observation, info = env.reset()
        
        while episodes_completed < num_episodes:
            # Random actions for demo
            action_space_shape = env.action_cfg_['action'].single_shape
            if len(action_space_shape) == 0:  # Discrete action
                actions = {'action': np.random.randint(0, 2, size=(1, num_envs, 1))}
            else:  # Continuous action
                actions = {'action': np.random.uniform(-1, 1, size=(1, num_envs) + action_space_shape)}
            
            # Step environment
            next_observation, reward, done, truncated, info = env.step(actions)
            
            # Collect step data
            self.buffer_collector.collect_step(
                observation=observation,
                action=actions,
                reward=reward,
                next_observation=next_observation,
                done=done,
                truncated=truncated,
                info=info
            )
            
            # Count completed episodes
            episodes_completed += np.sum(done) + np.sum(truncated)
            observation = next_observation
        
        # End collection
        self.buffer_collector.end_collection()
        
        # Print statistics
        stats = self.buffer_collector.get_buffer_statistics()
        logger.info(f"Vector collection completed. Statistics: {stats}")
    
    def demo_trajectory_collection(self, env_name: str = "CartPole-v1", 
                                 num_episodes: int = 15) -> None:
        """
        Demonstrate trajectory-focused data collection.
        
        Args:
            env_name: Name of the environment
            num_episodes: Number of episodes to collect
        """
        logger.info(f"Starting trajectory collection demo with {env_name}")
        
        # Create environment wrapper
        env = GymnasiumWrapper(env_name)
        
        # Get environment info
        env_info = env.getEnvInfo()
        
        # Initialize trajectory collector with filtering
        trajectory_filter = {
            'min_reward': 10.0,  # Only keep trajectories with reward >= 10
            'min_length': 5,     # Minimum trajectory length
            'allowed_completion_types': ['done', 'truncated']
        }
        
        self.trajectory_collector = TrajectoryCollector(
            name=f"trajectory_{env_name}_collector",
            save_path=self.save_path / "trajectories",
            max_buffer_size=1000,
            max_trajectory_length=500,
            min_trajectory_length=5,
            enable_trajectory_filtering=True,
            trajectory_filter_criteria=trajectory_filter
        )
        
        # Initialize collector with environment info
        self.trajectory_collector.initialize_configs(env_info)
        self.trajectory_collector.start_collection()
        
        # Collect data
        for episode in range(num_episodes):
            observation, info = env.reset()
            episode_reward = 0
            
            while True:
                # Random action for demo
                action_space_shape = env.action_cfg_['action'].single_shape
                if len(action_space_shape) == 0:  # Discrete action
                    action = {'action': np.array([np.random.randint(0, 2)])}
                else:  # Continuous action
                    action = {'action': np.random.uniform(-1, 1, action_space_shape)}
                
                # Step environment
                next_observation, reward, done, truncated, info = env.step(action)
                
                # Collect step data
                self.trajectory_collector.collect_step(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    done=done,
                    truncated=truncated,
                    info=info
                )
                
                episode_reward += reward['reward'][0, 0]
                observation = next_observation
                
                if done or truncated:
                    break
            
            logger.info(f"Episode {episode + 1}/{num_episodes} completed, reward: {episode_reward:.2f}")
        
        # End collection
        self.trajectory_collector.end_collection()
        
        # Print trajectory statistics
        stats = self.trajectory_collector.get_trajectory_statistics()
        logger.info(f"Trajectory collection completed. Statistics: {stats}")
        
        # Export trajectories
        self.trajectory_collector.export_trajectories(
            self.save_path / "trajectories" / f"{env_name}_trajectories.pkl",
            format="pickle",
            include_metadata=True
        )
    
    def _export_collected_data(self, collector: RLCollector, env_name: str) -> None:
        """Export collected data for different use cases."""
        
        # Export for offline RL
        offline_rl_path = self.save_path / "offline_rl" / env_name
        collector.export_for_offline_rl(offline_rl_path)
        logger.info(f"Exported data for offline RL to {offline_rl_path}")
        
        # Export for teacher-student learning
        teacher_student_path = self.save_path / "teacher_student" / env_name
        collector.export_for_teacher_student(teacher_student_path)
        logger.info(f"Exported data for teacher-student learning to {teacher_student_path}")
    
    def demonstrate_isaac_lab_integration(self) -> None:
        """
        Demonstrate how to integrate with Isaac Lab environment.
        
        Note: This is a conceptual demonstration. Actual Isaac Lab integration
        would require the Isaac Lab environment wrapper.
        """
        logger.info("Demonstrating Isaac Lab integration concept")
        
        # Conceptual Isaac Lab environment configuration
        isaac_lab_config = {
            'observation_cfg': {
                'proprioceptive': {
                    'name': 'proprioceptive',
                    'dtype': 'float32',
                    'single_shape': (48,),  # Example: robot joint states
                    'size': 10000
                },
                'visual': {
                    'name': 'visual',
                    'dtype': 'float32',
                    'single_shape': (84, 84, 3),  # Example: camera image
                    'size': 10000
                }
            },
            'action_cfg': {
                'joint_actions': {
                    'name': 'joint_actions',
                    'dtype': 'float32',
                    'single_shape': (12,),  # Example: joint torques
                    'size': 10000
                }
            },
            'reward_cfg': {
                'task_reward': {
                    'name': 'task_reward',
                    'dtype': 'float32',
                    'single_shape': (1,),
                    'size': 10000
                }
            }
        }
        
        # Initialize specialized collector for Isaac Lab
        isaac_collector = RLCollector(
            name="isaac_lab_collector",
            save_path=self.save_path / "isaac_lab",
            max_buffer_size=200000,  # Larger buffer for complex environments
            enable_trajectory_buffer=True,
            async_collection=True  # Use async for performance
        )
        
        # Initialize with Isaac Lab config
        isaac_collector.initialize_configs(isaac_lab_config)
        isaac_collector.start_collection()
        
        # Start async collection
        isaac_collector.start_async_collection()
        
        # Simulate data collection
        logger.info("Simulating Isaac Lab data collection...")
        
        for step in range(100):  # Simulate 100 steps
            # Simulate Isaac Lab observations
            observation = {
                'proprioceptive': np.random.randn(1, 1, 48).astype(np.float32),
                'visual': np.random.randint(0, 255, (1, 1, 84, 84, 3), dtype=np.uint8)
            }
            
            # Simulate actions
            action = {
                'joint_actions': np.random.randn(1, 1, 12).astype(np.float32)
            }
            
            # Simulate rewards
            reward = {
                'task_reward': np.random.randn(1, 1, 1).astype(np.float32)
            }
            
            # Simulate next observation
            next_observation = {
                'proprioceptive': np.random.randn(1, 1, 48).astype(np.float32),
                'visual': np.random.randint(0, 255, (1, 1, 84, 84, 3), dtype=np.uint8)
            }
            
            # Collect step
            isaac_collector.collect_step(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=np.random.random() < 0.02,  # 2% chance of episode end
                truncated=np.random.random() < 0.01,  # 1% chance of truncation
                info={"step": step}
            )
            
            time.sleep(0.01)  # Simulate processing time
        
        # Stop async collection
        isaac_collector.stop_async_collection()
        isaac_collector.end_collection()
        
        # Get statistics
        stats = isaac_collector.get_dataset_statistics()
        logger.info(f"Isaac Lab simulation completed. Statistics: {stats}")
        
        # Export for offline RL
        isaac_collector.export_for_offline_rl(self.save_path / "isaac_lab" / "offline_rl")
        logger.info("Exported Isaac Lab data for offline RL")
    
    def run_full_demo(self) -> None:
        """Run the complete demonstration."""
        logger.info("Starting AquaML Data Collection System Demo")
        
        try:
            # Demo 1: Basic Gymnasium collection
            self.demo_gymnasium_collection("CartPole-v1", num_episodes=5)
            
            # Demo 2: Vector environment collection
            self.demo_vector_environment_collection("CartPole-v1", num_envs=4, num_episodes=10)
            
            # Demo 3: Trajectory collection
            self.demo_trajectory_collection("CartPole-v1", num_episodes=8)
            
            # Demo 4: Isaac Lab integration concept
            self.demonstrate_isaac_lab_integration()
            
            logger.info("All demos completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed with error: {e}")
            raise


def main():
    """Main function to run the demo."""
    demo = CollectorDemo(save_path="./demo_collected_data")
    demo.run_full_demo()


if __name__ == "__main__":
    main() 