"""Base trainer class for AquaML framework

This module provides the base trainer class that integrates with the AquaML coordinator
and supports dictionary-based actions, rewards, and states.
"""

from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
import sys
import atexit
from loguru import logger
import torch
from dataclasses import dataclass

from AquaML import coordinator
from AquaML.environment.base_env import BaseEnv
from AquaML.learning.reinforcement.base import Agent
from AquaML.data.base_unit import BaseUnit


@dataclass
class TrainerConfig:
    """Base configuration for trainers"""
    timesteps: int = 100000
    headless: bool = False
    disable_progressbar: bool = False
    close_environment_at_exit: bool = True
    environment_info: str = "episode"
    stochastic_evaluation: bool = False
    device: str = "auto"
    checkpoint_interval: int = 1000
    tensorboard: bool = False


class BaseTrainer(ABC):
    """Base trainer class for AquaML framework
    
    This class provides the foundation for all trainers in AquaML, integrating
    with the coordinator system and supporting dictionary-based environments.
    """
    
    def __init__(self, 
                 env: BaseEnv,
                 agents: Union[Agent, List[Agent]],
                 cfg: TrainerConfig):
        """Initialize the base trainer
        
        Args:
            env: AquaML environment instance
            agents: Agent or list of agents to train
            cfg: Trainer configuration
        """
        self.cfg = cfg
        self.env = env
        self.agents = agents if isinstance(agents, list) else [agents]
        
        # Device setup
        if cfg.device == "auto":
            self.device = coordinator.get_device()
        else:
            self.device = cfg.device
            
        # Get environment configuration
        self.env_info = env.getEnvInfo()
        self.observation_cfg = self.env_info['observation_cfg']
        self.action_cfg = self.env_info['action_cfg']
        self.num_envs = self.env_info['num_envs']
        
        # Setup data units based on environment configuration
        self._setup_data_units()
        
        # Register environment and agents with coordinator
        self._register_components()
        
        # Configuration parameters
        self.timesteps = cfg.timesteps
        self.headless = cfg.headless
        self.disable_progressbar = cfg.disable_progressbar
        self.close_environment_at_exit = cfg.close_environment_at_exit
        self.environment_info = cfg.environment_info
        self.stochastic_evaluation = cfg.stochastic_evaluation
        
        self.initial_timestep = 0
        self.num_agents = len(self.agents)
        
        # Register environment closing
        if self.close_environment_at_exit:
            @atexit.register
            def close_env():
                logger.info("Closing environment")
                try:
                    self.env.close()
                except:
                    pass
                logger.info("Environment closed")
        
        logger.info(f"Base trainer initialized with {self.num_agents} agents on device: {self.device}")
        
    def _setup_data_units(self):
        """Setup data units based on environment configuration"""
        from AquaML.data import unitCfg
        from AquaML.data.numpy_unit import NumpyUnit
        import numpy as np
        
        # Create data units for observations
        self.observation_units = {}
        for obs_name, obs_cfg in self.observation_cfg.items():
            # Create unit config for observation
            unit_cfg = unitCfg(
                name=obs_cfg['name'],
                dtype=obs_cfg['dtype'],
                single_shape=obs_cfg['single_shape'],
                size=self.num_envs
            )
            
            # Create and register data unit
            data_unit = NumpyUnit(unit_cfg)
            data_unit.createData()
            
            self.observation_units[obs_name] = data_unit
            
        # Create data units for actions  
        self.action_units = {}
        for act_name, act_cfg in self.action_cfg.items():
            # Create unit config for action
            unit_cfg = unitCfg(
                name=act_cfg['name'],
                dtype=act_cfg['dtype'],
                single_shape=act_cfg['single_shape'],
                size=self.num_envs
            )
            
            data_unit = NumpyUnit(unit_cfg)
            data_unit.createData()
            
            self.action_units[act_name] = data_unit
        
        # Create data units for rewards
        reward_cfg = unitCfg(
            name='reward',
            dtype=np.float32,
            single_shape=(1,),
            size=self.num_envs
        )
        
        self.reward_unit = NumpyUnit(reward_cfg)
        self.reward_unit.createData()
        
        # Create data units for done flags
        done_cfg = unitCfg(
            name='done',
            dtype=np.bool_,
            single_shape=(1,),
            size=self.num_envs
        )
        
        self.done_unit = NumpyUnit(done_cfg)
        self.done_unit.createData()
        
        # Create data units for truncated flags
        truncated_cfg = unitCfg(
            name='truncated',
            dtype=np.bool_,
            single_shape=(1,),
            size=self.num_envs
        )
        
        self.truncated_unit = NumpyUnit(truncated_cfg)
        self.truncated_unit.createData()
        
        logger.info("Data units created successfully")
        
    def _register_components(self):
        """Register environment and agents with coordinator"""
        # Register environment
        coordinator.registerEnv(self.env.__class__)
        
        # Register agents
        for agent in self.agents:
            coordinator.registerAgent(agent.__class__)
            
        logger.info("Components registered with coordinator")
        
    def _convert_to_tensors(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert dictionary data to PyTorch tensors
        
        Args:
            data_dict: Dictionary of data (numpy arrays or tensors)
            
        Returns:
            Dictionary of PyTorch tensors
        """
        tensor_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                tensor_dict[key] = value.to(self.device)
            else:
                tensor_dict[key] = torch.tensor(value, device=self.device)
        return tensor_dict
        
    def _convert_to_numpy(self, data_dict) -> Dict[str, Any]:
        """Convert dictionary of tensors to numpy arrays
        
        Args:
            data_dict: Dictionary of PyTorch tensors or single tensor
            
        Returns:
            Dictionary of numpy arrays
        """
        if isinstance(data_dict, torch.Tensor):
            # Handle single tensor case
            return {"actions": data_dict.cpu().numpy()}
        elif isinstance(data_dict, dict):
            # Handle dictionary case
            numpy_dict = {}
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    numpy_dict[key] = value.cpu().numpy()
                else:
                    numpy_dict[key] = value
            return numpy_dict
        else:
            # Handle other cases
            return {"actions": data_dict}
        
    def _update_data_units(self, 
                          observations: Dict[str, Any], 
                          actions: Dict[str, Any], 
                          rewards: Any,
                          dones: Any,
                          truncated: Any):
        """Update data units with new data
        
        Args:
            observations: Dictionary of observations
            actions: Dictionary of actions
            rewards: Reward values
            dones: Done flags
            truncated: Truncated flags
        """
        # Update observation units
        for obs_name, obs_data in observations.items():
            if obs_name in self.observation_units:
                self.observation_units[obs_name].data_[:] = obs_data
                
        # Update action units
        for act_name, act_data in actions.items():
            if act_name in self.action_units:
                self.action_units[act_name].data_[:] = act_data
                
        # Update reward, done, and truncated units
        # Handle rewards - extract scalar value if it's a dictionary
        if isinstance(rewards, dict):
            reward_values = rewards.get('reward', list(rewards.values())[0])
        else:
            reward_values = rewards
        
        self.reward_unit.data_[:] = reward_values
        self.done_unit.data_[:] = dones
        self.truncated_unit.data_[:] = truncated
        
    def get_status(self) -> Dict[str, Any]:
        """Get trainer status
        
        Returns:
            Status dictionary
        """
        return {
            "timesteps": self.timesteps,
            "num_agents": self.num_agents,
            "num_envs": self.num_envs,
            "device": self.device,
            "environment_info": self.env_info,
            "observation_units": list(self.observation_units.keys()),
            "action_units": list(self.action_units.keys()),
        }
        
    @abstractmethod
    def train(self) -> None:
        """Train the agents (to be implemented by subclasses)"""
        raise NotImplementedError("Child class must implement train method")
        
    @abstractmethod
    def eval(self) -> None:
        """Evaluate the agents (to be implemented by subclasses)"""
        raise NotImplementedError("Child class must implement eval method")
        
    def __str__(self) -> str:
        """String representation of the trainer"""
        string = f"Trainer: {self.__class__.__name__}"
        string += f"\n  |-- Number of environments: {self.num_envs}"
        string += f"\n  |-- Number of agents: {self.num_agents}"
        string += f"\n  |-- Device: {self.device}"
        string += f"\n  |-- Timesteps: {self.timesteps}"
        string += "\n  |-- Observation spaces:"
        for obs_name, obs_cfg in self.observation_cfg.items():
            string += f"\n  |     |-- {obs_name}: {obs_cfg['single_shape']}"
        string += "\n  |-- Action spaces:"
        for act_name, act_cfg in self.action_cfg.items():
            string += f"\n  |     |-- {act_name}: {act_cfg['single_shape']}"
        return string