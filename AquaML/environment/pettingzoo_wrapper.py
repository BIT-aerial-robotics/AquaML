"""PettingZoo environment wrapper for multi-agent reinforcement learning"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
from loguru import logger

try:
    from pettingzoo import AECEnv, ParallelEnv
    from pettingzoo.utils import agent_selector
    HAS_PETTINGZOO = True
except ImportError:
    HAS_PETTINGZOO = False
    logger.warning("PettingZoo not installed. Please install with: pip install pettingzoo")

from AquaML.environment.base_env import BaseEnv
from AquaML import coordinator


@coordinator.registerEnvironment
class PettingZooWrapper(BaseEnv):
    """Wrapper for PettingZoo environments to integrate with AquaML framework
    
    This wrapper supports both AEC (turn-based) and Parallel (simultaneous) 
    PettingZoo environments, providing a unified interface for multi-agent RL.
    """
    
    def __init__(self, 
                 env_name: str,
                 env_kwargs: Optional[Dict[str, Any]] = None,
                 parallel: bool = True,
                 flatten_observations: bool = True,
                 **kwargs):
        """Initialize PettingZoo wrapper
        
        Args:
            env_name: Name of the PettingZoo environment
            env_kwargs: Additional arguments for environment creation
            parallel: Whether to use parallel (simultaneous) mode
            flatten_observations: Whether to flatten observation dictionaries
        """
        super().__init__()
        
        if not HAS_PETTINGZOO:
            raise ImportError("PettingZoo is required but not installed")
        
        self.env_name = env_name
        self.parallel = parallel
        self.flatten_observations = flatten_observations
        self.env_kwargs = env_kwargs or {}
        
        # Create environment
        self._create_environment()
        
        # Initialize agent and environment info
        self._setup_agents()
        self._setup_spaces()
        
        logger.info(f"PettingZoo wrapper initialized for {env_name} with {len(self.agents)} agents")
    
    def _create_environment(self):
        """Create the PettingZoo environment"""
        try:
            if self.parallel:
                # Import parallel environment
                exec(f"from pettingzoo.{self.env_name} import parallel_env")
                self._env = eval(f"parallel_env(**self.env_kwargs)")
            else:
                # Import AEC environment
                exec(f"from pettingzoo.{self.env_name} import env")
                self._env = eval(f"env(**self.env_kwargs)")
                
            self._original_env = self._env
        except ImportError as e:
            raise ImportError(f"Failed to import {self.env_name}: {e}")
    
    def _setup_agents(self):
        """Setup agent information"""
        self.agents = self._env.possible_agents
        self.num_agents = len(self.agents)
        self.agent_selector = agent_selector(self.agents) if not self.parallel else None
    
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Get spaces for first agent (assuming homogeneous agents)
        first_agent = self.agents[0]
        
        # Observation space
        obs_space = self._env.observation_space(first_agent)
        if self.flatten_observations and hasattr(obs_space, 'spaces'):
            # Handle Dict spaces by flattening
            self.single_observation_shape = sum(np.prod(space.shape) 
                                               for space in obs_space.spaces.values())
        else:
            self.single_observation_shape = obs_space.shape
        
        # Action space
        action_space = self._env.action_space(first_agent)
        self.single_action_shape = action_space.shape
        self.action_type = type(action_space).__name__
    
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment
        
        Returns:
            Tuple of (observations, info) where observations are dict mapping
            agent names to their observations
        """
        self._env.reset()
        
        if self.parallel:
            observations, infos = self._env.last()
            return self._process_observations(observations), infos
        else:
            # AEC mode - start with first agent
            self.agent_selector.reset()
            self.current_agent = self.agent_selector.next()
            observations = {self.current_agent: self._env.observe(self.current_agent)}
            infos = {self.current_agent: {}}
            return observations, infos
    
    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], 
                                                     Dict[str, bool], Dict[str, bool], 
                                                     Dict[str, Any]]:
        """Step the environment
        
        Args:
            actions: Dictionary mapping agent names to their actions
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        if self.parallel:
            return self._step_parallel(actions)
        else:
            return self._step_aec(actions)
    
    def _step_parallel(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], 
                                                              Dict[str, bool], Dict[str, bool], 
                                                              Dict[str, Any]]:
        """Step for parallel (simultaneous) environment"""
        # Unpack actions for parallel env
        action_dict = {agent: actions[agent] for agent in self.agents if agent in actions}
        
        # Step environment
        observations, rewards, terminated, truncated, infos = self._env.step(action_dict)
        
        # Process observations
        processed_obs = self._process_observations(observations)
        
        return processed_obs, rewards, terminated, truncated, infos
    
    def _step_aec(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], 
                                                         Dict[str, bool], Dict[str, bool], 
                                                         Dict[str, Any]]:
        """Step for AEC (turn-based) environment"""
        current_agent = list(actions.keys())[0]  # Get current agent
        action = actions[current_agent]
        
        # Step current agent
        self._env.step(action)
        
        # Check if episode is done
        terminated = self._env.terminations[current_agent]
        truncated = self._env.truncations[current_agent]
        
        # Get reward for current agent
        reward = self._env.rewards[current_agent]
        
        # Get next agent
        if not (terminated or truncated):
            self.current_agent = self.agent_selector.next()
        
        # Get observations for next agent
        observations = {self.current_agent: self._env.observe(self.current_agent)}
        rewards = {self.current_agent: reward}
        
        # Create terminated/truncated for all agents
        terminated_dict = {agent: self._env.terminations[agent] for agent in self.agents}
        truncated_dict = {agent: self._env.truncations[agent] for agent in self.agents}
        
        infos = {self.current_agent: {}}
        
        return observations, rewards, terminated_dict, truncated_dict, infos
    
    def _process_observations(self, observations: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process observations for AquaML format
        
        Args:
            observations: Raw observations from environment
            
        Returns:
            Processed observations in tensor format
        """
        processed = {}
        
        for agent, obs in observations.items():
            if isinstance(obs, dict) and self.flatten_observations:
                # Flatten Dict observations
                flattened = []
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        flattened.extend(value.flatten())
                    else:
                        flattened.append(value)
                processed[agent] = torch.tensor(flattened, dtype=torch.float32)
            elif isinstance(obs, np.ndarray):
                processed[agent] = torch.tensor(obs, dtype=torch.float32)
            else:
                processed[agent] = torch.tensor(obs, dtype=torch.float32)
        
        return processed
    
    def getEnvInfo(self) -> Dict[str, Any]:
        """Get environment information for AquaML integration
        
        Returns:
            Dictionary containing environment configuration
        """
        first_agent = self.agents[0]
        
        # Observation configuration
        obs_space = self._env.observation_space(first_agent)
        observation_cfg = {
            "state": {
                "shape": obs_space.shape if hasattr(obs_space, 'shape') else (self.single_observation_shape,),
                "single_shape": obs_space.shape if hasattr(obs_space, 'shape') else (self.single_observation_shape,),
                "dtype": np.float32,
                "num_observations": self.single_observation_shape
            }
        }
        
        # Action configuration
        action_space = self._env.action_space(first_agent)
        action_cfg = {
            "action": {
                "shape": action_space.shape,
                "single_shape": action_space.shape,
                "dtype": np.float32,
                "num_actions": np.prod(action_space.shape)
            }
        }
        
        return {
            "observation_cfg": observation_cfg,
            "action_cfg": action_cfg,
            "num_envs": 1,  # Single PettingZoo environment
            "max_episode_steps": getattr(self._env, 'max_cycles', 1000),
            "agents": self.agents,
            "num_agents": self.num_agents,
            "parallel": self.parallel
        }
    
    def render(self) -> None:
        """Render the environment"""
        if hasattr(self._env, 'render'):
            self._env.render()
    
    def close(self) -> None:
        """Close the environment"""
        if hasattr(self._env, 'close'):
            self._env.close()
    
    def seed(self, seed: int) -> None:
        """Set random seed"""
        if hasattr(self._env, 'seed'):
            self._env.seed(seed)
        
        # Also set numpy seed
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    @property
    def unwrapped(self):
        """Get the underlying PettingZoo environment"""
        return self._env
    
    def __str__(self) -> str:
        return f"PettingZooWrapper({self.env_name}, {len(self.agents)} agents, parallel={self.parallel})"