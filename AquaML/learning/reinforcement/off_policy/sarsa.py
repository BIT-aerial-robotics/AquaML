from typing import Dict, Any, Optional, Tuple, Union
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger

from AquaML import coordinator
from AquaML.learning.model import Model
from AquaML.learning.reinforcement.base import Agent
from AquaML.config import configclass


@configclass
class SARSACfg:
    """SARSA (State-Action-Reward-State-Action) configuration class following AquaML patterns"""
    
    # Core SARSA parameters
    discount_factor: float = 0.99  # Discount factor (gamma)
    learning_rate: float = 0.5  # Learning rate (alpha)
    
    # Exploration parameters
    random_timesteps: int = 0  # Random exploration steps
    learning_starts: int = 0  # Start learning after this many steps
    
    # Training parameters
    batch_size: int = 1  # SARSA is typically done with single samples
    
    # Preprocessing
    state_preprocessor: Optional[Any] = None  # State preprocessor
    state_preprocessor_kwargs: Optional[Dict[str, Any]] = None  # State preprocessor kwargs
    rewards_shaper: Optional[Any] = None  # Reward shaping function
    
    # Device and precision
    device: str = "auto"  # Device to use
    mixed_precision: bool = False  # Mixed precision training (typically not used for SARSA)
    
    # Memory parameters (for compatibility, though traditional SARSA uses Q-tables)
    memory_size: int = 10000  # Size of experience buffer if using function approximation


class SARSAMemory:
    """Simple memory buffer for SARSA transitions"""
    
    def __init__(self, memory_size: int, device: str = "cpu"):
        self.memory_size = memory_size
        self.device = device
        self.position = 0
        self.full = False
        self.tensors = {}
        
    def create_tensor(self, name: str, size: Union[int, Tuple[int]], dtype: torch.dtype = torch.float32):
        """Create a tensor in memory"""
        if isinstance(size, int):
            size = (size,)
        self.tensors[name] = torch.zeros(
            (self.memory_size,) + size, dtype=dtype, device=self.device
        )
    
    def add_samples(self, **kwargs):
        """Add samples to memory"""
        for key, value in kwargs.items():
            if key in self.tensors:
                if isinstance(value, dict):
                    # Flatten dictionary for storage
                    flattened = self._flatten_dict(value)
                    value = flattened
                
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    # Dynamic resize if needed
                    if self.tensors[key].shape[1:] != value.shape[1:]:
                        new_shape = (self.memory_size,) + value.shape[1:]
                        self.tensors[key] = torch.zeros(
                            new_shape, dtype=self.tensors[key].dtype, device=self.device
                        )
                        logger.debug(f"Resized tensor {key} to shape {new_shape}")
                    self.tensors[key][self.position] = value
                else:
                    self.tensors[key][self.position] = torch.tensor(value, device=self.device)
        
        self.position = (self.position + 1) % self.memory_size
        if self.position == 0:
            self.full = True
    
    def _flatten_dict(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten dictionary for storage"""
        tensors = []
        for key in sorted(data_dict.keys()):
            tensor = data_dict[key]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor, device=self.device)
            
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            
            tensors.append(tensor)
        
        if tensors:
            return torch.cat(tensors, dim=-1)
        else:
            return torch.tensor([], device=self.device)
    
    def get_last_sample(self, names: list):
        """Get the most recent sample (for SARSA updates)"""
        if self.position == 0 and not self.full:
            return None
        
        idx = (self.position - 1) % self.memory_size
        
        batch_data = []
        for name in names:
            if name in self.tensors:
                batch_data.append(self.tensors[name][idx])
            else:
                batch_data.append(torch.tensor([], device=self.device))
        
        return batch_data
    
    def get_stored_samples_count(self) -> int:
        """Get number of stored samples"""
        return self.memory_size if self.full else self.position
    
    def is_ready_for_update(self, min_samples: int) -> bool:
        """Check if memory has enough samples"""
        return self.get_stored_samples_count() >= min_samples


class SARSA(Agent):
    """SARSA (State-Action-Reward-State-Action) algorithm implementation for AquaML
    
    SARSA is an on-policy temporal difference learning algorithm that learns Q-values
    by using the actual next action taken by the policy (unlike Q-Learning which uses
    the maximum Q-value).
    
    Key features:
    - On-policy learning: uses the policy's actual next action for updates
    - Supports both discrete tabular SARSA and function approximation
    - Compatible with AquaML's dictionary-based state/action interface
    - Follows AquaML patterns for configuration and model management
    
    The SARSA update rule is:
    Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
    where a' is the actual next action taken by the policy
    """
    
    @coordinator.registerAgent
    def __init__(
        self,
        models: Dict[str, Model],
        cfg: SARSACfg,
        observation_space: Optional[Dict[str, Any]] = None,
        action_space: Optional[Dict[str, Any]] = None,
    ):
        """Initialize SARSA agent
        
        Args:
            models: Dictionary containing the policy model (Q-function)
            cfg: SARSA configuration
            observation_space: Observation space info
            action_space: Action space info
        """
        super().__init__(models, None, observation_space, action_space, cfg.device, cfg)
        
        self.cfg = cfg
        
        # Device setup
        if cfg.device == "auto":
            self.device = coordinator.get_device()
        else:
            self.device = cfg.device
        
        logger.info(f"SARSA initialized with device: {self.device}")
        
        # Models validation
        if "policy" not in models:
            raise ValueError("Policy model is required for SARSA")
        
        # Policy model (Q-function)
        self.policy = models["policy"]
        self.policy.to(self.device)
        
        # Configuration parameters
        self._discount_factor = cfg.discount_factor
        self._learning_rate = cfg.learning_rate
        self._random_timesteps = cfg.random_timesteps
        self._learning_starts = cfg.learning_starts
        self._mixed_precision = cfg.mixed_precision
        
        # Setup memory for experience storage
        self.memory = SARSAMemory(cfg.memory_size, self.device)
        
        # Mixed precision scaler
        try:
            self.scaler = torch.amp.GradScaler(device_type='cuda', enabled=self._mixed_precision)
        except TypeError:
            self.scaler = torch.amp.GradScaler(enabled=self._mixed_precision)
        
        # Preprocessors
        self._state_preprocessor = None
        self._rewards_shaper = cfg.rewards_shaper
        
        if cfg.state_preprocessor is not None:
            self._state_preprocessor = cfg.state_preprocessor(**(cfg.state_preprocessor_kwargs or {}))
            logger.info("State preprocessor initialized")
        
        # Optimizer for function approximation (if using neural networks)
        if hasattr(self.policy, 'parameters'):
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(), 
                lr=cfg.learning_rate
            )
        else:
            self.optimizer = None  # For tabular SARSA
        
        # Initialize memory tensors
        self._init_memory()
        
        # Store current and next transitions for SARSA update
        self._current_states = None
        self._current_actions = None
        self._current_rewards = None
        self._current_next_states = None
        self._current_next_actions = None  # Key difference from Q-Learning: we need the actual next action
        self._current_dones = None
        
        # Flag to track if we have a complete SARSA tuple (s,a,r,s',a')
        self._has_complete_tuple = False
        
        # Tensor names for sampling
        self._tensors_names = [
            "states", "actions", "rewards", "next_states", "next_actions", "terminated", "truncated"
        ]
        
        logger.info("SARSA agent initialized successfully")
    
    def _init_memory(self):
        """Initialize memory tensors"""
        # Create placeholder tensors (will be resized dynamically)
        self.memory.create_tensor("states", (1,), torch.float32)
        self.memory.create_tensor("actions", (1,), torch.long)  # Discrete actions
        self.memory.create_tensor("rewards", (1,), torch.float32)
        self.memory.create_tensor("next_states", (1,), torch.float32)
        self.memory.create_tensor("next_actions", (1,), torch.long)  # Next actions for SARSA
        self.memory.create_tensor("terminated", (1,), torch.bool)
        self.memory.create_tensor("truncated", (1,), torch.bool)
    
    def act(
        self, 
        states: Dict[str, torch.Tensor], 
        timestep: int, 
        timesteps: int
    ) -> Dict[str, torch.Tensor]:
        """Generate actions using the policy (typically epsilon-greedy)
        
        Args:
            states: Dictionary of environment states
            timestep: Current timestep
            timesteps: Total timesteps
            
        Returns:
            Dictionary containing actions and Q-values
        """
        # Convert states to proper device
        states = {k: v.to(self.device) for k, v in states.items()}
        
        # Apply state preprocessing if available
        if self._state_preprocessor is not None:
            processed_states = {}
            for k, v in states.items():
                processed_states[k] = self._state_preprocessor(v)
            states = processed_states
        
        # Random exploration during initial timesteps
        if timestep < self._random_timesteps:
            if hasattr(self.policy, 'random_act'):
                outputs = self.policy.random_act(states)
            else:
                # Fallback: generate random actions
                batch_size = list(states.values())[0].shape[0]
                if hasattr(self.policy, 'action_space_size'):
                    action_space_size = self.policy.action_space_size
                else:
                    action_space_size = 2  # Default binary actions
                
                random_actions = torch.randint(
                    0, action_space_size, (batch_size, 1), 
                    device=self.device, dtype=torch.long
                )
                outputs = {"actions": random_actions}
            
            logger.debug(f"Random exploration at timestep {timestep}")
            return outputs
        
        # Generate actions from policy (typically epsilon-greedy for SARSA)
        with torch.cuda.amp.autocast(enabled=self._mixed_precision):
            outputs = self.policy.act(states)
            
            # Ensure actions are in the correct format
            if "actions" not in outputs:
                # If policy returns Q-values, use policy's action selection (e.g., epsilon-greedy)
                if "q_values" in outputs:
                    q_values = outputs["q_values"]
                    # For SARSA, we should use the policy's action selection mechanism
                    # This could be epsilon-greedy, softmax, etc.
                    if hasattr(self.policy, 'select_action'):
                        actions = self.policy.select_action(q_values)
                    else:
                        # Default to greedy action selection if no selection method provided
                        actions = torch.argmax(q_values, dim=-1, keepdim=True)
                    outputs["actions"] = actions
                elif "values" in outputs:
                    # Assume values are Q-values for all actions
                    q_values = outputs["values"]
                    if hasattr(self.policy, 'select_action'):
                        actions = self.policy.select_action(q_values)
                    else:
                        actions = torch.argmax(q_values, dim=-1, keepdim=True)
                    outputs["actions"] = actions
        
        logger.debug(f"Generated action at timestep {timestep}")
        return outputs
    
    def record_transition(
        self,
        states: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
        next_states: Dict[str, torch.Tensor],
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ):
        """Record environment transition for SARSA update
        
        For SARSA, we need to wait for the next action to be taken before we can
        perform the update, as SARSA uses the actual next action (not the max).
        
        Args:
            states: Current states
            actions: Actions taken
            rewards: Rewards received
            next_states: Next states
            terminated: Episode termination flags
            truncated: Episode truncation flags
            infos: Additional info
            timestep: Current timestep
            timesteps: Total timesteps
        """
        # Convert to proper device
        states = {k: v.to(self.device) for k, v in states.items()}
        if isinstance(actions, dict):
            actions = {k: v.to(self.device) for k, v in actions.items()}
        else:
            actions = actions.to(self.device)
        next_states = {k: v.to(self.device) for k, v in next_states.items()}
        rewards = rewards.to(self.device)
        terminated = terminated.to(self.device)
        truncated = truncated.to(self.device)
        
        # Reward shaping
        if self._rewards_shaper is not None:
            rewards = self._rewards_shaper(rewards, timestep, timesteps)
        
        # If we have a previous complete tuple, perform SARSA update
        if self._has_complete_tuple and timestep >= self._learning_starts:
            # Get the next action for the current states (this becomes a' in SARSA)
            with torch.no_grad():
                next_action_outputs = self.act(next_states, timestep, timesteps)
                if isinstance(next_action_outputs, dict):
                    next_actions = next_action_outputs.get("actions", next_action_outputs.get("action"))
                else:
                    next_actions = next_action_outputs
            
            # Store the next action
            self._current_next_actions = next_actions
            
            # Set model to training mode for update
            self.policy.train()
            
            # Perform SARSA update with complete (s,a,r,s',a') tuple
            self._update(timestep, timesteps)
            
            # Set model back to eval mode
            self.policy.eval()
        
        # Store current transition for next update
        self._current_states = states
        self._current_actions = actions
        self._current_rewards = rewards
        self._current_next_states = next_states
        self._current_dones = terminated | truncated
        
        # Mark that we now have a transition ready for the next update
        self._has_complete_tuple = True
        
        # Handle episode termination: if episode ends, we can update with zero next Q-value
        if terminated.any() or truncated.any():
            self._current_next_actions = torch.zeros_like(
                actions["actions"] if isinstance(actions, dict) else actions
            )
            
            # Set model to training mode for update
            self.policy.train()
            
            # Perform final SARSA update for terminal state
            self._update(timestep, timesteps)
            
            # Set model back to eval mode
            self.policy.eval()
            
            # Reset for next episode
            self._has_complete_tuple = False
        
        # Store in memory for potential experience replay (if using function approximation)
        if self._current_next_actions is not None:
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                next_actions=self._current_next_actions,
                terminated=terminated,
                truncated=truncated,
            )
        
        logger.debug(f"Recorded transition at timestep {timestep}")
    
    def post_interaction(self, timestep: int, timesteps: int):
        """Called after each environment interaction"""
        # SARSA updates are handled in record_transition to ensure proper (s,a,r,s',a') tuples
        pass
    
    def _update(self, timestep: int, timesteps: int):
        """Main SARSA update step
        
        Implements the SARSA update rule:
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        where a' is the actual next action taken by the policy
        """
        if (self._current_states is None or 
            self._current_actions is None or 
            self._current_next_actions is None):
            return
        
        with torch.cuda.amp.autocast(enabled=self._mixed_precision):
            # Check if using tabular SARSA or function approximation
            if hasattr(self.policy, 'table') and callable(getattr(self.policy, 'table')):
                # Tabular SARSA update
                self._update_tabular()
            else:
                # Function approximation SARSA update
                self._update_function_approximation()
    
    def _update_tabular(self):
        """Update Q-table for tabular SARSA"""
        q_table = self.policy.table()
        
        # Get current states, actions, rewards, next states, and next actions
        states = self._current_states
        actions = self._current_actions
        rewards = self._current_rewards
        next_states = self._current_next_states
        next_actions = self._current_next_actions
        dones = self._current_dones
        
        # Convert states and actions to indices if needed
        if isinstance(states, dict):
            # Assume single state key for simplicity
            state_key = list(states.keys())[0]
            current_state_indices = states[state_key].long().squeeze()
            next_state_indices = next_states[state_key].long().squeeze()
        else:
            current_state_indices = states.long().squeeze()
            next_state_indices = next_states.long().squeeze()
        
        if isinstance(actions, dict):
            action_indices = actions["actions"].long().squeeze()
        else:
            action_indices = actions.long().squeeze()
        
        if isinstance(next_actions, dict):
            next_action_indices = next_actions["actions"].long().squeeze()
        else:
            next_action_indices = next_actions.long().squeeze()
        
        # Handle batch dimension
        if current_state_indices.dim() == 0:
            current_state_indices = current_state_indices.unsqueeze(0)
            next_state_indices = next_state_indices.unsqueeze(0)
            action_indices = action_indices.unsqueeze(0)
            next_action_indices = next_action_indices.unsqueeze(0)
            rewards = rewards.unsqueeze(0) if rewards.dim() == 0 else rewards
            dones = dones.unsqueeze(0) if dones.dim() == 0 else dones
        
        # Environment IDs for multi-environment support
        env_ids = torch.arange(rewards.shape[0], device=self.device)
        
        # Get current Q-values Q(s,a)
        current_q_values = q_table[env_ids, current_state_indices, action_indices]
        
        # Get next Q-values Q(s',a') - KEY DIFFERENCE FROM Q-LEARNING
        # SARSA uses the actual next action, not the max
        next_q_values = q_table[env_ids, next_state_indices, next_action_indices]
        
        # Compute target Q-values using SARSA update rule
        target_q_values = rewards + self._discount_factor * (1 - dones.float()) * next_q_values
        
        # Update Q-table using SARSA update rule
        td_error = target_q_values - current_q_values
        q_table[env_ids, current_state_indices, action_indices] += self._learning_rate * td_error
        
        # Track data
        self.track_data("Training/TD_Error", torch.abs(td_error).mean().item())
        self.track_data("Training/Q_Value_Mean", current_q_values.mean().item())
        self.track_data("Training/Next_Q_Value_Mean", next_q_values.mean().item())
        self.track_data("Training/Reward_Mean", rewards.mean().item())
        
        logger.debug(f"Tabular SARSA update - TD Error: {torch.abs(td_error).mean().item():.6f}")
    
    def _update_function_approximation(self):
        """Update Q-function using neural network approximation"""
        states = self._current_states
        actions = self._current_actions
        rewards = self._current_rewards
        next_states = self._current_next_states
        next_actions = self._current_next_actions
        dones = self._current_dones
        
        # Compute current Q-values Q(s,a)
        q_outputs = self.policy.act(states)
        
        if "q_values" in q_outputs:
            # Q-values for all actions
            all_q_values = q_outputs["q_values"]
            if isinstance(actions, dict):
                action_indices = actions["actions"].long()
            else:
                action_indices = actions.long()
            current_q_values = all_q_values.gather(1, action_indices)
        elif "values" in q_outputs:
            # Assume values represent Q-value for the specific action
            current_q_values = q_outputs["values"]
        else:
            logger.error("Policy output must contain 'q_values' or 'values'")
            return
        
        # Compute next Q-values Q(s',a') - KEY DIFFERENCE FROM Q-LEARNING
        # SARSA uses the actual next action, not the max
        with torch.no_grad():
            next_q_outputs = self.policy.act(next_states)
            if "q_values" in next_q_outputs:
                next_all_q_values = next_q_outputs["q_values"]
                if isinstance(next_actions, dict):
                    next_action_indices = next_actions["actions"].long()
                else:
                    next_action_indices = next_actions.long()
                next_q_values = next_all_q_values.gather(1, next_action_indices)
            elif "values" in next_q_outputs:
                next_q_values = next_q_outputs["values"]
            else:
                logger.error("Policy output must contain 'q_values' or 'values'")
                return
            
            # Compute target Q-values using SARSA rule
            target_q_values = rewards.unsqueeze(-1) + self._discount_factor * (1 - dones.float().unsqueeze(-1)) * next_q_values
        
        # Compute loss (Mean Squared Error)
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        # Track data
        td_error = torch.abs(target_q_values - current_q_values).mean()
        self.track_data("Loss/SARSA", loss.item())
        self.track_data("Training/TD_Error", td_error.item())
        self.track_data("Training/Q_Value_Mean", current_q_values.mean().item())
        self.track_data("Training/Next_Q_Value_Mean", next_q_values.mean().item())
        self.track_data("Training/Target_Q_Value_Mean", target_q_values.mean().item())
        self.track_data("Training/Reward_Mean", rewards.mean().item())
        
        logger.debug(f"Function approximation SARSA update - Loss: {loss.item():.6f}, TD Error: {td_error.item():.6f}")
    
    def save(self, path: str):
        """Save model parameters"""
        modules = {
            "policy": self.policy.state_dict(),
            "cfg": self.cfg,
        }
        
        if self.optimizer is not None:
            modules["optimizer"] = self.optimizer.state_dict()
        
        torch.save(modules, path)
        logger.info(f"SARSA model saved to {path}")
    
    def load(self, path: str):
        """Load model parameters"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint["policy"])
        
        if self.optimizer is not None and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        logger.info(f"SARSA model loaded from {path}")