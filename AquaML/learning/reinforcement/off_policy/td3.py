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
class TD3Cfg:
    """Twin Delayed DDPG (TD3) configuration class following AquaML patterns"""
    
    # Training parameters
    gradient_steps: int = 1  # gradient steps per environment step
    batch_size: int = 64  # training batch size
    
    # Discount factor and soft update
    discount_factor: float = 0.99  # discount factor (gamma)
    polyak: float = 0.005  # soft update coefficient (tau)
    
    # Learning rates
    actor_learning_rate: float = 1e-3  # actor learning rate
    critic_learning_rate: float = 1e-3  # critic learning rate
    learning_rate_scheduler: Optional[Any] = None  # learning rate scheduler class
    learning_rate_scheduler_kwargs: Optional[Dict[str, Any]] = None  # scheduler kwargs
    
    # Preprocessors
    state_preprocessor: Optional[Any] = None  # state preprocessor class
    state_preprocessor_kwargs: Optional[Dict[str, Any]] = None  # preprocessor kwargs
    
    # Exploration
    random_timesteps: int = 0  # random exploration steps
    learning_starts: int = 0  # learning starts after this many steps
    
    # Gradient clipping
    grad_norm_clip: float = 0  # gradient clipping coefficient
    
    # Exploration noise configuration
    exploration_noise: Optional[Any] = None  # exploration noise class
    exploration_initial_scale: float = 1.0  # initial noise scale
    exploration_final_scale: float = 1e-3  # final noise scale
    exploration_timesteps: Optional[int] = None  # timesteps for noise decay
    
    # TD3-specific parameters
    policy_delay: int = 2  # policy delay update with respect to critic update
    smooth_regularization_noise: Optional[Any] = None  # smooth noise for target policy smoothing
    smooth_regularization_clip: float = 0.5  # clip for smooth regularization
    
    # Rewards shaping
    rewards_shaper: Optional[Any] = None  # rewards shaping function
    
    # Mixed precision training
    mixed_precision: bool = False  # enable automatic mixed precision
    
    # Memory and device
    memory_size: int = 10000  # replay buffer size
    device: str = "auto"  # device to use for training


class TD3Memory:
    """Replay buffer for TD3 algorithm"""
    
    def __init__(self, memory_size: int, device: str = "cpu"):
        self.memory_size = memory_size
        self.device = device
        self.position = 0
        self.full = False
        self.tensors = {}
        self.data_structure_info = {}
    
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
                    # Store structure info for dictionaries
                    if key not in self.data_structure_info:
                        self.data_structure_info[key] = {
                            "type": "dict",
                            "keys": list(value.keys()),
                            "shapes": {k: v.shape for k, v in value.items() if isinstance(v, torch.Tensor)}
                        }
                    # Flatten the dictionary for storage
                    flattened = self._flatten_dict(value)
                    value = flattened
                
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    # Dynamic resize tensor if needed
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
        """Flatten dictionary to tensor"""
        tensors = []
        for key in sorted(data_dict.keys()):
            tensor = data_dict[key]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor, device=self.device)
            
            # Ensure proper dimensions for concatenation
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() > 2:
                tensor = tensor.flatten(start_dim=1)
            
            tensors.append(tensor)
        
        if tensors:
            return torch.cat(tensors, dim=-1)
        else:
            return torch.tensor([], device=self.device)
    
    def _unflatten_dict(self, flattened_tensor: torch.Tensor, structure_name: str) -> Dict[str, torch.Tensor]:
        """Reconstruct dictionary from flattened tensor"""
        if structure_name not in self.data_structure_info:
            return {"data": flattened_tensor}
        
        structure_info = self.data_structure_info[structure_name]
        if structure_info["type"] != "dict":
            return {"data": flattened_tensor}
        
        result = {}
        start_idx = 0
        
        for key in structure_info["keys"]:
            if key in structure_info["shapes"]:
                shape = structure_info["shapes"][key]
                size = np.prod(shape[1:]) if len(shape) > 1 else 1
                end_idx = start_idx + size
                
                # Extract and reshape
                tensor_data = flattened_tensor[..., start_idx:end_idx]
                if len(shape) > 1:
                    tensor_data = tensor_data.reshape(tensor_data.shape[:-1] + shape[1:])
                
                result[key] = tensor_data
                start_idx = end_idx
        
        return result
    
    def sample(self, names: list, batch_size: int):
        """Sample a batch from memory"""
        total_samples = self.memory_size if self.full else self.position
        
        if total_samples < batch_size:
            batch_size = total_samples
        
        indices = torch.randint(0, total_samples, (batch_size,), device=self.device)
        
        batch_data = []
        for name in names:
            if name in self.tensors:
                batch_data.append(self.tensors[name][indices])
            else:
                batch_data.append(torch.tensor([], device=self.device))
        
        return batch_data
    
    def get_stored_samples_count(self) -> int:
        """Get number of stored samples"""
        return self.memory_size if self.full else self.position


class TD3(Agent):
    """Twin Delayed DDPG (TD3) algorithm implementation for AquaML
    
    Based on: https://arxiv.org/abs/1802.09477
    
    TD3 improves upon DDPG with three key improvements:
    1. Twin Critic Networks (Clipped Double Q-Learning): Uses minimum of two critic networks to reduce overestimation bias
    2. Delayed Policy Updates: Updates policy less frequently than critics to reduce variance
    3. Target Policy Smoothing: Adds noise to target actions to regularize targets
    
    This implementation follows the AquaML architecture patterns and integrates
    with the AquaML ecosystem for device management, file systems, and logging.
    """
    
    @coordinator.registerAgent
    def __init__(
        self,
        models: Dict[str, Model],
        cfg: TD3Cfg,
        observation_space: Optional[Dict[str, Any]] = None,
        action_space: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize TD3 agent
        
        Args:
            models: Dictionary containing policy, target_policy, critic_1, critic_2, target_critic_1, target_critic_2 models
            cfg: TD3 configuration
            observation_space: Dictionary describing observation space
            action_space: Dictionary describing action space
        """
        super().__init__(models, None, observation_space, action_space, cfg.device, cfg)
        
        self.cfg = cfg
        
        # Device setup
        if cfg.device == "auto":
            self.device = coordinator.get_device()
        else:
            self.device = cfg.device
        
        logger.info(f"TD3 initialized with device: {self.device}")
        
        # Models validation
        required_models = ["policy", "target_policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2"]
        for model_name in required_models:
            if model_name not in models:
                raise ValueError(f"{model_name} model is required for TD3")
        
        # Models
        self.policy = models["policy"]
        self.target_policy = models["target_policy"]
        self.critic_1 = models["critic_1"]
        self.critic_2 = models["critic_2"]
        self.target_critic_1 = models["target_critic_1"]
        self.target_critic_2 = models["target_critic_2"]
        
        # Move models to device
        self.policy.to(self.device)
        self.target_policy.to(self.device)
        self.critic_1.to(self.device)
        self.critic_2.to(self.device)
        self.target_critic_1.to(self.device)
        self.target_critic_2.to(self.device)
        
        # Freeze target networks (they are updated via soft updates)
        for param in self.target_policy.parameters():
            param.requires_grad = False
        for param in self.target_critic_1.parameters():
            param.requires_grad = False
        for param in self.target_critic_2.parameters():
            param.requires_grad = False
        
        # Initialize target networks (hard update)
        self._hard_update(self.target_policy, self.policy)
        self._hard_update(self.target_critic_1, self.critic_1)
        self._hard_update(self.target_critic_2, self.critic_2)
        
        # Setup memory
        self.memory = TD3Memory(cfg.memory_size, self.device)
        
        # Configuration parameters
        self._gradient_steps = cfg.gradient_steps
        self._batch_size = cfg.batch_size
        self._discount_factor = cfg.discount_factor
        self._polyak = cfg.polyak
        
        self._actor_learning_rate = cfg.actor_learning_rate
        self._critic_learning_rate = cfg.critic_learning_rate
        self._learning_rate_scheduler = cfg.learning_rate_scheduler
        
        self._random_timesteps = cfg.random_timesteps
        self._learning_starts = cfg.learning_starts
        self._grad_norm_clip = cfg.grad_norm_clip
        
        self._exploration_noise = cfg.exploration_noise
        self._exploration_initial_scale = cfg.exploration_initial_scale
        self._exploration_final_scale = cfg.exploration_final_scale
        self._exploration_timesteps = cfg.exploration_timesteps
        
        # TD3-specific parameters
        self._policy_delay = cfg.policy_delay
        self._critic_update_counter = 0  # Track number of critic updates for delayed policy updates
        
        self._smooth_regularization_noise = cfg.smooth_regularization_noise
        self._smooth_regularization_clip = cfg.smooth_regularization_clip
        if self._smooth_regularization_noise is None:
            logger.warning("TD3: No smooth regularization noise specified - target policy smoothing disabled")
        
        self._rewards_shaper = cfg.rewards_shaper
        self._mixed_precision = cfg.mixed_precision
        
        # Mixed precision scaler
        try:
            self.scaler = torch.amp.GradScaler(device_type='cuda', enabled=self._mixed_precision)
        except TypeError:
            self.scaler = torch.amp.GradScaler(enabled=self._mixed_precision)
        
        # Setup preprocessors
        self._state_preprocessor = None
        if cfg.state_preprocessor is not None:
            self._state_preprocessor = cfg.state_preprocessor(**(cfg.state_preprocessor_kwargs or {}))
            logger.info("State preprocessor initialized")
        
        # Setup optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self._actor_learning_rate
        )
        # Use itertools.chain to combine parameters from both critics
        import itertools
        self.critic_optimizer = torch.optim.Adam(
            itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), 
            lr=self._critic_learning_rate
        )
        
        # Setup learning rate schedulers
        if self._learning_rate_scheduler is not None:
            self.policy_scheduler = self._learning_rate_scheduler(
                self.policy_optimizer, **cfg.learning_rate_scheduler_kwargs or {}
            )
            self.critic_scheduler = self._learning_rate_scheduler(
                self.critic_optimizer, **cfg.learning_rate_scheduler_kwargs or {}
            )
        else:
            self.policy_scheduler = None
            self.critic_scheduler = None
        
        # Initialize memory tensors
        self._init_memory()
        
        # Action space bounds for clipping
        self.clip_actions_min = None
        self.clip_actions_max = None
        
        # Register checkpoint modules
        self.register_checkpoint_module("policy", self.policy)
        self.register_checkpoint_module("target_policy", self.target_policy)
        self.register_checkpoint_module("critic_1", self.critic_1)
        self.register_checkpoint_module("critic_2", self.critic_2)
        self.register_checkpoint_module("target_critic_1", self.target_critic_1)
        self.register_checkpoint_module("target_critic_2", self.target_critic_2)
        self.register_checkpoint_module("policy_optimizer", self.policy_optimizer)
        self.register_checkpoint_module("critic_optimizer", self.critic_optimizer)
        if self.policy_scheduler is not None:
            self.register_checkpoint_module("policy_scheduler", self.policy_scheduler)
        if self.critic_scheduler is not None:
            self.register_checkpoint_module("critic_scheduler", self.critic_scheduler)
        
        logger.info("TD3 agent initialized successfully")
    
    def _init_memory(self):
        """Initialize memory tensors"""
        # Create placeholder tensors that will be resized dynamically
        self.memory.create_tensor("states", (1,), torch.float32)
        self.memory.create_tensor("next_states", (1,), torch.float32)
        self.memory.create_tensor("actions", (1,), torch.float32)
        self.memory.create_tensor("rewards", (1,), torch.float32)
        self.memory.create_tensor("terminated", (1,), torch.bool)
        self.memory.create_tensor("truncated", (1,), torch.bool)
        
        self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated", "truncated"]
    
    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg)
        self.set_running_mode("eval")
        
        # Set action space bounds for clipping
        if self.action_space is not None:
            if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
                self.clip_actions_min = torch.tensor(
                    self.action_space.low, device=self.device, dtype=torch.float32
                )
                self.clip_actions_max = torch.tensor(
                    self.action_space.high, device=self.device, dtype=torch.float32
                )
                logger.info(f"Action bounds: [{self.clip_actions_min}, {self.clip_actions_max}]")
    
    def _hard_update(self, target: nn.Module, source: nn.Module):
        """Hard update target network"""
        target.load_state_dict(source.state_dict())
    
    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """Soft update target network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
    
    def act(
        self, states: Dict[str, torch.Tensor], timestep: int, timesteps: int
    ) -> Dict[str, torch.Tensor]:
        """Generate actions from policy
        
        Args:
            states: Dictionary of environment states
            timestep: Current timestep
            timesteps: Total timesteps
            
        Returns:
            Dictionary containing actions and other outputs
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
            # Generate random actions within action space bounds
            if self.clip_actions_min is not None and self.clip_actions_max is not None:
                batch_size = list(states.values())[0].shape[0]
                random_actions = torch.rand(
                    (batch_size,) + self.clip_actions_min.shape, device=self.device
                ) * (self.clip_actions_max - self.clip_actions_min) + self.clip_actions_min
                return {"actions": random_actions}
            else:
                # Fallback to policy for random actions
                logger.warning("No action bounds defined, using policy for random actions")
        
        # Get deterministic actions from policy
        with torch.autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cpu', enabled=self._mixed_precision):
            outputs = self.policy.act(states)
        
        actions = outputs.get("actions", outputs.get("mean_actions"))
        if actions is None:
            raise ValueError("Policy must return 'actions' or 'mean_actions' in output")
        
        # Add exploration noise
        if self._exploration_noise is not None and timestep >= self._random_timesteps:
            # Sample noise
            noise = self._exploration_noise.sample(actions.shape)
            
            # Calculate noise scale with decay
            scale = self._exploration_final_scale
            if self._exploration_timesteps is None:
                self._exploration_timesteps = timesteps
            
            if timestep <= self._exploration_timesteps:
                scale = (1 - timestep / self._exploration_timesteps) * (
                    self._exploration_initial_scale - self._exploration_final_scale
                ) + self._exploration_final_scale
            
            # Apply noise
            noise = noise * scale
            actions = actions + noise
            
            # Clip actions to bounds
            if self.clip_actions_min is not None and self.clip_actions_max is not None:
                actions = torch.clamp(actions, self.clip_actions_min, self.clip_actions_max)
            
            # Track exploration noise
            self.track_data("Exploration/Noise_Scale", scale)
            self.track_data("Exploration/Noise_Max", torch.max(noise).item())
            self.track_data("Exploration/Noise_Min", torch.min(noise).item())
            self.track_data("Exploration/Noise_Mean", torch.mean(noise).item())
        
        outputs["actions"] = actions
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
        """Record environment transition
        
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
        
        # Store transition in memory
        self.memory.add_samples(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
        )
        
        logger.debug(f"Recorded transition at timestep {timestep}")
    
    def post_interaction(self, timestep: int, timesteps: int):
        """Called after each environment interaction"""
        if timestep >= self._learning_starts and self.memory.get_stored_samples_count() >= self._batch_size:
            self.set_running_mode("train")
            self._update(timestep, timesteps)
            self.set_running_mode("eval")
    
    def _update(self, timestep: int, timesteps: int):
        """Algorithm's main update step implementing TD3's three key improvements"""
        for gradient_step in range(self._gradient_steps):
            # Sample a batch from memory
            batch = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)
            (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = batch
            
            with torch.autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cpu', enabled=self._mixed_precision):
                # Reconstruct dictionaries from flattened tensors
                states_dict = self.memory._unflatten_dict(sampled_states, "states")
                next_states_dict = self.memory._unflatten_dict(sampled_next_states, "next_states")
                actions_dict = self.memory._unflatten_dict(sampled_actions, "actions")
                
                # Apply preprocessing
                if self._state_preprocessor is not None:
                    states_dict = {k: self._state_preprocessor(v) for k, v in states_dict.items()}
                    next_states_dict = {k: self._state_preprocessor(v) for k, v in next_states_dict.items()}
                
                # Compute target Q-values with TD3's improvements
                with torch.no_grad():
                    # Get target actions from target policy
                    next_actions_outputs = self.target_policy.act(next_states_dict)
                    next_actions = next_actions_outputs.get("actions", next_actions_outputs.get("mean_actions"))
                    
                    # TD3 Improvement 3: Target Policy Smoothing
                    # Add clipped noise to target actions to regularize targets
                    if self._smooth_regularization_noise is not None:
                        noise = self._smooth_regularization_noise.sample(next_actions.shape)
                        noise = torch.clamp(
                            noise,
                            -self._smooth_regularization_clip,
                            self._smooth_regularization_clip
                        )
                        next_actions = next_actions + noise
                        
                        # Clip smoothed actions to action bounds
                        if self.clip_actions_min is not None and self.clip_actions_max is not None:
                            next_actions = torch.clamp(next_actions, self.clip_actions_min, self.clip_actions_max)
                    
                    # TD3 Improvement 1: Twin Critic Networks (Clipped Double Q-Learning)
                    # Get Q-values from both target critics and take the minimum
                    target_critic_1_inputs = {**next_states_dict, "taken_actions": next_actions}
                    target_critic_2_inputs = {**next_states_dict, "taken_actions": next_actions}
                    
                    target_q1_outputs = self.target_critic_1.act(target_critic_1_inputs)
                    target_q2_outputs = self.target_critic_2.act(target_critic_2_inputs)
                    
                    target_q1_values = target_q1_outputs.get("values", target_q1_outputs.get("actions"))
                    target_q2_values = target_q2_outputs.get("values", target_q2_outputs.get("actions"))
                    
                    # Take minimum of both critics to reduce overestimation bias
                    target_q_values = torch.min(target_q1_values, target_q2_values)
                    
                    target_values = (
                        sampled_rewards +
                        self._discount_factor *
                        (sampled_terminated | sampled_truncated).logical_not() *
                        target_q_values.squeeze()
                    )
                
                # Compute critic losses for both critics
                critic_1_inputs = {**states_dict, "taken_actions": actions_dict}
                critic_2_inputs = {**states_dict, "taken_actions": actions_dict}
                
                critic_1_outputs = self.critic_1.act(critic_1_inputs)
                critic_2_outputs = self.critic_2.act(critic_2_inputs)
                
                critic_1_values = critic_1_outputs.get("values", critic_1_outputs.get("actions"))
                critic_2_values = critic_2_outputs.get("values", critic_2_outputs.get("actions"))
                
                # Both critics are trained to match the same target
                critic_1_loss = F.mse_loss(critic_1_values.squeeze(), target_values)
                critic_2_loss = F.mse_loss(critic_2_values.squeeze(), target_values)
                critic_loss = critic_1_loss + critic_2_loss
            
            # Update both critics
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            
            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                import itertools
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                    self._grad_norm_clip
                )
            
            self.scaler.step(self.critic_optimizer)
            
            # TD3 Improvement 2: Delayed Policy Updates
            # Update policy less frequently than critics to reduce variance
            self._critic_update_counter += 1
            if self._critic_update_counter % self._policy_delay == 0:
                
                with torch.autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cpu', enabled=self._mixed_precision):
                    # Compute policy loss using only critic_1 (as in original TD3 paper)
                    policy_outputs = self.policy.act(states_dict)
                    policy_actions = policy_outputs.get("actions", policy_outputs.get("mean_actions"))
                    
                    # Use only the first critic for policy updates to reduce computation
                    critic_1_inputs_policy = {**states_dict, "taken_actions": policy_actions}
                    critic_1_outputs_policy = self.critic_1.act(critic_1_inputs_policy)
                    critic_1_values_policy = critic_1_outputs_policy.get("values", critic_1_outputs_policy.get("actions"))
                    
                    policy_loss = -critic_1_values_policy.mean()
                
                # Update policy
                self.policy_optimizer.zero_grad()
                self.scaler.scale(policy_loss).backward()
                
                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.policy_optimizer)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                
                self.scaler.step(self.policy_optimizer)
                
                # Soft update target networks only when policy is updated
                self._soft_update(self.target_policy, self.policy, self._polyak)
                self._soft_update(self.target_critic_1, self.critic_1, self._polyak)
                self._soft_update(self.target_critic_2, self.critic_2, self._polyak)
                
                # Track policy loss only when policy is updated
                self.track_data("Loss/Policy", policy_loss.item())
            
            self.scaler.update()
            
            # Update learning rate schedulers
            if self.policy_scheduler is not None and self._critic_update_counter % self._policy_delay == 0:
                self.policy_scheduler.step()
            if self.critic_scheduler is not None:
                self.critic_scheduler.step()
            
            # Track training metrics
            self.track_data("Loss/Critic_1", critic_1_loss.item())
            self.track_data("Loss/Critic_2", critic_2_loss.item())
            self.track_data("Loss/Critic_Total", critic_loss.item())
            
            self.track_data("Q-network/Q1_Max", torch.max(critic_1_values).item())
            self.track_data("Q-network/Q1_Min", torch.min(critic_1_values).item())
            self.track_data("Q-network/Q1_Mean", torch.mean(critic_1_values).item())
            
            self.track_data("Q-network/Q2_Max", torch.max(critic_2_values).item())
            self.track_data("Q-network/Q2_Min", torch.min(critic_2_values).item())
            self.track_data("Q-network/Q2_Mean", torch.mean(critic_2_values).item())
            
            self.track_data("Target/Target_Max", torch.max(target_values).item())
            self.track_data("Target/Target_Min", torch.min(target_values).item())
            self.track_data("Target/Target_Mean", torch.mean(target_values).item())
            
            # Track policy delay counter
            self.track_data("Training/Critic_Update_Counter", self._critic_update_counter)
            self.track_data("Training/Policy_Delay", self._policy_delay)
            
            if self.policy_scheduler is not None:
                self.track_data("Learning/Policy_LR", self.policy_scheduler.get_last_lr()[0])
            if self.critic_scheduler is not None:
                self.track_data("Learning/Critic_LR", self.critic_scheduler.get_last_lr()[0])
        
        logger.debug(f"TD3 update completed at timestep {timestep}")
    
    def save(self, path: str):
        """Save model parameters"""
        save_dict = {
            "policy_state_dict": self.policy.state_dict(),
            "target_policy_state_dict": self.target_policy.state_dict(),
            "critic_1_state_dict": self.critic_1.state_dict(),
            "critic_2_state_dict": self.critic_2.state_dict(),
            "target_critic_1_state_dict": self.target_critic_1.state_dict(),
            "target_critic_2_state_dict": self.target_critic_2.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "cfg": self.cfg,
            "critic_update_counter": self._critic_update_counter,
        }
        
        # Save schedulers if they exist
        if self.policy_scheduler is not None:
            save_dict["policy_scheduler_state_dict"] = self.policy_scheduler.state_dict()
        if self.critic_scheduler is not None:
            save_dict["critic_scheduler_state_dict"] = self.critic_scheduler.state_dict()
        
        torch.save(save_dict, path)
        logger.info(f"TD3 model saved to {path}")
    
    def load(self, path: str):
        """Load model parameters"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load model states
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.target_policy.load_state_dict(checkpoint["target_policy_state_dict"])
        self.critic_1.load_state_dict(checkpoint["critic_1_state_dict"])
        self.critic_2.load_state_dict(checkpoint["critic_2_state_dict"])
        self.target_critic_1.load_state_dict(checkpoint["target_critic_1_state_dict"])
        self.target_critic_2.load_state_dict(checkpoint["target_critic_2_state_dict"])
        
        # Load optimizer states
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        
        # Load scheduler states if they exist
        if "policy_scheduler_state_dict" in checkpoint and self.policy_scheduler is not None:
            self.policy_scheduler.load_state_dict(checkpoint["policy_scheduler_state_dict"])
        if "critic_scheduler_state_dict" in checkpoint and self.critic_scheduler is not None:
            self.critic_scheduler.load_state_dict(checkpoint["critic_scheduler_state_dict"])
        
        # Load critic update counter
        if "critic_update_counter" in checkpoint:
            self._critic_update_counter = checkpoint["critic_update_counter"]
        
        logger.info(f"TD3 model loaded from {path}")