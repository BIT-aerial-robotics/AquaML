from typing import Dict, Any, Optional, Tuple, Union
import copy
import itertools
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
class SACCfg:
    """SAC (Soft Actor-Critic) configuration"""
    
    # Training parameters
    gradient_steps: int = 1  # Number of gradient steps per update
    batch_size: int = 64  # Batch size for training
    
    # Algorithm hyperparameters
    discount_factor: float = 0.99  # Discount factor (gamma)
    polyak: float = 0.005  # Soft update coefficient (tau)
    
    # Learning rates
    actor_learning_rate: float = 1e-3  # Actor learning rate
    critic_learning_rate: float = 1e-3  # Critic learning rate
    learning_rate_scheduler: Optional[Any] = None  # Learning rate scheduler
    learning_rate_scheduler_kwargs: Optional[Dict[str, Any]] = None  # Scheduler kwargs
    
    # Exploration parameters
    random_timesteps: int = 0  # Random exploration steps
    learning_starts: int = 0  # Start learning after this many steps
    
    # Gradient clipping
    grad_norm_clip: float = 0  # Gradient norm clipping
    
    # Entropy regularization
    learn_entropy: bool = True  # Whether to learn entropy coefficient
    entropy_learning_rate: float = 1e-3  # Entropy learning rate
    initial_entropy_value: float = 0.2  # Initial entropy value
    target_entropy: Optional[float] = None  # Target entropy (auto-calculated if None)
    
    # Preprocessors
    state_preprocessor: Optional[Any] = None  # State preprocessor
    state_preprocessor_kwargs: Optional[Dict[str, Any]] = None  # State preprocessor kwargs
    rewards_shaper: Optional[Any] = None  # Reward shaping function
    
    # Training settings
    mixed_precision: bool = False  # Mixed precision training
    
    # Memory parameters
    memory_size: int = 10000  # Size of replay buffer
    device: str = "auto"  # Device to use


class SACMemory:
    """Experience replay buffer for SAC algorithm"""
    
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
    
    def is_ready_for_update(self, min_samples: int) -> bool:
        """Check if memory has enough samples"""
        return self.get_stored_samples_count() >= min_samples


class SAC(Agent):
    """Soft Actor-Critic (SAC) algorithm implementation for AquaML
    
    SAC is an off-policy maximum entropy deep reinforcement learning algorithm.
    It learns both a Q-function and a stochastic policy simultaneously.
    """
    
    def __init__(
        self,
        models: Dict[str, Model],
        cfg: SACCfg,
        observation_space: Optional[Dict[str, Any]] = None,
        action_space: Optional[Dict[str, Any]] = None,
    ):
        """Initialize SAC agent
        
        Args:
            models: Dictionary containing policy, critic_1, critic_2, target_critic_1, target_critic_2
            cfg: SAC configuration
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
        
        logger.info(f"SAC initialized with device: {self.device}")
        
        # Models validation
        required_models = ["policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2"]
        for model_name in required_models:
            if model_name not in models:
                raise ValueError(f"Model '{model_name}' is required for SAC")
        
        # Models
        self.policy = models["policy"]
        self.critic_1 = models["critic_1"]
        self.critic_2 = models["critic_2"]
        self.target_critic_1 = models["target_critic_1"]
        self.target_critic_2 = models["target_critic_2"]
        
        # Move models to device
        for model in [self.policy, self.critic_1, self.critic_2, 
                     self.target_critic_1, self.target_critic_2]:
            model.to(self.device)
        
        # Initialize target networks with same parameters as critics
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        # Setup memory
        self.memory = SACMemory(cfg.memory_size, self.device)
        
        # Configuration parameters
        self._gradient_steps = cfg.gradient_steps
        self._batch_size = cfg.batch_size
        self._discount_factor = cfg.discount_factor
        self._polyak = cfg.polyak
        self._grad_norm_clip = cfg.grad_norm_clip
        self._random_timesteps = cfg.random_timesteps
        self._learning_starts = cfg.learning_starts
        self._mixed_precision = cfg.mixed_precision
        
        # Entropy parameters
        self._learn_entropy = cfg.learn_entropy
        self._initial_entropy_value = cfg.initial_entropy_value
        
        # Target entropy calculation (auto-tune if not specified)
        if cfg.target_entropy is None:
            # Use negative of action dimension as default
            if action_space is not None:
                if hasattr(action_space, 'shape'):
                    action_dim = np.prod(action_space.shape)
                elif isinstance(action_space, dict) and 'shape' in action_space:
                    action_dim = np.prod(action_space['shape'])
                else:
                    action_dim = 1  # Default fallback
                self._target_entropy = -action_dim
            else:
                self._target_entropy = -1  # Default fallback
        else:
            self._target_entropy = cfg.target_entropy
        
        # Entropy coefficient (learnable if enabled)
        if self._learn_entropy:
            self.log_entropy_coeff = torch.tensor(
                np.log(self._initial_entropy_value), 
                device=self.device, 
                requires_grad=True
            )
        else:
            self.log_entropy_coeff = torch.tensor(
                np.log(self._initial_entropy_value), 
                device=self.device
            )
        
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
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=cfg.actor_learning_rate
        )
        
        self.critic_optimizer = torch.optim.Adam(
            itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
            lr=cfg.critic_learning_rate
        )
        
        if self._learn_entropy:
            self.entropy_optimizer = torch.optim.Adam(
                [self.log_entropy_coeff], 
                lr=cfg.entropy_learning_rate
            )
        
        # Learning rate schedulers
        self.policy_scheduler = None
        self.critic_scheduler = None
        
        if cfg.learning_rate_scheduler is not None:
            self.policy_scheduler = cfg.learning_rate_scheduler(
                self.policy_optimizer, **(cfg.learning_rate_scheduler_kwargs or {})
            )
            self.critic_scheduler = cfg.learning_rate_scheduler(
                self.critic_optimizer, **(cfg.learning_rate_scheduler_kwargs or {})
            )
        
        # Initialize memory tensors
        self._init_memory()
        
        # Register checkpoint modules
        self.register_checkpoint_module("policy", self.policy)
        self.register_checkpoint_module("critic_1", self.critic_1)
        self.register_checkpoint_module("critic_2", self.critic_2)
        self.register_checkpoint_module("target_critic_1", self.target_critic_1)
        self.register_checkpoint_module("target_critic_2", self.target_critic_2)
        self.register_checkpoint_module("policy_optimizer", self.policy_optimizer)
        self.register_checkpoint_module("critic_optimizer", self.critic_optimizer)
        
        if self._learn_entropy:
            self.register_checkpoint_module("entropy_optimizer", self.entropy_optimizer)
            self.register_checkpoint_module("log_entropy_coeff", self.log_entropy_coeff)
        
        # Tensor names for sampling
        self._tensors_names = [
            "states", "actions", "rewards", "next_states", "terminated", "truncated"
        ]
        
        logger.info("SAC agent initialized successfully")
    
    def _init_memory(self):
        """Initialize memory tensors"""
        # Create placeholder tensors (will be resized dynamically)
        self.memory.create_tensor("states", (1,), torch.float32)
        self.memory.create_tensor("actions", (1,), torch.float32)
        self.memory.create_tensor("rewards", (1,), torch.float32)
        self.memory.create_tensor("next_states", (1,), torch.float32)
        self.memory.create_tensor("terminated", (1,), torch.bool)
        self.memory.create_tensor("truncated", (1,), torch.bool)
    
    def act(
        self, 
        states: Dict[str, torch.Tensor], 
        timestep: int, 
        timesteps: int
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
            # Use policy's random action method if available
            if hasattr(self.policy, 'random_act'):
                outputs = self.policy.random_act(states)
            else:
                # Fallback: use policy but sample randomly
                with torch.no_grad():
                    outputs = self.policy.act(states)
                    # Add noise to actions for exploration
                    if 'actions' in outputs:
                        noise = torch.randn_like(outputs['actions']) * 0.5
                        outputs['actions'] = torch.tanh(outputs['actions'] + noise)
            logger.debug(f"Random exploration at timestep {timestep}")
            return outputs
        
        # Generate actions from policy
        with torch.cuda.amp.autocast(enabled=self._mixed_precision):
            if timestep < self._learning_starts:
                # During initial exploration, sample from policy
                outputs = self.policy.act(states)
            else:
                # During training, use deterministic actions for better exploitation
                outputs = self.policy.act(states, deterministic=False)
        
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
        # Start learning after enough samples and steps
        if (timestep >= self._learning_starts and 
            self.memory.is_ready_for_update(self._batch_size)):
            
            # Set models to training mode
            self.policy.train()
            self.critic_1.train()
            self.critic_2.train()
            
            # Perform multiple gradient steps
            for _ in range(self._gradient_steps):
                self._update(timestep, timesteps)
            
            # Set models back to eval mode
            self.policy.eval()
            self.critic_1.eval()
            self.critic_2.eval()
    
    def _update(self, timestep: int, timesteps: int):
        """Main SAC update step"""
        # Sample batch from memory
        batch = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)
        (
            sampled_states,
            sampled_actions,
            sampled_rewards,
            sampled_next_states,
            sampled_terminated,
            sampled_truncated,
        ) = batch
        
        # Detach tensors
        sampled_states = sampled_states.detach()
        sampled_actions = sampled_actions.detach()
        sampled_rewards = sampled_rewards.detach()
        sampled_next_states = sampled_next_states.detach()
        sampled_terminated = sampled_terminated.detach()
        sampled_truncated = sampled_truncated.detach()
        
        dones = sampled_terminated | sampled_truncated
        
        with torch.cuda.amp.autocast(enabled=self._mixed_precision):
            # Update critics
            critic_loss = self._update_critics(
                sampled_states, sampled_actions, sampled_rewards, 
                sampled_next_states, dones
            )
            
            # Update policy
            policy_loss = self._update_policy(sampled_states)
            
            # Update entropy coefficient
            entropy_loss = None
            if self._learn_entropy:
                entropy_loss = self._update_entropy(sampled_states)
            
            # Soft update target networks
            self._soft_update_targets()
        
        # Update learning rate schedulers
        if self.policy_scheduler is not None:
            self.policy_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
        
        # Track data
        self.track_data("Loss/Critic", critic_loss.item())
        self.track_data("Loss/Policy", policy_loss.item())
        if entropy_loss is not None:
            self.track_data("Loss/Entropy", entropy_loss.item())
        
        # Track entropy coefficient
        entropy_coeff = torch.exp(self.log_entropy_coeff).item()
        self.track_data("Training/Entropy_Coefficient", entropy_coeff)
        self.track_data("Training/Target_Entropy", self._target_entropy)
        
        logger.debug(f"SAC update at timestep {timestep} - "
                    f"Critic Loss: {critic_loss.item():.6f}, "
                    f"Policy Loss: {policy_loss.item():.6f}")
    
    def _update_critics(self, states, actions, rewards, next_states, dones):
        """Update critic networks"""
        # Convert flattened tensors back to dictionaries (simplified for this implementation)
        states_dict = {"state": states}  # Simplified - in practice, use proper reconstruction
        next_states_dict = {"state": next_states}
        actions_dict = {"actions": actions}
        
        # Compute target Q values
        with torch.no_grad():
            # Sample next actions from policy
            next_actions_outputs = self.policy.act(next_states_dict)
            next_actions = next_actions_outputs.get("actions", next_actions_outputs.get("mean_actions"))
            next_log_prob = next_actions_outputs.get("log_prob", torch.zeros_like(next_actions[..., :1]))
            
            # Compute target Q values using target critics
            target_q1_outputs = self.target_critic_1.act({**next_states_dict, **{"actions": next_actions}})
            target_q2_outputs = self.target_critic_2.act({**next_states_dict, **{"actions": next_actions}})
            
            target_q1 = target_q1_outputs.get("values", target_q1_outputs.get("actions"))
            target_q2 = target_q2_outputs.get("values", target_q2_outputs.get("actions"))
            
            # Take minimum and subtract entropy
            entropy_coeff = torch.exp(self.log_entropy_coeff)
            target_q = torch.min(target_q1, target_q2) - entropy_coeff * next_log_prob
            
            # Compute target values
            target_values = rewards + self._discount_factor * (1 - dones.float()) * target_q
        
        # Compute current Q values
        q1_outputs = self.critic_1.act({**states_dict, **actions_dict})
        q2_outputs = self.critic_2.act({**states_dict, **actions_dict})
        
        q1_values = q1_outputs.get("values", q1_outputs.get("actions"))
        q2_values = q2_outputs.get("values", q2_outputs.get("actions"))
        
        # Compute critic loss
        critic_1_loss = F.mse_loss(q1_values, target_values)
        critic_2_loss = F.mse_loss(q2_values, target_values)
        critic_loss = critic_1_loss + critic_2_loss
        
        # Optimize critics
        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        
        if self._grad_norm_clip > 0:
            self.scaler.unscale_(self.critic_optimizer)
            nn.utils.clip_grad_norm_(
                itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                self._grad_norm_clip
            )
        
        self.scaler.step(self.critic_optimizer)
        self.scaler.update()
        
        return critic_loss
    
    def _update_policy(self, states):
        """Update policy network"""
        states_dict = {"state": states}  # Simplified
        
        # Sample actions from current policy
        policy_outputs = self.policy.act(states_dict)
        actions = policy_outputs.get("actions", policy_outputs.get("mean_actions"))
        log_prob = policy_outputs.get("log_prob", torch.zeros_like(actions[..., :1]))
        
        # Compute Q values for policy actions
        q1_outputs = self.critic_1.act({**states_dict, **{"actions": actions}})
        q2_outputs = self.critic_2.act({**states_dict, **{"actions": actions}})
        
        q1_values = q1_outputs.get("values", q1_outputs.get("actions"))
        q2_values = q2_outputs.get("values", q2_outputs.get("actions"))
        
        # Take minimum Q value
        q_values = torch.min(q1_values, q2_values)
        
        # Compute policy loss (maximize Q - entropy_coeff * log_prob)
        entropy_coeff = torch.exp(self.log_entropy_coeff)
        policy_loss = (entropy_coeff * log_prob - q_values).mean()
        
        # Optimize policy
        self.policy_optimizer.zero_grad()
        self.scaler.scale(policy_loss).backward()
        
        if self._grad_norm_clip > 0:
            self.scaler.unscale_(self.policy_optimizer)
            nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
        
        self.scaler.step(self.policy_optimizer)
        self.scaler.update()
        
        return policy_loss
    
    def _update_entropy(self, states):
        """Update entropy coefficient"""
        states_dict = {"state": states}  # Simplified
        
        with torch.no_grad():
            policy_outputs = self.policy.act(states_dict)
            log_prob = policy_outputs.get("log_prob", torch.zeros(1, device=self.device))
        
        # Compute entropy loss
        entropy_loss = -(self.log_entropy_coeff * (log_prob + self._target_entropy)).mean()
        
        # Optimize entropy coefficient
        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_optimizer.step()
        
        return entropy_loss
    
    def _soft_update_targets(self):
        """Soft update target networks"""
        # Update target_critic_1
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self._polyak) + param.data * self._polyak
            )
        
        # Update target_critic_2
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self._polyak) + param.data * self._polyak
            )
    
    def save(self, path: str):
        """Save model parameters"""
        modules = {
            "policy": self.policy.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_entropy_coeff": self.log_entropy_coeff,
            "cfg": self.cfg,
        }
        
        if self._learn_entropy:
            modules["entropy_optimizer"] = self.entropy_optimizer.state_dict()
        
        torch.save(modules, path)
        logger.info(f"SAC model saved to {path}")
    
    def load(self, path: str):
        """Load model parameters"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.target_critic_1.load_state_dict(checkpoint["target_critic_1"])
        self.target_critic_2.load_state_dict(checkpoint["target_critic_2"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.log_entropy_coeff = checkpoint["log_entropy_coeff"]
        
        if self._learn_entropy and "entropy_optimizer" in checkpoint:
            self.entropy_optimizer.load_state_dict(checkpoint["entropy_optimizer"])
        
        logger.info(f"SAC model loaded from {path}")