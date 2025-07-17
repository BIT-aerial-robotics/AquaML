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
from AquaML.utils.schedulers import KLAdaptiveLR


@configclass
class PPOCfg:
    # Rollout and learning parameters
    rollouts: int = 16  # number of rollouts before updating
    learning_epochs: int = 8  # number of learning epochs during each update
    mini_batches: int = 2  # number of mini batches during each learning epoch

    # Discount and lambda parameters
    discount_factor: float = 0.99  # discount factor (gamma)
    lambda_value: float = (
        0.95  # TD(lambda) coefficient (lam) for computing returns and advantages
    )

    # Learning rate parameters
    learning_rate: float = 1e-3  # learning rate
    learning_rate_scheduler: Optional[Any] = (
        None  # learning rate scheduler class (see torch.optim.lr_scheduler)
    )
    learning_rate_scheduler_kwargs: Optional[Dict[str, Any]] = None  # learning rate scheduler's kwargs

    # Preprocessor parameters
    state_preprocessor: Optional[Any] = None  # state preprocessor class
    state_preprocessor_kwargs: Optional[Dict[str, Any]] = None  # state preprocessor's kwargs
    value_preprocessor: Optional[Any] = None  # value preprocessor class
    value_preprocessor_kwargs: Optional[Dict[str, Any]] = None  # value preprocessor's kwargs

    # Exploration and learning start parameters
    random_timesteps: int = 0  # random exploration steps
    learning_starts: int = 0  # learning starts after this many steps

    # Clipping parameters
    grad_norm_clip: float = 0.5  # clipping coefficient for the norm of the gradients
    ratio_clip: float = (
        0.2  # clipping coefficient for computing the clipped surrogate objective
    )
    value_clip: float = (
        0.2  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    )
    clip_predicted_values: bool = (
        False  # clip predicted values during value loss computation
    )

    # Loss scaling parameters
    entropy_loss_scale: float = 0.0  # entropy loss scaling factor
    value_loss_scale: float = 1.0  # value loss scaling factor

    # KL divergence threshold
    kl_threshold: float = 0  # KL divergence threshold for early stopping

    # Rewards shaping and time limit bootstrap
    rewards_shaper: Optional[Any] = (
        None  # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    )
    time_limit_bootstrap: bool = (
        False  # bootstrap at timeout termination (episode truncation)
    )

    # Mixed precision training
    mixed_precision: bool = (
        False  # enable automatic mixed precision for higher performance
    )

    # Memory and buffer parameters
    memory_size: int = 10000  # size of the replay buffer
    device: str = "auto"  # device to use for training


class PPOMemory:
    """Enhanced memory buffer for PPO algorithm with better data structure handling"""

    def __init__(self, memory_size: int, device: str = "cpu"):
        self.memory_size = memory_size
        self.device = device
        self.position = 0
        self.full = False
        self.tensors = {}
        self.data_structure_info = {}  # Store metadata about data structures

    def create_tensor(
        self,
        name: str,
        size: Union[int, Tuple[int]],
        dtype: torch.dtype = torch.float32,
    ):
        """Create a tensor in memory"""
        if isinstance(size, int):
            size = (size,)
        self.tensors[name] = torch.zeros(
            (self.memory_size,) + size, dtype=dtype, device=self.device
        )

    def store_data_structure(self, name: str, structure_info: Dict[str, Any]):
        """Store metadata about dictionary data structures"""
        self.data_structure_info[name] = structure_info

    def add_samples(self, **kwargs):
        """Add samples to memory with improved structure preservation"""
        # Handle dictionary data with structure preservation
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
                    flattened = self._flatten_dict_structured(value)
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
                    self.tensors[key][self.position] = torch.tensor(
                        value, device=self.device
                    )

        self.position = (self.position + 1) % self.memory_size
        if self.position == 0:
            self.full = True

    def _flatten_dict_structured(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten dictionary with better structure preservation"""
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
                # Store original shape info for reconstruction
                original_shape = tensor.shape
                tensor = tensor.flatten(start_dim=1)
            
            tensors.append(tensor)
        
        if tensors:
            return torch.cat(tensors, dim=-1)
        else:
            return torch.tensor([], device=self.device)

    def _unflatten_dict_structured(self, flattened_tensor: torch.Tensor, structure_name: str) -> Dict[str, torch.Tensor]:
        """Reconstruct dictionary from flattened tensor using stored structure info"""
        if structure_name not in self.data_structure_info:
            # Fallback to simple reconstruction
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

    def get_tensor_by_name(self, name: str) -> torch.Tensor:
        """Get tensor by name"""
        if name in self.tensors:
            if self.full:
                return self.tensors[name]
            else:
                return self.tensors[name][: self.position]
        return None

    def set_tensor_by_name(self, name: str, tensor: torch.Tensor):
        """Set tensor by name"""
        if name in self.tensors:
            # 确保tensor形状匹配
            if tensor.shape != self.tensors[name].shape:
                # 如果形状不匹配，重新创建tensor
                self.tensors[name] = torch.zeros_like(tensor, device=self.device)
            
            if self.full:
                self.tensors[name] = tensor
            else:
                # 确保tensor形状匹配存储空间
                target_shape = self.tensors[name][: self.position].shape
                if tensor.shape != target_shape:
                    # 调整tensor形状以匹配目标形状
                    if tensor.dim() > len(target_shape):
                        # 如果tensor维度过多，需要压缩
                        tensor = tensor.squeeze()
                    elif tensor.dim() < len(target_shape):
                        # 如果tensor维度不足，需要扩展
                        tensor = tensor.unsqueeze(-1)
                    
                    # 确保形状完全匹配
                    if tensor.shape != target_shape:
                        tensor = tensor.view(target_shape)
                
                self.tensors[name][: self.position] = tensor

    def sample_all(self, names: list, mini_batches: int = 1):
        """Sample all data in mini-batches with improved efficiency"""
        total_samples = self.memory_size if self.full else self.position
        
        if total_samples == 0:
            return
            
        batch_size = max(1, total_samples // mini_batches)
        
        # Generate random permutation once for better efficiency
        indices = torch.randperm(total_samples, device=self.device)

        for i in range(mini_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples) if i < mini_batches - 1 else total_samples
            
            if start_idx >= end_idx:
                continue
                
            batch_indices = indices[start_idx:end_idx]

            batch_data = []
            for name in names:
                if name in self.tensors:
                    batch_data.append(self.tensors[name][batch_indices])
                else:
                    batch_data.append(torch.tensor([], device=self.device))

            yield batch_data

    def clear(self):
        """Clear memory"""
        self.position = 0
        self.full = False
        for tensor in self.tensors.values():
            tensor.zero_()

    def get_stored_samples_count(self) -> int:
        """Get number of stored samples"""
        return self.memory_size if self.full else self.position

    def is_ready_for_update(self, min_samples: int) -> bool:
        """Check if memory has enough samples for update"""
        return self.get_stored_samples_count() >= min_samples


class PPO(Agent):
    """Proximal Policy Optimization (PPO) algorithm implementation for AquaML

    This implementation follows the dictionary-based architecture of AquaML
    where observations and actions are dictionaries, allowing flexible model inputs.
    """

    def __init__(
        self,
        models: Dict[str, Model],
        cfg: PPOCfg,
        observation_space: Optional[Dict[str, Any]] = None,
        action_space: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize PPO agent

        Args:
            models: Dictionary containing policy and value models
            cfg: PPO configuration
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

        logger.info(f"PPO initialized with device: {self.device}")

        # Models
        self.policy = models.get("policy")
        self.value = models.get("value")

        if self.policy is None:
            raise ValueError("Policy model is required for PPO")
        if self.value is None:
            raise ValueError("Value model is required for PPO")

        # Move models to device
        self.policy.to(self.device)
        self.value.to(self.device)

        # Setup memory
        self.memory = PPOMemory(cfg.memory_size, self.device)

        # Configuration parameters
        self._learning_epochs = cfg.learning_epochs
        self._mini_batches = cfg.mini_batches
        self._rollouts = cfg.rollouts
        self._rollout = 0

        self._grad_norm_clip = cfg.grad_norm_clip
        self._ratio_clip = cfg.ratio_clip
        self._value_clip = cfg.value_clip
        self._clip_predicted_values = cfg.clip_predicted_values

        self._value_loss_scale = cfg.value_loss_scale
        self._entropy_loss_scale = cfg.entropy_loss_scale

        self._kl_threshold = cfg.kl_threshold

        self._learning_rate = cfg.learning_rate
        self._learning_rate_scheduler = cfg.learning_rate_scheduler

        self._discount_factor = cfg.discount_factor
        self._lambda = cfg.lambda_value

        self._random_timesteps = cfg.random_timesteps
        self._learning_starts = cfg.learning_starts

        self._rewards_shaper = cfg.rewards_shaper
        self._time_limit_bootstrap = cfg.time_limit_bootstrap

        self._mixed_precision = cfg.mixed_precision

        # Mixed precision scaler
        try:
            # Try with device_type parameter (newer PyTorch)
            self.scaler = torch.amp.GradScaler(device_type='cuda', enabled=self._mixed_precision)
        except TypeError:
            # Fallback for older PyTorch versions
            self.scaler = torch.amp.GradScaler(enabled=self._mixed_precision)

        # Initialize preprocessors
        self._state_preprocessor = None
        self._value_preprocessor = None
        
        if cfg.state_preprocessor is not None:
            self._state_preprocessor = cfg.state_preprocessor(**(cfg.state_preprocessor_kwargs or {}))
            logger.info("State preprocessor initialized")
        
        if cfg.value_preprocessor is not None:
            self._value_preprocessor = cfg.value_preprocessor(**(cfg.value_preprocessor_kwargs or {}))
            logger.info("Value preprocessor initialized")

        # Setup optimizer
        if self.policy is self.value:
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=self._learning_rate
            )
        else:
            self.optimizer = torch.optim.Adam(
                itertools.chain(self.policy.parameters(), self.value.parameters()),
                lr=self._learning_rate,
            )

        # Setup learning rate scheduler
        if self._learning_rate_scheduler is not None:
            if self._learning_rate_scheduler == "KLAdaptiveLR":
                # Use KL adaptive learning rate
                self.scheduler = KLAdaptiveLR(
                    self.optimizer, **cfg.learning_rate_scheduler_kwargs or {}
                )
                logger.info("Using KL adaptive learning rate scheduler")
            else:
                # Use custom scheduler
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **cfg.learning_rate_scheduler_kwargs or {}
                )
                logger.info(f"Using custom learning rate scheduler: {self._learning_rate_scheduler}")
        else:
            self.scheduler = None

        # Initialize memory tensors
        self._init_memory()

        # Current state variables
        self._current_log_prob = None
        self._current_next_states = None

        # Register checkpoint modules (following SKRL pattern)
        self.register_checkpoint_module("policy", self.policy)
        self.register_checkpoint_module("value", self.value)
        self.register_checkpoint_module("optimizer", self.optimizer)
        if self.scheduler is not None:
            self.register_checkpoint_module("scheduler", self.scheduler)

        logger.info("PPO agent initialized successfully")

    def _init_memory(self):
        """Initialize memory tensors"""
        # We'll determine sizes dynamically based on first interaction
        # For now, create placeholder tensors
        self.memory.create_tensor("states", (1,), torch.float32)  # Will be resized
        self.memory.create_tensor("actions", (1,), torch.float32)  # Will be resized
        self.memory.create_tensor("rewards", (1,), torch.float32)
        self.memory.create_tensor("terminated", (1,), torch.bool)
        self.memory.create_tensor("truncated", (1,), torch.bool)
        self.memory.create_tensor("log_prob", (1,), torch.float32)
        self.memory.create_tensor("values", (1,), torch.float32)
        self.memory.create_tensor("returns", (1,), torch.float32)
        self.memory.create_tensor("advantages", (1,), torch.float32)

        # Tensors used during training
        self._tensors_names = [
            "states",
            "actions",
            "log_prob",
            "values",
            "returns",
            "advantages",
        ]

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
            logger.debug(f"Random exploration at timestep {timestep}")
            # For random actions, we'd need to implement random_act in the policy
            # For now, just use the policy

        # Get actions from policy
        with torch.cuda.amp.autocast(enabled=self._mixed_precision):
            outputs = self.policy.act(states)

        # Store current log probability for memory
        self._current_log_prob = outputs.get("log_prob")

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

        self._current_next_states = next_states

        # Reward shaping
        if self._rewards_shaper is not None:
            rewards = self._rewards_shaper(rewards, timestep, timesteps)

        # Apply state preprocessing for value computation
        states_for_value = states
        if self._state_preprocessor is not None:
            processed_states = {}
            for k, v in states.items():
                processed_states[k] = self._state_preprocessor(v)
            states_for_value = processed_states

        # Compute values for current states
        with torch.cuda.amp.autocast(enabled=self._mixed_precision):
            value_outputs = self.value.act(states_for_value)
            values = value_outputs.get(
                "values", value_outputs.get("actions")
            )  # Fallback to actions if no values key
            
            # Apply value preprocessing if available
            if self._value_preprocessor is not None:
                values = self._value_preprocessor(values)

        # Time-limit bootstrapping
        if self._time_limit_bootstrap:
            rewards += self._discount_factor * values * truncated

        # Store transition in memory with improved data handling
        self.memory.add_samples(
            states=states,  # Store as dictionary directly
            actions=actions,  # Store actions as is (dict or tensor)
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            log_prob=self._current_log_prob,
            values=values,
        )

        logger.debug(f"Recorded transition at timestep {timestep}")


    def post_interaction(self, timestep: int, timesteps: int):
        """Called after each environment interaction"""
        self._rollout += 1

        # Update policy after collecting enough rollouts
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            # Check if we have enough samples for update
            if not self.memory.is_ready_for_update(self._rollouts):
                logger.warning(f"Not enough samples for update at timestep {timestep}")
                return
                
            logger.debug(f"Starting PPO update at timestep {timestep}")
            
            # Compute GAE before training
            self._compute_gae()
            
            # Set models to training mode
            if self.policy is not None:
                self.policy.train()
            if self.value is not None:
                self.value.train()
                
            # Perform update
            self._update(timestep, timesteps)
            
            # Set models back to eval mode
            if self.policy is not None:
                self.policy.eval()
            if self.value is not None:
                self.value.eval()

    def _update(self, timestep: int, timesteps: int):
        """Main PPO update step"""
        # GAE is already computed in post_interaction

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # Learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # Sample fresh batches for each epoch
            sampled_batches = self.memory.sample_all(
                names=self._tensors_names, mini_batches=self._mini_batches
            )

            # Mini-batch loop
            for batch in sampled_batches:
                (
                    sampled_states,
                    sampled_actions,
                    sampled_log_prob,
                    sampled_values,
                    sampled_returns,
                    sampled_advantages,
                ) = batch

                # Detach all tensors to prevent gradient accumulation issues
                sampled_states = sampled_states.detach()
                sampled_actions = sampled_actions.detach()
                sampled_log_prob = sampled_log_prob.detach()
                sampled_values = sampled_values.detach()
                sampled_returns = sampled_returns.detach()
                sampled_advantages = sampled_advantages.detach()

                with torch.cuda.amp.autocast(enabled=self._mixed_precision):
                    # Convert flattened states back to dictionary format
                    states_dict = self.memory._unflatten_dict_structured(sampled_states, "states")
                    actions_dict = self.memory._unflatten_dict_structured(sampled_actions, "actions")

                    # Get new log probabilities
                    policy_outputs = self.policy.act(
                        {**states_dict, "taken_actions": actions_dict}
                    )
                    new_log_prob = policy_outputs.get("log_prob")

                    # Compute approximate KL divergence
                    with torch.no_grad():
                        ratio = new_log_prob.detach() - sampled_log_prob.detach()
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence.item())
                        self._last_kl_divergence = kl_divergence.item()

                    # Early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        logger.warning(
                            f"Early stopping due to KL divergence: {kl_divergence}"
                        )
                        break

                    # Compute entropy loss
                    entropy_loss = 0
                    if self._entropy_loss_scale > 0:
                        entropy = self.policy.get_entropy()
                        entropy_loss = -self._entropy_loss_scale * entropy.mean()

                    # Compute policy loss (clipped surrogate objective)
                    ratio = torch.exp(new_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clamp(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # Compute value loss
                    value_outputs = self.value.act(states_dict)
                    predicted_values = value_outputs.get(
                        "values", value_outputs.get("actions")
                    )

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clamp(
                            predicted_values - sampled_values,
                            min=-self._value_clip,
                            max=self._value_clip,
                        )

                    value_loss = self._value_loss_scale * F.mse_loss(
                        sampled_returns, predicted_values
                    )

                # Optimization step
                self.optimizer.zero_grad()
                total_loss = policy_loss + entropy_loss + value_loss
                self.scaler.scale(total_loss).backward()

                # Gradient clipping with norm recording
                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.policy.parameters(), self._grad_norm_clip
                        )
                    else:
                        grad_norm = nn.utils.clip_grad_norm_(
                            itertools.chain(
                                self.policy.parameters(), self.value.parameters()
                            ),
                            self._grad_norm_clip,
                        )
                    self._last_grad_norm = grad_norm.item()
                else:
                    # Calculate gradient norm without clipping
                    total_norm = 0
                    for p in itertools.chain(self.policy.parameters(), 
                                           self.value.parameters() if self.policy is not self.value else []):
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    self._last_grad_norm = total_norm ** (1. / 2)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale > 0:
                    cumulative_entropy_loss += entropy_loss.item()

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    # Use KL divergence for adaptive learning rate
                    if len(kl_divergences) > 0:
                        kl_mean = float(np.mean([kl.item() if isinstance(kl, torch.Tensor) else kl for kl in kl_divergences]))
                        self.scheduler.step(kl_mean)
                        logger.debug(f"KL adaptive LR step: KL={kl_mean:.6f}, LR={self.scheduler.get_last_lr()[0]:.6f}")
                    else:
                        self.scheduler.step()
                elif hasattr(self.scheduler, "step"):
                    # Regular scheduler
                    self.scheduler.step()

        # Log training metrics with enhanced details
        total_batches = self._learning_epochs * self._mini_batches
        avg_policy_loss = cumulative_policy_loss / total_batches
        avg_value_loss = cumulative_value_loss / total_batches
        
        # Core losses
        self.track_data("Loss/Policy", avg_policy_loss)
        self.track_data("Loss/Value", avg_value_loss)
        self.track_data("Loss/Total", avg_policy_loss + avg_value_loss)
        
        # Entropy loss if enabled
        if self._entropy_loss_scale > 0:
            avg_entropy_loss = cumulative_entropy_loss / total_batches
            self.track_data("Loss/Entropy", avg_entropy_loss)
        
        # Training statistics
        self.track_data("Training/Learning_Rate", self.optimizer.param_groups[0]['lr'])
        self.track_data("Training/Gradient_Norm", self._last_grad_norm if hasattr(self, '_last_grad_norm') else 0.0)
        self.track_data("Training/Samples_Count", self.memory.get_stored_samples_count())
        
        # GAE statistics
        advantages_tensor = self.memory.get_tensor_by_name("advantages")
        if advantages_tensor is not None:
            self.track_data("GAE/Advantages_Mean", advantages_tensor.mean().item())
            self.track_data("GAE/Advantages_Std", advantages_tensor.std().item())
        
        returns_tensor = self.memory.get_tensor_by_name("returns")
        if returns_tensor is not None:
            self.track_data("GAE/Returns_Mean", returns_tensor.mean().item())
            self.track_data("GAE/Returns_Std", returns_tensor.std().item())
        
        # KL divergence if available
        if hasattr(self, '_last_kl_divergence'):
            self.track_data("Policy/KL_Divergence", self._last_kl_divergence)
        
        # Policy statistics (if policy supports it)
        if hasattr(self.policy, 'distribution') and callable(getattr(self.policy, 'distribution', None)):
            try:
                dist = self.policy.distribution()
                if hasattr(dist, 'stddev'):
                    self.track_data("Policy/Action_Std", dist.stddev.mean().item())
            except:
                pass  # Skip if not available
        
        logger.info(f"PPO Update - Policy Loss: {avg_policy_loss:.6f}, Value Loss: {avg_value_loss:.6f}, "
                   f"Samples: {self.memory.get_stored_samples_count()}")

        # Clear memory for next rollout
        self.memory.clear()

    def _compute_gae(self):
        """Compute Generalized Advantage Estimation"""

        # Get last values for GAE computation
        with torch.no_grad():
            if self._current_next_states is not None:
                # Apply state preprocessing if available
                next_states_for_value = self._current_next_states
                if self._state_preprocessor is not None:
                    processed_states = {}
                    for k, v in self._current_next_states.items():
                        processed_states[k] = self._state_preprocessor(v)
                    next_states_for_value = processed_states
                    
                with torch.cuda.amp.autocast(enabled=self._mixed_precision):
                    last_value_outputs = self.value.act(next_states_for_value)
                    last_values = last_value_outputs.get(
                        "values", last_value_outputs.get("actions")
                    )
                    
                    # Apply value preprocessing if available
                    if self._value_preprocessor is not None:
                        last_values = self._value_preprocessor(last_values)
            else:
                last_values = torch.zeros(1, device=self.device)

        # Get stored values
        rewards = self.memory.get_tensor_by_name("rewards")
        values = self.memory.get_tensor_by_name("values")
        terminated = self.memory.get_tensor_by_name("terminated")
        truncated = self.memory.get_tensor_by_name("truncated")
        dones = terminated | truncated

        # Compute GAE
        returns, advantages = self._compute_gae_values(
            rewards, dones, values, last_values, self._discount_factor, self._lambda
        )

        # Store computed returns and advantages
        self.memory.set_tensor_by_name("returns", returns)
        self.memory.set_tensor_by_name("advantages", advantages)

    def _compute_gae_values(
        self, rewards, dones, values, next_values, discount_factor, lambda_coeff
    ):
        """Compute GAE values"""
        # 确保所有tensor都是正确的形状
        if rewards.dim() > 1:
            rewards = rewards.squeeze()
        if dones.dim() > 1:
            dones = dones.squeeze()
        if values.dim() > 1:
            values = values.squeeze()
        if next_values.dim() > 1:
            next_values = next_values.squeeze()
        
        advantage = 0
        advantages = torch.zeros_like(rewards)
        not_dones = dones.logical_not()
        memory_size = rewards.shape[0]

        # Advantages computation
        for i in reversed(range(memory_size)):
            next_value = values[i + 1] if i < memory_size - 1 else next_values
            advantage = (
                rewards[i]
                - values[i]
                + discount_factor
                * not_dones[i]
                * (next_value + lambda_coeff * advantage)
            )
            advantages[i] = advantage

        # Returns computation
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 确保返回的tensor形状为 (memory_size,) 1D tensor
        # 这样可以避免与存储tensor的形状不匹配
        if returns.dim() != 1:
            returns = returns.squeeze()
        if advantages.dim() != 1:
            advantages = advantages.squeeze()

        return returns, advantages


    def save(self, path: str):
        """Save model parameters"""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_state_dict": self.value.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "cfg": self.cfg,
            },
            path,
        )
        logger.info(f"PPO model saved to {path}")

    def load(self, path: str):
        """Load model parameters"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "policy" in checkpoint:
            self.policy.load_state_dict(checkpoint["policy"])
            self.value.load_state_dict(checkpoint["value"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.value.load_state_dict(checkpoint["value_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"PPO model loaded from {path}")
