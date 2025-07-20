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
from AquaML.learning.memory import SequentialMemory, SequentialMemoryCfg
from AquaML.config import configclass
from AquaML.utils.schedulers import KLAdaptiveLR


@configclass
class A2CCfg:
    """Configuration class for A2C (Advantage Actor-Critic) algorithm"""
    
    # Rollout and learning parameters
    rollouts: int = 16  # number of rollouts before updating
    mini_batches: int = 1  # number of mini batches during each update (A2C typically uses 1)

    # Discount and lambda parameters
    discount_factor: float = 0.99  # discount factor (gamma)
    lambda_value: float = 0.95  # TD(lambda) coefficient (lam) for computing returns and advantages

    # Learning rate parameters
    learning_rate: float = 1e-3  # learning rate
    learning_rate_scheduler: Optional[Any] = None  # learning rate scheduler class
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

    # Loss scaling parameters
    entropy_loss_scale: float = 0.0  # entropy loss scaling factor

    # Rewards shaping and time limit bootstrap
    rewards_shaper: Optional[Any] = None  # rewards shaping function
    time_limit_bootstrap: bool = False  # bootstrap at timeout termination (episode truncation)

    # Mixed precision training
    mixed_precision: bool = False  # enable automatic mixed precision for higher performance

    # Memory and buffer parameters
    memory_size: int = 10000  # size of the replay buffer
    device: str = "auto"  # device to use for training


class A2C(Agent):
    """Advantage Actor-Critic (A2C) algorithm implementation for AquaML
    
    A2C is a simpler on-policy algorithm compared to PPO, using GAE for advantage estimation
    without clipping or multiple epochs of updates.
    """

    @coordinator.registerAgent
    def __init__(
        self,
        models: Dict[str, Model],
        cfg: A2CCfg,
        observation_space: Optional[Dict[str, Any]] = None,
        action_space: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize A2C agent

        Args:
            models: Dictionary containing policy and value models
            cfg: A2C configuration
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

        logger.info(f"A2C initialized with device: {self.device}")

        # Models
        self.policy = models.get("policy")
        self.value = models.get("value")

        if self.policy is None:
            raise ValueError("Policy model is required for A2C")
        if self.value is None:
            raise ValueError("Value model is required for A2C")

        # Move models to device
        self.policy.to(self.device)
        self.value.to(self.device)

        # Setup memory - support multi-environment rollouts
        memory_cfg = SequentialMemoryCfg(
            memory_size=cfg.memory_size,
            device=self.device,
            num_envs=None  # Will be determined from environment info
        )
        self.memory = SequentialMemory(memory_cfg)

        # Configuration parameters
        self._mini_batches = cfg.mini_batches
        self._rollouts = cfg.rollouts
        self._rollout = 0

        self._grad_norm_clip = cfg.grad_norm_clip
        self._entropy_loss_scale = cfg.entropy_loss_scale

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

        # Register checkpoint modules
        self.register_checkpoint_module("policy", self.policy)
        self.register_checkpoint_module("value", self.value)
        self.register_checkpoint_module("optimizer", self.optimizer)
        if self.scheduler is not None:
            self.register_checkpoint_module("scheduler", self.scheduler)

        logger.info("A2C agent initialized successfully")

    def _init_memory(self):
        """Initialize memory tensors"""
        # Create placeholder tensors that will be resized dynamically
        self.memory.create_tensor("states", (1,), torch.float32)
        self.memory.create_tensor("actions", (1,), torch.float32)
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
            # For A2C, we assume the policy has a random_act method
            # or we can just use normal policy with high temperature/noise

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

        # Store transition in memory
        self.memory.add_samples(
            states=states,
            actions=actions,
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
                
            logger.debug(f"Starting A2C update at timestep {timestep}")
            
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
        """Main A2C update step"""
        # Compute GAE returns and advantages
        self._compute_gae()

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        kl_divergences = []

        # Sample batches for A2C update (typically just one mini-batch)
        sampled_batches = self.memory.sample_all_shuffled(
            names=self._tensors_names, mini_batches=self._mini_batches
        )

        # Mini-batch loop (usually just one iteration for A2C)
        for batch in sampled_batches:
            (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_returns,
                sampled_advantages,
            ) = batch

            # Detach all tensors to prevent gradient accumulation issues
            sampled_states = sampled_states.detach()
            sampled_actions = sampled_actions.detach()
            sampled_log_prob = sampled_log_prob.detach()
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

                # Compute approximate KL divergence for KL adaptive scheduler
                if self.scheduler and isinstance(self.scheduler, KLAdaptiveLR):
                    with torch.no_grad():
                        ratio = new_log_prob.detach() - sampled_log_prob.detach()
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence.item())

                # Compute entropy loss
                entropy_loss = 0
                if self._entropy_loss_scale > 0:
                    entropy = self.policy.get_entropy()
                    entropy_loss = -self._entropy_loss_scale * entropy.mean()

                # Compute policy loss (vanilla policy gradient with advantages)
                policy_loss = -(sampled_advantages * new_log_prob).mean()

                # Compute value loss
                value_outputs = self.value.act(states_dict)
                predicted_values = value_outputs.get(
                    "values", value_outputs.get("actions")
                )

                value_loss = F.mse_loss(sampled_returns, predicted_values)

            # Optimization step
            self.optimizer.zero_grad()
            total_loss = policy_loss + entropy_loss + value_loss
            self.scaler.scale(total_loss).backward()

            # Gradient clipping
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
                    kl_mean = float(np.mean(kl_divergences))
                    self.scheduler.step(kl_mean)
                    logger.debug(f"KL adaptive LR step: KL={kl_mean:.6f}, LR={self.scheduler.get_last_lr()[0]:.6f}")
                else:
                    self.scheduler.step()
            elif hasattr(self.scheduler, "step"):
                # Regular scheduler
                self.scheduler.step()

        # Log training metrics
        total_batches = self._mini_batches
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
        if hasattr(self, '_last_grad_norm'):
            self.track_data("Training/Gradient_Norm", self._last_grad_norm)
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
        
        # Policy statistics (if policy supports it)
        if hasattr(self.policy, 'distribution') and callable(getattr(self.policy, 'distribution', None)):
            try:
                dist = self.policy.distribution()
                if hasattr(dist, 'stddev'):
                    self.track_data("Policy/Action_Std", dist.stddev.mean().item())
            except:
                pass  # Skip if not available
        
        logger.info(f"A2C Update - Policy Loss: {avg_policy_loss:.6f}, Value Loss: {avg_value_loss:.6f}, "
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
        # Ensure all tensors have proper shape
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

        # Ensure proper tensor shapes
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
        logger.info(f"A2C model saved to {path}")

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
        logger.info(f"A2C model loaded from {path}")