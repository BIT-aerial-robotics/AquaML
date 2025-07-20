from typing import Dict, Any, Optional, Tuple, Union
import copy
import itertools
import math
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
class RPOCfg:
    """Configuration class for RPO (Robust Policy Optimization) algorithm"""
    
    # Rollout and learning parameters
    rollouts: int = 16  # number of rollouts before updating
    learning_epochs: int = 8  # number of learning epochs during each update
    mini_batches: int = 2  # number of mini batches during each learning epoch

    # Robustness parameter
    alpha: float = 0.5  # amount of uniform random perturbation on the mean actions: U(-alpha, alpha)

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
    ratio_clip: float = 0.2  # clipping coefficient for computing the clipped surrogate objective
    value_clip: float = 0.2  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    clip_predicted_values: bool = False  # clip predicted values during value loss computation

    # Loss scaling parameters
    entropy_loss_scale: float = 0.0  # entropy loss scaling factor
    value_loss_scale: float = 1.0  # value loss scaling factor

    # KL divergence threshold
    kl_threshold: float = 0  # KL divergence threshold for early stopping

    # Rewards shaping and time limit bootstrap
    rewards_shaper: Optional[Any] = None  # rewards shaping function
    time_limit_bootstrap: bool = False  # bootstrap at timeout termination (episode truncation)

    # Mixed precision training
    mixed_precision: bool = False  # enable automatic mixed precision for higher performance

    # Memory and buffer parameters
    memory_size: int = 10000  # size of the replay buffer
    device: str = "auto"  # device to use for training


class RPO(Agent):
    """Robust Policy Optimization (RPO) algorithm implementation for AquaML
    
    Based on: https://arxiv.org/abs/2212.07536
    
    RPO learns robust policies by training against worst-case perturbations,
    making the policy more resilient to uncertainties and disturbances.
    """

    @coordinator.registerAgent
    def __init__(
        self,
        models: Dict[str, Model],
        cfg: RPOCfg,
        observation_space: Optional[Dict[str, Any]] = None,
        action_space: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RPO agent

        Args:
            models: Dictionary containing policy and value models
            cfg: RPO configuration
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

        logger.info(f"RPO initialized with device: {self.device}")

        # Models
        self.policy = models.get("policy")
        self.value = models.get("value")

        if self.policy is None:
            raise ValueError("Policy model is required for RPO")
        if self.value is None:
            raise ValueError("Value model is required for RPO")

        # Move models to device
        self.policy.to(self.device)
        self.value.to(self.device)

        # Setup memory
        memory_cfg = SequentialMemoryCfg(
            memory_size=cfg.memory_size,
            device=self.device,
            num_envs=None  # Will be determined from environment info
        )
        self.memory = SequentialMemory(memory_cfg)

        # Configuration parameters
        self._learning_epochs = cfg.learning_epochs
        self._mini_batches = cfg.mini_batches
        self._rollouts = cfg.rollouts
        self._rollout = 0

        self._alpha = cfg.alpha
        self._discount_factor = cfg.discount_factor
        self._lambda = cfg.lambda_value

        self._learning_rate = cfg.learning_rate
        self._learning_rate_scheduler = cfg.learning_rate_scheduler
        self._learning_rate_scheduler_kwargs = cfg.learning_rate_scheduler_kwargs or {}

        self._random_timesteps = cfg.random_timesteps
        self._learning_starts = cfg.learning_starts

        self._grad_norm_clip = cfg.grad_norm_clip
        self._ratio_clip = cfg.ratio_clip
        self._value_clip = cfg.value_clip
        self._clip_predicted_values = cfg.clip_predicted_values

        self._entropy_loss_scale = cfg.entropy_loss_scale
        self._value_loss_scale = cfg.value_loss_scale
        self._kl_threshold = cfg.kl_threshold

        self._rewards_shaper = cfg.rewards_shaper
        self._time_limit_bootstrap = cfg.time_limit_bootstrap
        self._mixed_precision = cfg.mixed_precision

        # Setup automatic mixed precision
        if torch.__version__ >= "2.4":
            self.scaler = torch.amp.GradScaler(device=self.device, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # Setup optimizer
        if self.policy is self.value:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
        else:
            self.optimizer = torch.optim.Adam(
                itertools.chain(self.policy.parameters(), self.value.parameters()),
                lr=self._learning_rate
            )

        # Setup learning rate scheduler
        if self._learning_rate_scheduler is not None:
            self.scheduler = self._learning_rate_scheduler(
                self.optimizer, **self._learning_rate_scheduler_kwargs
            )
        else:
            self.scheduler = None

        # Initialize memory tensors
        self._setup_memory()

        # Temporary variables for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

        logger.info(f"RPO agent initialized successfully with alpha={self._alpha}")

    def _setup_memory(self):
        """Setup memory tensors for storing transitions"""
        # Create basic tensors - shapes will be determined dynamically
        self.memory.create_tensor("states", size=1, dtype=torch.float32)
        self.memory.create_tensor("actions", size=1, dtype=torch.float32)
        self.memory.create_tensor("rewards", size=1, dtype=torch.float32)
        self.memory.create_tensor("terminated", size=1, dtype=torch.bool)
        self.memory.create_tensor("truncated", size=1, dtype=torch.bool)
        self.memory.create_tensor("log_prob", size=1, dtype=torch.float32)
        self.memory.create_tensor("values", size=1, dtype=torch.float32)
        self.memory.create_tensor("returns", size=1, dtype=torch.float32)
        self.memory.create_tensor("advantages", size=1, dtype=torch.float32)

        # Tensor names for sampling
        self._tensor_names = ["states", "actions", "log_prob", "values", "returns", "advantages"]

    def act(self, observations: Dict[str, torch.Tensor], timestep: int, timesteps: int) -> Dict[str, torch.Tensor]:
        """Process the environment's observations to make a decision (actions)

        Args:
            observations: Dictionary of environment observations
            timestep: Current timestep
            timesteps: Total number of timesteps

        Returns:
            Dictionary containing actions and additional outputs
        """
        # Sample random actions during initial exploration
        if timestep < self._random_timesteps:
            # For random actions, we'll call the policy's random action method
            with torch.no_grad():
                random_output = self.policy.act(observations)
                # Add alpha to the random output for consistency
                random_output["alpha"] = torch.tensor(self._alpha, device=self.device)
                return random_output

        # Sample actions using policy with robustness perturbation
        with torch.autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cpu', 
                           enabled=self._mixed_precision):
            # Add alpha parameter for robust policy
            robust_observations = observations.copy()
            robust_observations["alpha"] = torch.tensor(self._alpha, device=self.device)
            
            actions_output = self.policy.act(robust_observations)
            
            # Store log probabilities for training
            if "log_prob" in actions_output:
                self._current_log_prob = actions_output["log_prob"]
            
            return actions_output

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
        """Record an environment transition in memory

        Args:
            states: Current environment states
            actions: Actions taken by the agent
            rewards: Rewards received
            next_states: Next environment states
            terminated: Episode termination flags
            truncated: Episode truncation flags
            infos: Additional environment information
            timestep: Current timestep
            timesteps: Total number of timesteps
        """
        # Store next states for value computation
        self._current_next_states = next_states

        # Apply rewards shaping if configured
        if self._rewards_shaper is not None:
            rewards = self._rewards_shaper(rewards, timestep, timesteps)

        # Compute values for current states
        with torch.autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cpu',
                           enabled=self._mixed_precision):
            # Add alpha for robust value estimation
            robust_states = states.copy()
            robust_states["alpha"] = torch.tensor(self._alpha, device=self.device)
            
            value_output = self.value.act(robust_states)
            values = value_output.get("values", value_output.get("value", torch.tensor(0.0)))

        # Time-limit (truncation) bootstrapping
        if self._time_limit_bootstrap:
            rewards = rewards + self._discount_factor * values * truncated

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

    def pre_interaction(self, timestep: int, timesteps: int):
        """Callback called before the interaction with the environment"""
        pass

    def post_interaction(self, timestep: int, timesteps: int):
        """Callback called after the interaction with the environment"""
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        discount_factor: float = 0.99,
        lambda_coefficient: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the Generalized Advantage Estimator (GAE)

        Args:
            rewards: Rewards obtained by the agent
            dones: Signals to indicate that episodes have ended
            values: Values obtained by the agent
            next_values: Next values obtained by the agent
            discount_factor: Discount factor
            lambda_coefficient: Lambda coefficient

        Returns:
            Tuple of (returns, advantages)
        """
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
                + discount_factor * not_dones[i] * (next_value + lambda_coefficient * advantage)
            )
            advantages[i] = advantage

        # Returns computation
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def _update(self, timestep: int, timesteps: int):
        """Algorithm's main update step with robust policy optimization

        Args:
            timestep: Current timestep
            timesteps: Total number of timesteps
        """
        # Compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cpu',
                                           enabled=self._mixed_precision):
            self.value.train(False)
            
            # Add alpha for robust last value estimation
            robust_next_states = self._current_next_states.copy()
            robust_next_states["alpha"] = torch.tensor(self._alpha, device=self.device)
            
            last_value_output = self.value.act(robust_next_states)
            last_values = last_value_output.get("values", last_value_output.get("value", torch.tensor(0.0)))
            
            self.value.train(True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = self._compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("returns", returns)
        self.memory.set_tensor_by_name("advantages", advantages)

        # Sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensor_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # Learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # Mini-batches loop
            for batch_data in sampled_batches:
                (
                    sampled_states,
                    sampled_actions,
                    sampled_log_prob,
                    sampled_values,
                    sampled_returns,
                    sampled_advantages,
                ) = batch_data

                with torch.autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cpu',
                                   enabled=self._mixed_precision):

                    # Add alpha for robust policy evaluation
                    robust_states = sampled_states.copy() if isinstance(sampled_states, dict) else {"state": sampled_states}
                    robust_states["alpha"] = torch.tensor(self._alpha, device=self.device)
                    robust_states["taken_actions"] = sampled_actions

                    # Get current policy output
                    policy_output = self.policy.act(robust_states)
                    next_log_prob = policy_output.get("log_prob", torch.tensor(0.0))

                    # Compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # Early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # Compute entropy loss
                    if self._entropy_loss_scale:
                        entropy = policy_output.get("entropy", torch.tensor(0.0))
                        entropy_loss = -self._entropy_loss_scale * entropy.mean()
                    else:
                        entropy_loss = 0

                    # Compute policy loss (robust PPO objective)
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clamp(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # Compute value loss
                    robust_value_states = sampled_states.copy() if isinstance(sampled_states, dict) else {"state": sampled_states}
                    robust_value_states["alpha"] = torch.tensor(self._alpha, device=self.device)
                    
                    value_output = self.value.act(robust_value_states)
                    predicted_values = value_output.get("values", value_output.get("value", torch.tensor(0.0)))

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clamp(
                            predicted_values - sampled_values, 
                            min=-self._value_clip, 
                            max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # Optimization step
                self.optimizer.zero_grad()
                total_loss = policy_loss + entropy_loss + value_loss
                self.scaler.scale(total_loss).backward()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()),
                            self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Update cumulative losses
                cumulative_policy_loss += policy_loss.item() if isinstance(policy_loss, torch.Tensor) else 0
                cumulative_value_loss += value_loss.item() if isinstance(value_loss, torch.Tensor) else 0
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else 0

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # Record training metrics
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches))

        # Record RPO-specific metrics
        self.track_data("RPO / Alpha perturbation", self._alpha)
        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

        # Clear memory after update
        self.memory.clear()

        logger.debug(f"RPO update completed at timestep {timestep}")

    def save(self, path: str):
        """Save the agent's models and configuration

        Args:
            path: Path to save the agent
        """
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "cfg": self.cfg,
            "rollout": self._rollout,
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
        torch.save(checkpoint, path)
        logger.info(f"RPO agent saved to {path}")

    def load(self, path: str):
        """Load the agent's models and configuration

        Args:
            path: Path to load the agent from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value.load_state_dict(checkpoint["value_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._rollout = checkpoint.get("rollout", 0)
        
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        logger.info(f"RPO agent loaded from {path}")

    def set_mode(self, mode: str):
        """Set the agent's mode (train/eval)

        Args:
            mode: Mode to set ("train" or "eval")
        """
        if mode == "train":
            self.policy.train()
            self.value.train()
        elif mode == "eval":
            self.policy.eval()
            self.value.eval()
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'eval'")