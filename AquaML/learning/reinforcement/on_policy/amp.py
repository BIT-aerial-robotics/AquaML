from typing import Dict, Any, Optional, Tuple, Union, Callable
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
class AMPCfg:
    """Configuration class for AMP (Adversarial Motion Priors) algorithm"""
    
    # Rollout and learning parameters
    rollouts: int = 16  # number of rollouts before updating
    learning_epochs: int = 6  # number of learning epochs during each update
    mini_batches: int = 2  # number of mini batches during each learning epoch

    # Discount and lambda parameters
    discount_factor: float = 0.99  # discount factor (gamma)
    lambda_value: float = 0.95  # TD(lambda) coefficient (lam) for computing returns and advantages

    # Learning rate parameters
    learning_rate: float = 5e-5  # learning rate
    learning_rate_scheduler: Optional[Any] = None  # learning rate scheduler class
    learning_rate_scheduler_kwargs: Optional[Dict[str, Any]] = None  # learning rate scheduler's kwargs

    # Preprocessor parameters
    state_preprocessor: Optional[Any] = None  # state preprocessor class
    state_preprocessor_kwargs: Optional[Dict[str, Any]] = None  # state preprocessor's kwargs
    value_preprocessor: Optional[Any] = None  # value preprocessor class
    value_preprocessor_kwargs: Optional[Dict[str, Any]] = None  # value preprocessor's kwargs
    amp_state_preprocessor: Optional[Any] = None  # AMP state preprocessor class
    amp_state_preprocessor_kwargs: Optional[Dict[str, Any]] = None  # AMP state preprocessor's kwargs

    # Exploration and learning start parameters
    random_timesteps: int = 0  # random exploration steps
    learning_starts: int = 0  # learning starts after this many steps

    # Clipping parameters
    grad_norm_clip: float = 0.0  # clipping coefficient for the norm of the gradients
    ratio_clip: float = 0.2  # clipping coefficient for computing the clipped surrogate objective
    value_clip: float = 0.2  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    clip_predicted_values: bool = False  # clip predicted values during value loss computation

    # Loss scaling parameters
    entropy_loss_scale: float = 0.0  # entropy loss scaling factor
    value_loss_scale: float = 2.5  # value loss scaling factor
    discriminator_loss_scale: float = 5.0  # discriminator loss scaling factor

    # AMP specific parameters
    amp_batch_size: int = 512  # batch size for updating the reference motion dataset
    task_reward_weight: float = 0.0  # task-reward weight (wG)
    style_reward_weight: float = 1.0  # style-reward weight (wS)
    discriminator_batch_size: int = 0  # batch size for computing the discriminator loss (all samples if 0)
    discriminator_reward_scale: float = 2  # discriminator reward scaling factor
    discriminator_logit_regularization_scale: float = 0.05  # logit regularization scale factor for the discriminator loss
    discriminator_gradient_penalty_scale: float = 5  # gradient penalty scaling factor for the discriminator loss
    discriminator_weight_decay_scale: float = 0.0001  # weight decay scaling factor for the discriminator loss

    # Rewards shaping and time limit bootstrap
    rewards_shaper: Optional[Any] = None  # rewards shaping function
    time_limit_bootstrap: bool = False  # bootstrap at timeout termination (episode truncation)

    # Mixed precision training
    mixed_precision: bool = False  # enable automatic mixed precision for higher performance

    # Memory and buffer parameters
    memory_size: int = 10000  # size of the replay buffer
    motion_dataset_size: int = 10000  # size of the motion dataset
    reply_buffer_size: int = 10000  # size of the reply buffer
    device: str = "auto"  # device to use for training


class AMP(Agent):
    """Adversarial Motion Priors (AMP) algorithm implementation for AquaML
    
    Based on: https://arxiv.org/abs/2104.02180
    
    AMP learns robust and naturalistic behaviors by combining task rewards with
    style rewards from a motion discriminator trained on reference motion data.
    """

    @coordinator.registerAgent
    def __init__(
        self,
        models: Dict[str, Model],
        cfg: AMPCfg,
        observation_space: Optional[Dict[str, Any]] = None,
        action_space: Optional[Dict[str, Any]] = None,
        amp_observation_space: Optional[Dict[str, Any]] = None,
        motion_dataset: Optional[Any] = None,
        reply_buffer: Optional[Any] = None,
        collect_reference_motions: Optional[Callable[[int], torch.Tensor]] = None,
        collect_observation: Optional[Callable[[], torch.Tensor]] = None,
    ):
        """
        Initialize AMP agent

        Args:
            models: Dictionary containing policy, value, and discriminator models
            cfg: AMP configuration
            observation_space: Dictionary describing observation space
            action_space: Dictionary describing action space
            amp_observation_space: Dictionary describing AMP observation space
            motion_dataset: Reference motion dataset
            reply_buffer: Reply buffer for preventing discriminator overfitting
            collect_reference_motions: Callable to collect reference motions
            collect_observation: Callable to collect observation
        """
        super().__init__(models, None, observation_space, action_space, cfg.device, cfg)

        self.cfg = cfg
        self.amp_observation_space = amp_observation_space

        # Device setup
        if cfg.device == "auto":
            self.device = coordinator.get_device()
        else:
            self.device = cfg.device

        logger.info(f"AMP initialized with device: {self.device}")

        # Models
        self.policy = models.get("policy")
        self.value = models.get("value")
        self.discriminator = models.get("discriminator")

        if self.policy is None:
            raise ValueError("Policy model is required for AMP")
        if self.value is None:
            raise ValueError("Value model is required for AMP")
        if self.discriminator is None:
            raise ValueError("Discriminator model is required for AMP")

        # Move models to device
        self.policy.to(self.device)
        self.value.to(self.device)
        self.discriminator.to(self.device)

        # AMP specific components
        self.motion_dataset = motion_dataset
        self.reply_buffer = reply_buffer
        self.collect_reference_motions = collect_reference_motions
        self.collect_observation = collect_observation

        # Setup memory for main rollouts
        memory_cfg = SequentialMemoryCfg(
            memory_size=cfg.memory_size,
            device=self.device,
            num_envs=None  # Will be determined from environment info
        )
        self.memory = SequentialMemory(memory_cfg)

        # Setup motion dataset memory
        if self.motion_dataset is None and cfg.motion_dataset_size > 0:
            motion_dataset_cfg = SequentialMemoryCfg(
                memory_size=cfg.motion_dataset_size,
                device=self.device,
                num_envs=1
            )
            self.motion_dataset = SequentialMemory(motion_dataset_cfg)

        # Setup reply buffer memory
        if self.reply_buffer is None and cfg.reply_buffer_size > 0:
            reply_buffer_cfg = SequentialMemoryCfg(
                memory_size=cfg.reply_buffer_size,
                device=self.device,
                num_envs=1
            )
            self.reply_buffer = SequentialMemory(reply_buffer_cfg)

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
        self._discriminator_loss_scale = cfg.discriminator_loss_scale

        self._learning_rate = cfg.learning_rate
        self._learning_rate_scheduler = cfg.learning_rate_scheduler

        self._discount_factor = cfg.discount_factor
        self._lambda = cfg.lambda_value

        self._random_timesteps = cfg.random_timesteps
        self._learning_starts = cfg.learning_starts

        # AMP specific parameters
        self._amp_batch_size = cfg.amp_batch_size
        self._task_reward_weight = cfg.task_reward_weight
        self._style_reward_weight = cfg.style_reward_weight

        self._discriminator_batch_size = cfg.discriminator_batch_size
        self._discriminator_reward_scale = cfg.discriminator_reward_scale
        self._discriminator_logit_regularization_scale = cfg.discriminator_logit_regularization_scale
        self._discriminator_gradient_penalty_scale = cfg.discriminator_gradient_penalty_scale
        self._discriminator_weight_decay_scale = cfg.discriminator_weight_decay_scale

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
        self._amp_state_preprocessor = None
        
        if cfg.state_preprocessor is not None:
            self._state_preprocessor = cfg.state_preprocessor(**(cfg.state_preprocessor_kwargs or {}))
            logger.info("State preprocessor initialized")
        
        if cfg.value_preprocessor is not None:
            self._value_preprocessor = cfg.value_preprocessor(**(cfg.value_preprocessor_kwargs or {}))
            logger.info("Value preprocessor initialized")
            
        if cfg.amp_state_preprocessor is not None:
            self._amp_state_preprocessor = cfg.amp_state_preprocessor(**(cfg.amp_state_preprocessor_kwargs or {}))
            logger.info("AMP state preprocessor initialized")

        # Setup optimizer for all three models
        self.optimizer = torch.optim.Adam(
            itertools.chain(
                self.policy.parameters(), 
                self.value.parameters(), 
                self.discriminator.parameters()
            ),
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
        self._current_states = None

        # Register checkpoint modules
        self.register_checkpoint_module("policy", self.policy)
        self.register_checkpoint_module("value", self.value)
        self.register_checkpoint_module("discriminator", self.discriminator)
        self.register_checkpoint_module("optimizer", self.optimizer)
        if self.scheduler is not None:
            self.register_checkpoint_module("scheduler", self.scheduler)

        logger.info("AMP agent initialized successfully")

    def _init_memory(self):
        """Initialize memory tensors"""
        # Main memory for rollouts
        self.memory.create_tensor("states", (1,), torch.float32)
        self.memory.create_tensor("actions", (1,), torch.float32)
        self.memory.create_tensor("rewards", (1,), torch.float32)
        self.memory.create_tensor("next_states", (1,), torch.float32)
        self.memory.create_tensor("terminated", (1,), torch.bool)
        self.memory.create_tensor("truncated", (1,), torch.bool)
        self.memory.create_tensor("log_prob", (1,), torch.float32)
        self.memory.create_tensor("values", (1,), torch.float32)
        self.memory.create_tensor("returns", (1,), torch.float32)
        self.memory.create_tensor("advantages", (1,), torch.float32)
        
        # AMP specific tensors
        self.memory.create_tensor("amp_states", (1,), torch.float32)
        self.memory.create_tensor("next_values", (1,), torch.float32)

        # Initialize motion dataset memory
        if self.motion_dataset is not None:
            self.motion_dataset.create_tensor("states", (1,), torch.float32)

        # Initialize reply buffer memory
        if self.reply_buffer is not None:
            self.reply_buffer.create_tensor("states", (1,), torch.float32)

        # Tensors used during training
        self._tensors_names = [
            "states",
            "actions",
            "rewards",
            "next_states",
            "terminated",
            "log_prob",
            "values",
            "returns",
            "advantages",
            "amp_states",
            "next_values",
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
        # Use collected states if available
        if self._current_states is not None:
            states = self._current_states

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
            # For random actions, would need to implement random_act in the policy
            # For now, use normal policy

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
        # Use collected states if available
        if self._current_states is not None:
            states = self._current_states

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

        # Extract AMP observations from infos
        amp_states = infos.get("amp_obs")
        if amp_states is None:
            raise ValueError("AMP observations not found in infos. Expected 'amp_obs' key.")
        amp_states = amp_states.to(self.device)

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

        # Compute next values
        next_states_for_value = next_states
        if self._state_preprocessor is not None:
            processed_next_states = {}
            for k, v in next_states.items():
                processed_next_states[k] = self._state_preprocessor(v)
            next_states_for_value = processed_next_states

        with torch.cuda.amp.autocast(enabled=self._mixed_precision):
            next_value_outputs = self.value.act(next_states_for_value)
            next_values = next_value_outputs.get(
                "values", next_value_outputs.get("actions")
            )
            
            # Apply value preprocessing if available
            if self._value_preprocessor is not None:
                next_values = self._value_preprocessor(next_values)

            # Handle termination for next values
            if "terminate" in infos:
                next_values *= infos["terminate"].view(-1, 1).logical_not()
            else:
                next_values *= terminated.view(-1, 1).logical_not()

        # Store transition in memory
        self.memory.add_samples(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            log_prob=self._current_log_prob,
            values=values,
            amp_states=amp_states,
            next_values=next_values,
        )

        logger.debug(f"Recorded transition at timestep {timestep}")

    def pre_interaction(self, timestep: int, timesteps: int):
        """Called before each environment interaction"""
        if self.collect_observation is not None:
            self._current_states = self.collect_observation()

    def post_interaction(self, timestep: int, timesteps: int):
        """Called after each environment interaction"""
        self._rollout += 1

        # Update policy after collecting enough rollouts
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            # Check if we have enough samples for update
            if not self.memory.is_ready_for_update(self._rollouts):
                logger.warning(f"Not enough samples for update at timestep {timestep}")
                return
                
            logger.debug(f"Starting AMP update at timestep {timestep}")
            
            # Set models to training mode
            if self.policy is not None:
                self.policy.train()
            if self.value is not None:
                self.value.train()
            if self.discriminator is not None:
                self.discriminator.train()
                
            # Perform update
            self._update(timestep, timesteps)
            
            # Set models back to eval mode
            if self.policy is not None:
                self.policy.eval()
            if self.value is not None:
                self.value.eval()
            if self.discriminator is not None:
                self.discriminator.eval()

    def _update(self, timestep: int, timesteps: int):
        """Main AMP update step"""
        # Update motion dataset if available
        if self.motion_dataset is not None and self.collect_reference_motions is not None:
            reference_motions = self.collect_reference_motions(self._amp_batch_size)
            if reference_motions is not None:
                self.motion_dataset.add_samples(states=reference_motions.to(self.device))

        # Compute combined rewards (task + style)
        self._compute_combined_rewards()

        # Compute GAE returns and advantages
        self._compute_gae()

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_discriminator_loss = 0

        # Learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # Sample batches for this epoch
            sampled_batches = self.memory.sample_all_shuffled(
                names=self._tensors_names, mini_batches=self._mini_batches
            )

            # Sample motion batches
            motion_batches = []
            if self.motion_dataset is not None:
                batch_size = self.memory.get_stored_samples_count()
                motion_batches = self.motion_dataset.sample_all_shuffled(
                    names=["states"], mini_batches=self._mini_batches, batch_size=batch_size
                )

            # Sample replay batches
            replay_batches = []
            if self.reply_buffer is not None and len(self.reply_buffer) > 0:
                batch_size = self.memory.get_stored_samples_count()
                replay_batches = self.reply_buffer.sample_all_shuffled(
                    names=["states"], mini_batches=self._mini_batches, batch_size=batch_size
                )

            # Mini-batch loop
            for batch_idx, batch in enumerate(sampled_batches):
                (
                    sampled_states,
                    sampled_actions,
                    _,  # rewards (not used directly in policy update)
                    _,  # next_states
                    _,  # terminated
                    sampled_log_prob,
                    sampled_values,
                    sampled_returns,
                    sampled_advantages,
                    sampled_amp_states,
                    _,  # next_values
                ) = batch

                # Detach all tensors to prevent gradient accumulation issues
                sampled_states = sampled_states.detach()
                sampled_actions = sampled_actions.detach()
                sampled_log_prob = sampled_log_prob.detach()
                sampled_values = sampled_values.detach()
                sampled_returns = sampled_returns.detach()
                sampled_advantages = sampled_advantages.detach()
                sampled_amp_states = sampled_amp_states.detach()

                with torch.cuda.amp.autocast(enabled=self._mixed_precision):
                    # Convert flattened states back to dictionary format
                    states_dict = self.memory._unflatten_dict_structured(sampled_states, "states")
                    actions_dict = self.memory._unflatten_dict_structured(sampled_actions, "actions")

                    # Get new log probabilities from policy
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

                    # Compute discriminator loss
                    discriminator_loss = self._compute_discriminator_loss(
                        sampled_amp_states, motion_batches, replay_batches, batch_idx
                    )

                # Optimization step
                self.optimizer.zero_grad()
                total_loss = policy_loss + entropy_loss + value_loss + discriminator_loss
                self.scaler.scale(total_loss).backward()

                # Gradient clipping
                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        itertools.chain(
                            self.policy.parameters(),
                            self.value.parameters(),
                            self.discriminator.parameters(),
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
                cumulative_discriminator_loss += discriminator_loss.item()

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

        # Update AMP replay buffer
        if self.reply_buffer is not None:
            amp_states = self.memory.get_tensor_by_name("amp_states")
            if amp_states is not None:
                self.reply_buffer.add_samples(states=amp_states.view(-1, amp_states.shape[-1]))

        # Log training metrics
        total_batches = self._learning_epochs * self._mini_batches
        avg_policy_loss = cumulative_policy_loss / total_batches
        avg_value_loss = cumulative_value_loss / total_batches
        avg_discriminator_loss = cumulative_discriminator_loss / total_batches
        
        # Core losses
        self.track_data("Loss/Policy", avg_policy_loss)
        self.track_data("Loss/Value", avg_value_loss)
        self.track_data("Loss/Discriminator", avg_discriminator_loss)
        self.track_data("Loss/Total", avg_policy_loss + avg_value_loss + avg_discriminator_loss)
        
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
        
        logger.info(f"AMP Update - Policy Loss: {avg_policy_loss:.6f}, Value Loss: {avg_value_loss:.6f}, "
                   f"Discriminator Loss: {avg_discriminator_loss:.6f}, Samples: {self.memory.get_stored_samples_count()}")

        # Clear memory for next rollout
        self.memory.clear()

    def _compute_combined_rewards(self):
        """Compute combined rewards from task and style components"""
        rewards = self.memory.get_tensor_by_name("rewards")
        amp_states = self.memory.get_tensor_by_name("amp_states")

        if amp_states is None:
            logger.warning("No AMP states found, using only task rewards")
            return

        # Compute style reward using discriminator
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self._mixed_precision):
            # Apply AMP state preprocessing if available
            amp_states_processed = amp_states
            if self._amp_state_preprocessor is not None:
                amp_states_processed = self._amp_state_preprocessor(amp_states)

            # Get discriminator logits
            amp_states_dict = {"states": amp_states_processed}
            discriminator_outputs = self.discriminator.act(amp_states_dict)
            amp_logits = discriminator_outputs.get("values", discriminator_outputs.get("actions"))

            # Compute style reward: -log(1 - sigmoid(logits))
            style_reward = -torch.log(
                torch.maximum(
                    1 - 1 / (1 + torch.exp(-amp_logits)), 
                    torch.tensor(0.0001, device=self.device)
                )
            )
            style_reward *= self._discriminator_reward_scale
            style_reward = style_reward.view(rewards.shape)

        # Combine task and style rewards
        combined_rewards = self._task_reward_weight * rewards + self._style_reward_weight * style_reward

        # Update rewards in memory
        self.memory.set_tensor_by_name("rewards", combined_rewards)
        
        # Track reward components
        self.track_data("Rewards/Task_Mean", rewards.mean().item())
        self.track_data("Rewards/Style_Mean", style_reward.mean().item())
        self.track_data("Rewards/Combined_Mean", combined_rewards.mean().item())

    def _compute_discriminator_loss(self, sampled_amp_states, motion_batches, replay_batches, batch_idx):
        """Compute discriminator loss with gradient penalty and regularization"""
        # Determine batch size for discriminator training
        if self._discriminator_batch_size > 0:
            disc_batch_size = min(self._discriminator_batch_size, sampled_amp_states.shape[0])
            sampled_amp_states = sampled_amp_states[:disc_batch_size]
        
        # Get motion and replay states for this batch
        if motion_batches and batch_idx < len(motion_batches):
            motion_states = motion_batches[batch_idx][0]
            if self._discriminator_batch_size > 0:
                motion_states = motion_states[:disc_batch_size]
        else:
            # Fallback: generate some motion states
            motion_states = sampled_amp_states  # This should be replaced with actual motion data
            
        if replay_batches and batch_idx < len(replay_batches):
            replay_states = replay_batches[batch_idx][0]
            if self._discriminator_batch_size > 0:
                replay_states = replay_states[:disc_batch_size]
        else:
            # Use current AMP states as replay states if replay buffer is empty
            replay_states = sampled_amp_states

        # Apply AMP state preprocessing
        if self._amp_state_preprocessor is not None:
            sampled_amp_states = self._amp_state_preprocessor(sampled_amp_states)
            motion_states = self._amp_state_preprocessor(motion_states)
            replay_states = self._amp_state_preprocessor(replay_states)

        # Enable gradient computation for motion states (needed for gradient penalty)
        motion_states.requires_grad_(True)

        # Get discriminator outputs
        amp_states_dict = {"states": sampled_amp_states}
        replay_states_dict = {"states": replay_states}
        motion_states_dict = {"states": motion_states}

        amp_outputs = self.discriminator.act(amp_states_dict)
        amp_logits = amp_outputs.get("values", amp_outputs.get("actions"))

        replay_outputs = self.discriminator.act(replay_states_dict)
        amp_replay_logits = replay_outputs.get("values", replay_outputs.get("actions"))

        motion_outputs = self.discriminator.act(motion_states_dict)
        amp_motion_logits = motion_outputs.get("values", motion_outputs.get("actions"))

        # Combine AMP and replay logits
        amp_cat_logits = torch.cat([amp_logits, amp_replay_logits], dim=0)

        # Discriminator prediction loss
        # Real data (motion) should have logits close to 1, fake data (amp) should have logits close to 0
        discriminator_loss = 0.5 * (
            nn.BCEWithLogitsLoss()(amp_cat_logits, torch.zeros_like(amp_cat_logits))
            + nn.BCEWithLogitsLoss()(amp_motion_logits, torch.ones_like(amp_motion_logits))
        )

        # Discriminator logit regularization
        if self._discriminator_logit_regularization_scale > 0:
            # Find the last linear layer for logit regularization
            last_linear = None
            for module in reversed(list(self.discriminator.modules())):
                if isinstance(module, nn.Linear):
                    last_linear = module
                    break
            
            if last_linear is not None:
                logit_weights = torch.flatten(last_linear.weight)
                discriminator_loss += self._discriminator_logit_regularization_scale * torch.sum(
                    torch.square(logit_weights)
                )

        # Discriminator gradient penalty
        if self._discriminator_gradient_penalty_scale > 0:
            amp_motion_gradient = torch.autograd.grad(
                amp_motion_logits,
                motion_states,
                grad_outputs=torch.ones_like(amp_motion_logits),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            gradient_penalty = torch.sum(torch.square(amp_motion_gradient[0]), dim=-1).mean()
            discriminator_loss += self._discriminator_gradient_penalty_scale * gradient_penalty

        # Discriminator weight decay
        if self._discriminator_weight_decay_scale > 0:
            weights = []
            for module in self.discriminator.modules():
                if isinstance(module, nn.Linear):
                    weights.append(torch.flatten(module.weight))
            if weights:
                weight_decay = torch.sum(torch.square(torch.cat(weights, dim=-1)))
                discriminator_loss += self._discriminator_weight_decay_scale * weight_decay

        # Scale discriminator loss
        discriminator_loss *= self._discriminator_loss_scale

        return discriminator_loss

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
        next_values = self.memory.get_tensor_by_name("next_values")
        terminated = self.memory.get_tensor_by_name("terminated")
        truncated = self.memory.get_tensor_by_name("truncated")
        dones = terminated | truncated

        # Use next_values from memory instead of last_values for better accuracy
        returns, advantages = self._compute_gae_values(
            rewards, dones, values, next_values, self._discount_factor, self._lambda
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
            advantage = (
                rewards[i]
                - values[i]
                + discount_factor * not_dones[i] * (next_values[i] + lambda_coeff * advantage)
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
                "discriminator_state_dict": self.discriminator.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "cfg": self.cfg,
            },
            path,
        )
        logger.info(f"AMP model saved to {path}")

    def load(self, path: str):
        """Load model parameters"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "policy" in checkpoint:
            self.policy.load_state_dict(checkpoint["policy"])
            self.value.load_state_dict(checkpoint["value"])
            self.discriminator.load_state_dict(checkpoint["discriminator"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.value.load_state_dict(checkpoint["value_state_dict"])
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"AMP model loaded from {path}")