from typing import Dict, Any, Optional, Tuple, Union
import copy
import itertools
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
import numpy as np
from loguru import logger

from AquaML import coordinator
from AquaML.learning.model import Model
from AquaML.learning.reinforcement.base import Agent
from AquaML.learning.memory import SequentialMemory, SequentialMemoryCfg
from AquaML.config import configclass


@configclass
class TRPOCfg:
    """Configuration for Trust Region Policy Optimization (TRPO)"""
    
    # Rollout and learning parameters
    rollouts: int = 16  # number of rollouts before updating
    learning_epochs: int = 8  # number of learning epochs during each update (for value function)
    mini_batches: int = 2  # number of mini batches during each learning epoch
    
    # Discount and lambda parameters
    discount_factor: float = 0.99  # discount factor (gamma)
    lambda_value: float = 0.95  # TD(lambda) coefficient (lam) for computing returns and advantages
    
    # Value function learning rate (policy is updated via trust region method)
    value_learning_rate: float = 1e-3  # value learning rate
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
    
    # Gradient clipping and loss scaling
    grad_norm_clip: float = 0.5  # clipping coefficient for the norm of the gradients
    value_loss_scale: float = 1.0  # value loss scaling factor
    
    # TRPO-specific parameters
    damping: float = 0.1  # damping coefficient for computing the Hessian-vector product
    max_kl_divergence: float = 0.01  # maximum KL divergence between old and new policy
    conjugate_gradient_steps: int = 10  # maximum number of iterations for the conjugate gradient algorithm
    max_backtrack_steps: int = 10  # maximum number of backtracking steps during line search
    accept_ratio: float = 0.5  # accept ratio for the line search loss improvement
    step_fraction: float = 1.0  # fraction of the step size for the line search
    
    # Rewards shaping and time limit bootstrap
    rewards_shaper: Optional[Any] = None  # rewards shaping function
    time_limit_bootstrap: bool = False  # bootstrap at timeout termination (episode truncation)
    
    # Mixed precision training
    mixed_precision: bool = False  # enable automatic mixed precision for higher performance
    
    # Memory and buffer parameters
    memory_size: int = 10000  # size of the replay buffer
    device: str = "auto"  # device to use for training


class TRPO(Agent):
    """Trust Region Policy Optimization (TRPO) algorithm implementation for AquaML
    
    This implementation adapts the SKRL TRPO algorithm to the AquaML framework,
    following the dictionary-based architecture for flexible model inputs.
    
    Reference: Trust Region Policy Optimization (https://arxiv.org/abs/1502.05477)
    """

    @coordinator.registerAgent
    def __init__(
        self,
        models: Dict[str, Model],
        cfg: TRPOCfg,
        observation_space: Optional[Dict[str, Any]] = None,
        action_space: Optional[Dict[str, Any]] = None,
    ):
        """Initialize TRPO agent
        
        Args:
            models: Dictionary containing policy and value models
            cfg: TRPO configuration
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
            
        logger.info(f"TRPO initialized with device: {self.device}")
        
        # Models
        self.policy = models.get("policy")
        self.value = models.get("value")
        
        if self.policy is None:
            raise ValueError("Policy model is required for TRPO")
        if self.value is None:
            raise ValueError("Value model is required for TRPO")
            
        # Move models to device
        self.policy.to(self.device)
        self.value.to(self.device)
        
        # Create backup policy for line search
        self.backup_policy = copy.deepcopy(self.policy)
        
        # Setup memory - support multi-environment rollouts
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
        
        self._grad_norm_clip = cfg.grad_norm_clip
        self._value_loss_scale = cfg.value_loss_scale
        
        # TRPO-specific parameters
        self._max_kl_divergence = cfg.max_kl_divergence
        self._damping = cfg.damping
        self._conjugate_gradient_steps = cfg.conjugate_gradient_steps
        self._max_backtrack_steps = cfg.max_backtrack_steps
        self._accept_ratio = cfg.accept_ratio
        self._step_fraction = cfg.step_fraction
        
        self._value_learning_rate = cfg.value_learning_rate
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
            self.scaler = torch.amp.GradScaler(device_type='cuda', enabled=self._mixed_precision)
        except TypeError:
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
        
        # Setup value optimizer (policy is updated via trust region method)
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=self._value_learning_rate
        )
        
        # Setup learning rate scheduler
        if self._learning_rate_scheduler is not None:
            self.scheduler = self._learning_rate_scheduler(
                self.value_optimizer, **cfg.learning_rate_scheduler_kwargs or {}
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
        self.register_checkpoint_module("backup_policy", self.backup_policy)
        self.register_checkpoint_module("value_optimizer", self.value_optimizer)
        if self.scheduler is not None:
            self.register_checkpoint_module("scheduler", self.scheduler)
        
        logger.info("TRPO agent initialized successfully")
    
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
        self._tensors_names_policy = ["states", "actions", "log_prob", "advantages"]
        self._tensors_names_value = ["states", "returns"]
    
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
            values = value_outputs.get("values", value_outputs.get("actions"))
            
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
            
            logger.debug(f"Starting TRPO update at timestep {timestep}")
            
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
        """Main TRPO update step"""
        # Compute returns and advantages (GAE already computed)
        
        # Sample all data for policy update (TRPO typically uses full batch)
        policy_batch = self.memory.sample_all(names=self._tensors_names_policy, mini_batches=1)[0]
        sampled_states, sampled_actions, sampled_log_prob, sampled_advantages = policy_batch
        
        # Convert states back to dictionary format
        states_dict = self.memory._unflatten_dict_structured(sampled_states, "states")
        actions_dict = self.memory._unflatten_dict_structured(sampled_actions, "actions")
        
        # Detach all tensors
        states_dict = {k: v.detach() for k, v in states_dict.items()}
        if isinstance(actions_dict, dict):
            actions_dict = {k: v.detach() for k, v in actions_dict.items()}
        else:
            actions_dict = actions_dict.detach()
        sampled_log_prob = sampled_log_prob.detach()
        sampled_advantages = sampled_advantages.detach()
        
        # Compute policy loss gradient using surrogate objective
        policy_loss = self._surrogate_loss(self.policy, states_dict, actions_dict, sampled_log_prob, sampled_advantages)
        policy_loss_gradient = torch.autograd.grad(policy_loss, self.policy.parameters())
        flat_policy_loss_gradient = torch.cat([grad.view(-1) for grad in policy_loss_gradient])
        
        # Compute search direction using conjugate gradient algorithm
        search_direction = self._conjugate_gradient(
            self.policy, states_dict, flat_policy_loss_gradient.data, self._conjugate_gradient_steps
        )
        
        # Compute step size and full step
        xHx = (search_direction * self._fisher_vector_product(
            self.policy, states_dict, search_direction, self._damping
        )).sum(0, keepdim=True)
        step_size = torch.sqrt(2 * self._max_kl_divergence / xHx)[0]
        full_step = step_size * search_direction
        
        # Backtracking line search
        restore_policy_flag = True
        self.backup_policy.load_state_dict(self.policy.state_dict())
        params = parameters_to_vector(self.policy.parameters())
        
        expected_improvement = (flat_policy_loss_gradient * full_step).sum(0, keepdim=True)
        
        for alpha in [self._step_fraction * 0.5**i for i in range(self._max_backtrack_steps)]:
            new_params = params + alpha * full_step
            vector_to_parameters(new_params, self.policy.parameters())
            
            expected_improvement *= alpha
            kl = self._kl_divergence(self.backup_policy, self.policy, states_dict)
            loss = self._surrogate_loss(self.policy, states_dict, actions_dict, sampled_log_prob, sampled_advantages)
            
            if kl < self._max_kl_divergence and (loss - policy_loss) / expected_improvement > self._accept_ratio:
                restore_policy_flag = False
                break
        
        if restore_policy_flag:
            self.policy.load_state_dict(self.backup_policy.state_dict())
            logger.warning("Line search failed, restoring policy parameters")
        
        # Update value function
        cumulative_value_loss = 0
        sampled_batches = self.memory.sample_all(names=self._tensors_names_value, mini_batches=self._mini_batches)
        
        for epoch in range(self._learning_epochs):
            for batch in sampled_batches:
                sampled_states_v, sampled_returns = batch
                
                # Convert states back to dictionary format
                states_dict_v = self.memory._unflatten_dict_structured(sampled_states_v, "states")
                
                with torch.cuda.amp.autocast(enabled=self._mixed_precision):
                    # Compute value loss
                    value_outputs = self.value.act(states_dict_v)
                    predicted_values = value_outputs.get("values", value_outputs.get("actions"))
                    
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)
                
                # Optimization step (value)
                self.value_optimizer.zero_grad()
                self.scaler.scale(value_loss).backward()
                
                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.value_optimizer)
                    nn.utils.clip_grad_norm_(self.value.parameters(), self._grad_norm_clip)
                
                self.scaler.step(self.value_optimizer)
                self.scaler.update()
                
                cumulative_value_loss += value_loss.item()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Log training metrics
        total_batches = self._learning_epochs * self._mini_batches
        avg_value_loss = cumulative_value_loss / total_batches
        
        self.track_data("Loss/Policy", policy_loss.item())
        self.track_data("Loss/Value", avg_value_loss)
        self.track_data("Training/Learning_Rate", self.value_optimizer.param_groups[0]['lr'])
        self.track_data("Training/Samples_Count", self.memory.get_stored_samples_count())
        
        # GAE statistics
        advantages_tensor = self.memory.get_tensor_by_name("advantages")
        if advantages_tensor is not None:
            self.track_data("GAE/Advantages_Mean", advantages_tensor.mean().item())
            self.track_data("GAE/Advantages_Std", advantages_tensor.std().item())
        
        logger.info(f"TRPO Update - Policy Loss: {policy_loss.item():.6f}, Value Loss: {avg_value_loss:.6f}")
        
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
                    last_values = last_value_outputs.get("values", last_value_outputs.get("actions"))
                    
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
    
    def _compute_gae_values(self, rewards, dones, values, next_values, discount_factor, lambda_coeff):
        """Compute GAE values"""
        # Ensure all tensors have correct shape
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
                + discount_factor * not_dones[i] * (next_value + lambda_coeff * advantage)
            )
            advantages[i] = advantage
        
        # Returns computation
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Ensure returned tensors have proper shape
        if returns.dim() != 1:
            returns = returns.squeeze()
        if advantages.dim() != 1:
            advantages = advantages.squeeze()
        
        return returns, advantages
    
    def _surrogate_loss(self, policy, states, actions, log_prob, advantages):
        """Compute the surrogate objective (policy loss)"""
        # Get new log probabilities
        policy_outputs = policy.act({**states, "taken_actions": actions})
        new_log_prob = policy_outputs.get("log_prob")
        return (advantages * torch.exp(new_log_prob - log_prob.detach())).mean()
    
    def _conjugate_gradient(self, policy, states, b, num_iterations=10, residual_tolerance=1e-10):
        """Conjugate gradient algorithm to solve Ax = b"""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rr_old = torch.dot(r, r)
        
        for _ in range(num_iterations):
            hv = self._fisher_vector_product(policy, states, p, self._damping)
            alpha = rr_old / torch.dot(p, hv)
            x += alpha * p
            r -= alpha * hv
            rr_new = torch.dot(r, r)
            if rr_new < residual_tolerance:
                break
            p = r + rr_new / rr_old * p
            rr_old = rr_new
        
        return x
    
    def _fisher_vector_product(self, policy, states, vector, damping=0.1):
        """Compute the Fisher vector product (direct method)"""
        kl = self._kl_divergence(policy, policy, states)
        kl_gradient = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
        flat_kl_gradient = torch.cat([gradient.view(-1) for gradient in kl_gradient])
        hessian_vector_gradient = torch.autograd.grad(
            (flat_kl_gradient * vector).sum(), policy.parameters()
        )
        flat_hessian_vector_gradient = torch.cat(
            [gradient.contiguous().view(-1) for gradient in hessian_vector_gradient]
        )
        return flat_hessian_vector_gradient + damping * vector
    
    def _kl_divergence(self, policy_1, policy_2, states):
        """Compute the KL divergence between two policy distributions"""
        # Get outputs from both policies
        outputs_1 = policy_1.act(states)
        outputs_2 = policy_2.act(states)
        
        # Extract mean and log_std - this assumes Gaussian policies
        # Adjust this based on your specific policy implementation
        mu_1 = outputs_1.get("mean_actions", outputs_1.get("actions"))
        mu_2 = outputs_2.get("mean_actions", outputs_2.get("actions"))
        
        # For Gaussian policies, get log_std
        if hasattr(policy_1, 'get_log_std'):
            logstd_1 = policy_1.get_log_std()
        else:
            logstd_1 = outputs_1.get("log_std", torch.zeros_like(mu_1))
        
        if hasattr(policy_2, 'get_log_std'):
            logstd_2 = policy_2.get_log_std()
        else:
            logstd_2 = outputs_2.get("log_std", torch.zeros_like(mu_2))
        
        mu_1, logstd_1 = mu_1.detach(), logstd_1.detach()
        
        # Compute KL divergence for Gaussian distributions
        kl = (
            logstd_1 - logstd_2
            + 0.5 * (torch.square(logstd_1.exp()) + torch.square(mu_1 - mu_2)) / torch.square(logstd_2.exp())
            - 0.5
        )
        return torch.sum(kl, dim=-1).mean()
    
    def save(self, path: str):
        """Save model parameters"""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "backup_policy_state_dict": self.backup_policy.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "cfg": self.cfg,
        }, path)
        logger.info(f"TRPO model saved to {path}")
    
    def load(self, path: str):
        """Load model parameters"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value.load_state_dict(checkpoint["value_state_dict"])
        self.backup_policy.load_state_dict(checkpoint["backup_policy_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        logger.info(f"TRPO model loaded from {path}")