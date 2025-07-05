from AquaML.config import configclass
from dataclasses import MISSING
from typing import Optional, Dict, Any, Callable

from AquaML.rl_algo.base import BaseRLAgent

@configclass
class PPOCfg:
    # Rollout and learning parameters
    rollouts: int = 16  # number of rollouts before updating
    learning_epochs: int = 8  # number of learning epochs during each update
    mini_batches: int = 2  # number of mini batches during each learning epoch

    # Discount and lambda parameters
    discount_factor: float = 0.99  # discount factor (gamma)
    lambda_value: float = 0.95  # TD(lambda) coefficient (lam) for computing returns and advantages

    # Learning rate parameters
    learning_rate: float = 1e-3  # learning rate
    learning_rate_scheduler: Optional[Any] = None  # learning rate scheduler class (see torch.optim.lr_scheduler)
    learning_rate_scheduler_kwargs: Dict[str, Any] = None  # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    # Preprocessor parameters
    state_preprocessor: Optional[Any] = None  # state preprocessor class (see skrl.resources.preprocessors)
    state_preprocessor_kwargs: Dict[str, Any] = None  # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    value_preprocessor: Optional[Any] = None  # value preprocessor class (see skrl.resources.preprocessors)
    value_preprocessor_kwargs: Dict[str, Any] = None  # value preprocessor's kwargs (e.g. {"size": 1})

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

    
    
class PPO(BaseRLAgent):
    pass