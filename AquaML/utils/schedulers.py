"""
Learning rate schedulers for AquaML
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class KLAdaptiveLR(_LRScheduler):
    """KL-divergence adaptive learning rate scheduler
    
    Adjusts learning rate based on KL divergence between old and new policy.
    If KL divergence is too high, decrease learning rate.
    If KL divergence is too low, increase learning rate.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        kl_target: float = 0.01,
        kl_factor: float = 2.0,
        lr_min: float = 1e-6,
        lr_max: float = 1e-2,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            kl_target: Target KL divergence
            kl_factor: Factor for adjusting learning rate
            lr_min: Minimum learning rate
            lr_max: Maximum learning rate
            last_epoch: Last epoch index
        """
        self.kl_target = kl_target
        self.kl_factor = kl_factor
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.kl_value = None
        super().__init__(optimizer, last_epoch)
    
    def step(self, kl_value: Optional[float] = None):
        """Step with KL divergence value"""
        if kl_value is not None:
            self.kl_value = kl_value
        super().step()
    
    def get_lr(self):
        """Calculate new learning rates based on KL divergence"""
        if self.kl_value is None:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        new_lrs = []
        for group in self.optimizer.param_groups:
            current_lr = group['lr']
            
            if self.kl_value > self.kl_target:
                # KL too high, decrease learning rate
                new_lr = current_lr / self.kl_factor
            elif self.kl_value < self.kl_target / 2:
                # KL too low, increase learning rate
                new_lr = current_lr * self.kl_factor
            else:
                # KL in acceptable range, keep current lr
                new_lr = current_lr
            
            # Clamp to bounds
            new_lr = max(self.lr_min, min(self.lr_max, new_lr))
            new_lrs.append(new_lr)
        
        return new_lrs


class LinearWarmupScheduler(_LRScheduler):
    """Linear warmup scheduler followed by another scheduler"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        base_scheduler: Optional[_LRScheduler] = None,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            base_scheduler: Base scheduler to use after warmup
            warmup_start_lr: Starting learning rate for warmup
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rates with warmup"""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Use base scheduler if available
            if self.base_scheduler is not None:
                return self.base_scheduler.get_lr()
            else:
                return self.base_lrs
    
    def step(self, epoch=None):
        """Step both this scheduler and base scheduler"""
        super().step(epoch)
        if self.last_epoch >= self.warmup_steps and self.base_scheduler is not None:
            self.base_scheduler.step(epoch)