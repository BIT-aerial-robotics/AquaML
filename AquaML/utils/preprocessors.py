"""
Data preprocessors for AquaML
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod


class Preprocessor(ABC):
    """Base class for data preprocessors"""
    
    @abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing to data"""
        pass


class RunningMeanStd(nn.Module):
    """Running mean and standard deviation for normalization"""
    
    def __init__(self, shape: tuple, epsilon: float = 1e-8, momentum: float = 0.99):
        """
        Args:
            shape: Shape of the data (excluding batch dimension)
            epsilon: Small value to avoid division by zero
            momentum: Momentum for running statistics
        """
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('count', torch.zeros(1))
        
    def update(self, data: torch.Tensor):
        """Update running statistics"""
        if not self.training:
            return
            
        batch_mean = data.mean(dim=0)
        batch_var = data.var(dim=0, unbiased=False)
        batch_count = data.shape[0]
        
        # Update running statistics
        total_count = self.count + batch_count
        
        if self.count == 0:
            self.mean.copy_(batch_mean)
            self.var.copy_(batch_var)
        else:
            delta = batch_mean - self.mean
            self.mean += delta * batch_count / total_count
            self.var = (self.var * self.count + batch_var * batch_count + 
                       delta.pow(2) * self.count * batch_count / total_count) / total_count
        
        self.count.add_(batch_count)
    
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data using running statistics"""
        return (data - self.mean) / torch.sqrt(self.var + self.epsilon)
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data"""
        return data * torch.sqrt(self.var + self.epsilon) + self.mean
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic statistics update"""
        if self.training:
            self.update(data)
        return self.normalize(data)


class StatePreprocessor(Preprocessor):
    """Preprocessor for state/observation data"""
    
    def __init__(
        self,
        normalize: bool = True,
        clip_range: Optional[tuple] = None,
        epsilon: float = 1e-8,
        momentum: float = 0.99,
        shape: Optional[tuple] = None
    ):
        """
        Args:
            normalize: Whether to normalize states
            clip_range: Optional clipping range (min, max)
            epsilon: Small value for numerical stability
            momentum: Momentum for running statistics
            shape: Shape of the state data
        """
        self.normalize = normalize
        self.clip_range = clip_range
        self.running_stats = None
        
        if normalize and shape is not None:
            self.running_stats = RunningMeanStd(shape, epsilon, momentum)
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply state preprocessing"""
        # Clip values if specified
        if self.clip_range is not None:
            data = torch.clamp(data, self.clip_range[0], self.clip_range[1])
        
        # Normalize if enabled
        if self.normalize and self.running_stats is not None:
            data = self.running_stats(data)
        
        return data


class ValuePreprocessor(Preprocessor):
    """Preprocessor for value function outputs"""
    
    def __init__(
        self,
        normalize: bool = True,
        clip_range: Optional[tuple] = None,
        epsilon: float = 1e-8,
        momentum: float = 0.99
    ):
        """
        Args:
            normalize: Whether to normalize values
            clip_range: Optional clipping range (min, max)
            epsilon: Small value for numerical stability
            momentum: Momentum for running statistics
        """
        self.normalize = normalize
        self.clip_range = clip_range
        self.running_stats = None
        
        if normalize:
            self.running_stats = RunningMeanStd((1,), epsilon, momentum)
    
    def __call__(self, data: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Apply value preprocessing
        
        Args:
            data: Value data to process
            inverse: If True, apply inverse transformation
        """
        if inverse and self.normalize and self.running_stats is not None:
            # Denormalize
            data = self.running_stats.denormalize(data)
        elif not inverse:
            # Forward preprocessing
            if self.clip_range is not None:
                data = torch.clamp(data, self.clip_range[0], self.clip_range[1])
            
            if self.normalize and self.running_stats is not None:
                data = self.running_stats(data)
        
        return data


class RewardNormalizer(nn.Module):
    """Reward normalization for stable training"""
    
    def __init__(
        self,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip_range: float = 10.0
    ):
        """
        Args:
            gamma: Discount factor for return estimation
            epsilon: Small value for numerical stability
            clip_range: Range to clip normalized rewards
        """
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip_range = clip_range
        
        self.register_buffer('return_', torch.zeros(1))
        self.register_buffer('var', torch.ones(1))
        
    def update(self, rewards: torch.Tensor, dones: torch.Tensor):
        """Update return statistics"""
        if not self.training:
            return
            
        returns = torch.zeros_like(rewards)
        current_return = self.return_
        
        # Compute discounted returns
        for i in reversed(range(len(rewards))):
            current_return = rewards[i] + self.gamma * current_return * (1 - dones[i])
            returns[i] = current_return
        
        # Update variance
        self.var = self.var * 0.99 + returns.var() * 0.01
        self.return_ = current_return
    
    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards"""
        normalized = rewards / torch.sqrt(self.var + self.epsilon)
        return torch.clamp(normalized, -self.clip_range, self.clip_range)
    
    def forward(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic statistics update"""
        if self.training:
            self.update(rewards, dones)
        return self.normalize(rewards)