from typing import List
import torch
from loguru import logger

from .base import Memory, MemoryCfg
from AquaML.config import configclass


@configclass
class SequentialMemoryCfg(MemoryCfg):
    """Configuration for sequential memory"""
    pass  # Uses base configuration


class SequentialMemory(Memory):
    """Sequential memory for on-policy algorithms
    
    This memory implementation stores transitions sequentially and samples all data
    in order, typically used for on-policy algorithms like PPO, A2C, etc.
    The memory acts as a rollout buffer that collects experiences during policy rollouts.
    """

    def __init__(self, cfg: SequentialMemoryCfg):
        """Initialize sequential memory
        
        Args:
            cfg: Sequential memory configuration
        """
        super().__init__(cfg)
        logger.info("SequentialMemory initialized for on-policy algorithms")

    def sample(self, names: List[str], batch_size: int, mini_batches: int = 1) -> List[List[torch.Tensor]]:
        """Sample data sequentially from memory
        
        For sequential memory, this method samples all available data and splits it into mini-batches.
        The batch_size parameter is mainly for compatibility but isn't strictly used.
        
        Args:
            names: Tensor names to sample
            batch_size: Number of samples per batch (for compatibility)
            mini_batches: Number of mini-batches to split the data into
            
        Returns:
            List of batched tensors
        """
        total_samples = self.get_stored_samples_count()
        
        if total_samples == 0:
            logger.warning("No samples in memory to sample from")
            return []
        
        # For sequential memory, we sample all data and split into mini-batches
        return list(self.sample_all(names, mini_batches))

    def sample_all_shuffled(self, names: List[str], mini_batches: int = 1):
        """Sample all data with shuffling for better training
        
        Args:
            names: Tensor names to sample
            mini_batches: Number of mini-batches
            
        Yields:
            Batched tensors with shuffled data
        """
        total_samples = self.get_stored_samples_count()
        
        if total_samples == 0:
            return
            
        # Generate shuffled indices
        indices = torch.randperm(total_samples, device=self.device)
        
        # Split into mini-batches
        batch_size = max(1, total_samples // mini_batches)
        
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
                    logger.warning(f"Tensor {name} not found in memory")
                    batch_data.append(torch.tensor([], device=self.device))
            
            yield batch_data