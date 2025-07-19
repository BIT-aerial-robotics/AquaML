from typing import List
import torch
from loguru import logger

from .base import Memory, MemoryCfg
from AquaML.config import configclass


@configclass
class RandomMemoryCfg(MemoryCfg):
    """Configuration for random memory"""
    replacement: bool = True  # Sample with or without replacement


class RandomMemory(Memory):
    """Random sampling memory for replay buffers
    
    This memory implementation samples data randomly from the buffer,
    typically used for off-policy algorithms like SAC, DDPG, etc.
    """

    def __init__(self, cfg: RandomMemoryCfg):
        """Initialize random memory
        
        Args:
            cfg: Random memory configuration
        """
        super().__init__(cfg)
        self.replacement = cfg.replacement
        logger.info(f"RandomMemory initialized with replacement={self.replacement}")

    def sample(self, names: List[str], batch_size: int, mini_batches: int = 1) -> List[List[torch.Tensor]]:
        """Sample data randomly from memory
        
        Args:
            names: Tensor names to sample
            batch_size: Number of samples per batch
            mini_batches: Number of mini-batches
            
        Returns:
            List of batched tensors
        """
        total_samples = self.get_stored_samples_count()
        
        if total_samples == 0:
            logger.warning("No samples in memory to sample from")
            return []
        
        # Ensure batch size doesn't exceed available samples
        if not self.replacement:
            batch_size = min(batch_size, total_samples)
        
        batches = []
        for _ in range(mini_batches):
            # Generate random indices
            if self.replacement:
                indices = torch.randint(0, total_samples, (batch_size,), device=self.device)
            else:
                indices = torch.randperm(total_samples, device=self.device)[:batch_size]
            
            # Sample data
            batch_data = []
            for name in names:
                if name in self.tensors:
                    batch_data.append(self.tensors[name][indices])
                else:
                    logger.warning(f"Tensor {name} not found in memory")
                    batch_data.append(torch.tensor([], device=self.device))
            
            batches.append(batch_data)
        
        return batches