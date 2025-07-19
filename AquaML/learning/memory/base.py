from typing import Dict, Any, Optional, Tuple, Union, List
import torch
import numpy as np
from abc import ABC, abstractmethod
from loguru import logger

from AquaML import coordinator
from AquaML.config import configclass


@configclass
class MemoryCfg:
    """Base configuration for memory"""
    memory_size: int = 10000
    device: str = "auto"
    num_envs: int = 1


class Memory(ABC):
    """Base memory class for storing and sampling transitions
    
    This is a generic memory implementation that can be used by different RL algorithms.
    It handles both dictionary-based data structures and tensor data with proper device management.
    """

    def __init__(self, cfg: MemoryCfg):
        """Initialize memory
        
        Args:
            cfg: Memory configuration
        """
        self.cfg = cfg
        self.memory_size = cfg.memory_size
        self.num_envs = cfg.num_envs
        
        # Device setup
        if cfg.device == "auto":
            self.device = coordinator.get_device()
        else:
            self.device = cfg.device
            
        # Memory state
        self.position = 0
        self.full = False
        
        # Storage
        self.tensors = {}
        self.data_structure_info = {}
        
        logger.info(f"Memory initialized with size {self.memory_size} on device {self.device}")

    def create_tensor(
        self,
        name: str,
        size: Union[int, Tuple[int]],
        dtype: torch.dtype = torch.float32,
    ):
        """Create a tensor in memory
        
        Args:
            name: Tensor name
            size: Tensor size
            dtype: Data type
        """
        if isinstance(size, int):
            size = (size,)
        self.tensors[name] = torch.zeros(
            (self.memory_size,) + size, dtype=dtype, device=self.device
        )

    def store_data_structure(self, name: str, structure_info: Dict[str, Any]):
        """Store metadata about dictionary data structures
        
        Args:
            name: Structure name
            structure_info: Structure metadata
        """
        self.data_structure_info[name] = structure_info

    def add_samples(self, **kwargs):
        """Add samples to memory with improved structure preservation
        
        Args:
            **kwargs: Named data to store
        """
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
                    if self.tensors[key].shape[1:] != value.shape:
                        new_shape = (self.memory_size,) + value.shape
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
        """Flatten dictionary with better structure preservation
        
        Args:
            data_dict: Dictionary to flatten
            
        Returns:
            Flattened tensor
        """
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
        """Reconstruct dictionary from flattened tensor using stored structure info
        
        Args:
            flattened_tensor: Flattened tensor
            structure_name: Name of the structure
            
        Returns:
            Reconstructed dictionary
        """
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

    def get_tensor_by_name(self, name: str) -> Optional[torch.Tensor]:
        """Get tensor by name
        
        Args:
            name: Tensor name
            
        Returns:
            Tensor or None if not found
        """
        if name in self.tensors:
            if self.full:
                return self.tensors[name]
            else:
                return self.tensors[name][:self.position]
        return None

    def set_tensor_by_name(self, name: str, tensor: torch.Tensor):
        """Set tensor by name
        
        Args:
            name: Tensor name
            tensor: Tensor to set
        """
        if name in self.tensors:
            # Ensure tensor shape matches
            if tensor.shape != self.tensors[name].shape:
                # If shape doesn't match, recreate tensor
                self.tensors[name] = torch.zeros_like(tensor, device=self.device)
            
            if self.full:
                self.tensors[name] = tensor
            else:
                # Ensure tensor shape matches storage space
                target_shape = self.tensors[name][:self.position].shape
                if tensor.shape != target_shape:
                    # Adjust tensor shape to match target shape
                    if tensor.dim() > len(target_shape):
                        # If tensor has too many dimensions, squeeze
                        tensor = tensor.squeeze()
                    elif tensor.dim() < len(target_shape):
                        # If tensor has insufficient dimensions, expand
                        tensor = tensor.unsqueeze(-1)
                    
                    # Ensure shapes match exactly
                    if tensor.shape != target_shape:
                        tensor = tensor.view(target_shape)
                
                self.tensors[name][:self.position] = tensor

    @abstractmethod
    def sample(self, names: List[str], batch_size: int, mini_batches: int = 1) -> List[List[torch.Tensor]]:
        """Sample data from memory
        
        Args:
            names: Tensor names to sample
            batch_size: Number of samples per batch
            mini_batches: Number of mini-batches
            
        Returns:
            List of batched tensors
        """
        pass

    def sample_all(self, names: List[str], mini_batches: int = 1):
        """Sample all data in mini-batches with improved efficiency
        
        Args:
            names: Tensor names to sample
            mini_batches: Number of mini-batches
            
        Yields:
            Batched data
        """
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
        """Get number of stored samples
        
        Returns:
            Number of stored samples
        """
        return self.memory_size if self.full else self.position

    def is_ready_for_update(self, min_samples: int) -> bool:
        """Check if memory has enough samples for update
        
        Args:
            min_samples: Minimum number of samples needed
            
        Returns:
            True if ready for update
        """
        return self.get_stored_samples_count() >= min_samples

    def __len__(self) -> int:
        """Get memory length
        
        Returns:
            Number of stored samples
        """
        return self.get_stored_samples_count()