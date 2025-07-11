"""
AquaML Device Information

This module provides classes to store and manage device information,
including GPU specifications, memory status, and compute capabilities.
"""

import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class GPUInfo:
    """GPU device information class"""
    
    index: int
    name: str
    device_id: str  # e.g., "cuda:0"
    memory_total: int  # Total memory in bytes
    memory_allocated: int  # Currently allocated memory in bytes
    memory_reserved: int  # Reserved memory in bytes
    memory_free: int  # Free memory in bytes
    compute_capability: tuple  # (major, minor)
    sm_count: int  # Number of streaming multiprocessors
    max_threads_per_sm: int  # Maximum threads per SM
    max_threads_per_block: int  # Maximum threads per block
    max_shared_memory_per_block: int  # Maximum shared memory per block
    warp_size: int  # Warp size
    memory_clock_rate: int  # Memory clock rate in kHz
    memory_bus_width: int  # Memory bus width in bits
    l2_cache_size: int  # L2 cache size in bytes
    max_texture_1d: int  # Maximum 1D texture size
    max_texture_2d: tuple  # Maximum 2D texture size (width, height)
    max_texture_3d: tuple  # Maximum 3D texture size (width, height, depth)
    
    def __post_init__(self):
        """Post-initialization to update memory info"""
        self.update_memory_info()
    
    def update_memory_info(self):
        """Update current memory information"""
        if torch.cuda.is_available() and self.index < torch.cuda.device_count():
            self.memory_allocated = torch.cuda.memory_allocated(self.index)
            self.memory_reserved = torch.cuda.memory_reserved(self.index)
            self.memory_free = self.memory_total - self.memory_reserved
    
    @property
    def memory_usage_percent(self) -> float:
        """Calculate memory usage percentage"""
        if self.memory_total == 0:
            return 0.0
        return (self.memory_allocated / self.memory_total) * 100
    
    @property
    def memory_total_gb(self) -> float:
        """Get total memory in GB"""
        return self.memory_total / (1024**3)
    
    @property
    def memory_allocated_gb(self) -> float:
        """Get allocated memory in GB"""
        return self.memory_allocated / (1024**3)
    
    @property
    def memory_free_gb(self) -> float:
        """Get free memory in GB"""
        return self.memory_free / (1024**3)
    
    @property
    def compute_capability_str(self) -> str:
        """Get compute capability as string"""
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"
    
    @property
    def theoretical_fp32_performance(self) -> float:
        """Calculate theoretical FP32 performance in GFLOPS"""
        # This is a rough estimation based on SM count and compute capability
        base_performance = self.sm_count * self.max_threads_per_sm * 2  # 2 operations per thread roughly
        
        # Adjust based on compute capability
        if self.compute_capability >= (8, 0):  # Ampere and newer
            multiplier = 1.5
        elif self.compute_capability >= (7, 0):  # Turing/Volta
            multiplier = 1.3
        elif self.compute_capability >= (6, 0):  # Pascal
            multiplier = 1.1
        else:  # Older architectures
            multiplier = 1.0
        
        return base_performance * multiplier / 1000  # Convert to GFLOPS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert GPU info to dictionary"""
        return {
            'index': self.index,
            'name': self.name,
            'device_id': self.device_id,
            'memory': {
                'total': self.memory_total,
                'total_gb': self.memory_total_gb,
                'allocated': self.memory_allocated,
                'allocated_gb': self.memory_allocated_gb,
                'free': self.memory_free,
                'free_gb': self.memory_free_gb,
                'usage_percent': self.memory_usage_percent
            },
            'compute': {
                'capability': self.compute_capability_str,
                'sm_count': self.sm_count,
                'max_threads_per_sm': self.max_threads_per_sm,
                'max_threads_per_block': self.max_threads_per_block,
                'warp_size': self.warp_size,
                'theoretical_fp32_gflops': self.theoretical_fp32_performance
            },
            'memory_specs': {
                'clock_rate_khz': self.memory_clock_rate,
                'bus_width_bits': self.memory_bus_width,
                'l2_cache_size': self.l2_cache_size,
                'max_shared_memory_per_block': self.max_shared_memory_per_block
            },
            'texture_limits': {
                'max_texture_1d': self.max_texture_1d,
                'max_texture_2d': self.max_texture_2d,
                'max_texture_3d': self.max_texture_3d
            }
        }
    
    def __str__(self) -> str:
        """String representation of GPU info"""
        return (f"GPU {self.index}: {self.name}\n"
                f"  Memory: {self.memory_total_gb:.2f} GB "
                f"({self.memory_usage_percent:.1f}% used)\n"
                f"  Compute: {self.compute_capability_str} "
                f"({self.sm_count} SMs, {self.theoretical_fp32_performance:.0f} GFLOPS)")


def detect_gpu_devices() -> list:
    """Detect and create GPU device information objects
    
    Returns:
        List of GPUInfo objects for available GPUs
    """
    gpu_devices = []
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, no GPU devices detected")
        return gpu_devices
    
    device_count = torch.cuda.device_count()
    logger.info(f"Detected {device_count} CUDA GPU(s)")
    
    for i in range(device_count):
        try:
            # Get device properties
            props = torch.cuda.get_device_properties(i)
            
            # Create GPU info object with correct attribute names
            gpu_info = GPUInfo(
                index=i,
                name=props.name,
                device_id=f"cuda:{i}",
                memory_total=props.total_memory,
                memory_allocated=torch.cuda.memory_allocated(i),
                memory_reserved=torch.cuda.memory_reserved(i),
                memory_free=props.total_memory - torch.cuda.memory_reserved(i),
                compute_capability=(props.major, props.minor),
                sm_count=props.multi_processor_count,
                max_threads_per_sm=props.max_threads_per_multi_processor,
                max_threads_per_block=1024,  # Standard CUDA block size limit
                max_shared_memory_per_block=props.shared_memory_per_block,
                warp_size=props.warp_size,
                memory_clock_rate=0,  # Not available in this PyTorch version
                memory_bus_width=0,  # Not available in this PyTorch version
                l2_cache_size=props.L2_cache_size,
                max_texture_1d=0,  # Not available in this PyTorch version
                max_texture_2d=(0, 0),  # Not available in this PyTorch version
                max_texture_3d=(0, 0, 0)  # Not available in this PyTorch version
            )
            
            gpu_devices.append(gpu_info)
            logger.info(f"Detected GPU {i}: {props.name}")
            
        except Exception as e:
            logger.error(f"Error detecting GPU {i}: {e}")
    
    return gpu_devices


def get_optimal_device(gpu_devices: list) -> str:
    """Get optimal device based on available GPUs
    
    Args:
        gpu_devices: List of GPUInfo objects
        
    Returns:
        Optimal device string
    """
    if not gpu_devices:
        return "cpu"
    
    # Find GPU with most free memory
    best_gpu = max(gpu_devices, key=lambda gpu: gpu.memory_free)
    logger.info(f"Selected optimal device: {best_gpu.device_id} "
                f"({best_gpu.memory_free_gb:.2f} GB free)")
    
    return best_gpu.device_id 