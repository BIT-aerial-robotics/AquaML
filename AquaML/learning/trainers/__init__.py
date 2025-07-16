"""AquaML Trainers Module

This module provides trainer classes for reinforcement learning in AquaML.
All trainers integrate with the AquaML coordinator and support dictionary-based
actions, rewards, and states.
"""

from .base import BaseTrainer, TrainerConfig
from .sequential import SequentialTrainer

__all__ = [
    "BaseTrainer",
    "TrainerConfig", 
    "SequentialTrainer"
]