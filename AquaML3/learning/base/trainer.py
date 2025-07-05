"""Base Trainer

This module provides the base trainer class for AquaML training workflows.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from loguru import logger

try:
    from ...core.exceptions import LearningError
except ImportError:
    # Fallback for when used as standalone module
    class LearningError(Exception):
        pass
from .learner import BaseLearner


class BaseTrainer(ABC):
    """Base class for training workflows
    
    This class manages the training process for learners.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        
        # Training state
        self.is_training = False
        self.current_epoch = 0
        self.learners: List[BaseLearner] = []
        
        logger.info(f"Initialized {self.name} trainer")
    
    def add_learner(self, learner: BaseLearner) -> None:
        """Add a learner to train
        
        Args:
            learner: Learner to add
        """
        self.learners.append(learner)
        logger.info(f"Added learner {learner.name} to trainer")
    
    def remove_learner(self, learner: BaseLearner) -> None:
        """Remove a learner from training
        
        Args:
            learner: Learner to remove
        """
        if learner in self.learners:
            self.learners.remove(learner)
            logger.info(f"Removed learner {learner.name} from trainer")
    
    @abstractmethod
    def train(self, **kwargs) -> Dict[str, Any]:
        """Train all learners
        
        Returns:
            Training results
        """
        pass
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state
        
        Returns:
            Training state dictionary
        """
        return {
            'name': self.name,
            'is_training': self.is_training,
            'current_epoch': self.current_epoch,
            'num_learners': len(self.learners),
            'learner_names': [learner.name for learner in self.learners]
        } 