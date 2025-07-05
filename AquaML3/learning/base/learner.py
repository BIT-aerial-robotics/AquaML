"""Base Learner

This module provides the base learner class for AquaML learning algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
from loguru import logger

try:
    from ...core.exceptions import LearningError
except ImportError:
    # Fallback for when used as standalone module
    class LearningError(Exception):
        pass


class BaseLearner(ABC):
    """Base class for all learning algorithms
    
    This class provides the interface that all learning algorithms should implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the learner
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Learning state
        self.is_trained = False
        self.training_step = 0
        self.epoch = 0
        
        # Metrics tracking
        self.metrics = {}
        self.best_metrics = {}
        
        logger.info(f"Initialized {self.name} learner")
    
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, Any]:
        """Perform a single training step
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Make predictions
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, eval_data: Any) -> Dict[str, Any]:
        """Evaluate the model
        
        Args:
            eval_data: Evaluation data
            
        Returns:
            Evaluation metrics
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the model
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load the model
        
        Args:
            path: Path to load the model from
        """
        pass
    
    def train(self, train_data: Any, 
              eval_data: Optional[Any] = None,
              epochs: int = 1,
              **kwargs) -> Dict[str, Any]:
        """Train the model
        
        Args:
            train_data: Training data
            eval_data: Evaluation data (optional)
            epochs: Number of training epochs
            **kwargs: Additional training arguments
            
        Returns:
            Training results
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        training_results = {
            'epochs': epochs,
            'total_steps': 0,
            'train_metrics': [],
            'eval_metrics': []
        }
        
        try:
            for epoch in range(epochs):
                self.epoch = epoch
                
                # Training phase
                train_metrics = self._train_epoch(train_data, **kwargs)
                training_results['train_metrics'].append(train_metrics)
                
                # Evaluation phase
                if eval_data is not None:
                    eval_metrics = self.evaluate(eval_data)
                    training_results['eval_metrics'].append(eval_metrics)
                    
                    # Update best metrics
                    self._update_best_metrics(eval_metrics)
                
                # Log progress
                self._log_training_progress(epoch, epochs, train_metrics, 
                                          eval_metrics if eval_data else None)
            
            self.is_trained = True
            training_results['total_steps'] = self.training_step
            
            logger.info(f"Training completed after {epochs} epochs")
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise LearningError(f"Training failed: {e}")
    
    def _train_epoch(self, train_data: Any, **kwargs) -> Dict[str, Any]:
        """Train for one epoch
        
        Args:
            train_data: Training data
            **kwargs: Additional arguments
            
        Returns:
            Training metrics for the epoch
        """
        epoch_metrics = []
        
        # This is a simplified version - subclasses should implement proper data iteration
        if hasattr(train_data, '__iter__'):
            for batch in train_data:
                batch_metrics = self.train_step(batch)
                epoch_metrics.append(batch_metrics)
                self.training_step += 1
        else:
            # Single batch training
            batch_metrics = self.train_step(train_data)
            epoch_metrics.append(batch_metrics)
            self.training_step += 1
        
        # Aggregate metrics
        return self._aggregate_metrics(epoch_metrics)
    
    def _aggregate_metrics(self, metrics_list: list) -> Dict[str, Any]:
        """Aggregate metrics from multiple batches
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Aggregated metrics
        """
        if not metrics_list:
            return {}
        
        # Simple averaging for now
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[key] = sum(values) / len(values)
        
        return aggregated
    
    def _update_best_metrics(self, current_metrics: Dict[str, Any]) -> None:
        """Update best metrics
        
        Args:
            current_metrics: Current evaluation metrics
        """
        for key, value in current_metrics.items():
            if key not in self.best_metrics:
                self.best_metrics[key] = value
            else:
                # Assume higher is better for now - can be customized
                if value > self.best_metrics[key]:
                    self.best_metrics[key] = value
    
    def _log_training_progress(self, epoch: int, total_epochs: int,
                              train_metrics: Dict[str, Any],
                              eval_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Log training progress
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            train_metrics: Training metrics
            eval_metrics: Evaluation metrics (optional)
        """
        log_str = f"Epoch {epoch + 1}/{total_epochs}"
        
        # Add training metrics
        for key, value in train_metrics.items():
            log_str += f" - {key}: {value:.4f}"
        
        # Add evaluation metrics
        if eval_metrics:
            for key, value in eval_metrics.items():
                log_str += f" - val_{key}: {value:.4f}"
        
        logger.info(log_str)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics
        
        Returns:
            Current metrics
        """
        return self.metrics.copy()
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best metrics
        
        Returns:
            Best metrics achieved during training
        """
        return self.best_metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset metrics"""
        self.metrics.clear()
        self.best_metrics.clear()
    
    def get_state(self) -> Dict[str, Any]:
        """Get learner state
        
        Returns:
            State dictionary
        """
        return {
            'name': self.name,
            'is_trained': self.is_trained,
            'training_step': self.training_step,
            'epoch': self.epoch,
            'metrics': self.metrics,
            'best_metrics': self.best_metrics
        }
    
    def set_device(self, device: str) -> None:
        """Set device for computation
        
        Args:
            device: Device string (e.g., 'cuda', 'cpu')
        """
        self.device = device
        logger.info(f"Set device to {device}")
    
    def to(self, device: str) -> 'BaseLearner':
        """Move learner to device
        
        Args:
            device: Target device
            
        Returns:
            Self for method chaining
        """
        self.set_device(device)
        return self
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', device='{self.device}')" 