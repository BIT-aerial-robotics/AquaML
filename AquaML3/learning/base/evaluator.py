"""Base Evaluator

This module provides the base evaluator class for AquaML evaluation workflows.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from loguru import logger

try:
    from ...core.exceptions import LearningError
except ImportError:
    # Fallback for when used as standalone module
    class LearningError(Exception):
        pass
from .learner import BaseLearner


class BaseEvaluator(ABC):
    """Base class for model evaluation
    
    This class provides evaluation capabilities for learners.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the evaluator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        
        # Evaluation metrics
        self.metrics_history: List[Dict[str, Any]] = []
        self.best_metrics: Dict[str, Any] = {}
        
        logger.info(f"Initialized {self.name} evaluator")
    
    @abstractmethod
    def evaluate(self, learner: BaseLearner, eval_data: Any, **kwargs) -> Dict[str, Any]:
        """Evaluate a learner
        
        Args:
            learner: Learner to evaluate
            eval_data: Evaluation data
            **kwargs: Additional evaluation arguments
            
        Returns:
            Evaluation metrics
        """
        pass
    
    def evaluate_multiple(self, learners: List[BaseLearner], eval_data: Any, **kwargs) -> Dict[str, Dict[str, Any]]:
        """Evaluate multiple learners
        
        Args:
            learners: List of learners to evaluate
            eval_data: Evaluation data
            **kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary mapping learner names to metrics
        """
        results = {}
        
        for learner in learners:
            try:
                metrics = self.evaluate(learner, eval_data, **kwargs)
                results[learner.name] = metrics
                logger.info(f"Evaluated learner {learner.name}")
            except Exception as e:
                logger.error(f"Failed to evaluate learner {learner.name}: {e}")
                results[learner.name] = {'error': str(e)}
        
        return results
    
    def track_metrics(self, metrics: Dict[str, Any]) -> None:
        """Track evaluation metrics
        
        Args:
            metrics: Metrics to track
        """
        self.metrics_history.append(metrics)
        
        # Update best metrics
        for key, value in metrics.items():
            if key not in self.best_metrics:
                self.best_metrics[key] = value
            elif isinstance(value, (int, float)):
                # Assume higher is better - can be customized
                if value > self.best_metrics[key]:
                    self.best_metrics[key] = value
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get metrics history
        
        Returns:
            List of historical metrics
        """
        return self.metrics_history.copy()
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best metrics
        
        Returns:
            Best metrics achieved
        """
        return self.best_metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics_history.clear()
        self.best_metrics.clear()
        logger.info("Reset evaluator metrics")
    
    def get_evaluation_state(self) -> Dict[str, Any]:
        """Get evaluation state
        
        Returns:
            Evaluation state dictionary
        """
        return {
            'name': self.name,
            'num_evaluations': len(self.metrics_history),
            'best_metrics': self.best_metrics
        } 