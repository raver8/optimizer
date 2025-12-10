"""
Base optimizer class for all optimization algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    
    All optimizer implementations should inherit from this class and implement
    the step() and get_params() methods.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize the optimizer.
        
        Args:
            learning_rate: The learning rate for the optimizer
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        self.learning_rate = learning_rate
        self.iterations = 0
    
    @abstractmethod
    def step(self, params: List[float], gradients: List[float]) -> List[float]:
        """
        Perform a single optimization step.
        
        Args:
            params: Current parameter values
            gradients: Gradients of the loss function with respect to params
            
        Returns:
            Updated parameter values
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current optimizer parameters.
        
        Returns:
            Dictionary containing optimizer parameters
        """
        pass
