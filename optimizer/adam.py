"""
Adam optimizer implementation.
"""

from typing import Dict, List, Any
from .base import Optimizer


class AdamOptimizer(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Combines momentum and RMSProp for efficient optimization.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Step size for parameter updates
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        if not (0 <= beta1 < 1):
            raise ValueError("beta1 must be in [0, 1)")
        if not (0 <= beta2 < 1):
            raise ValueError("beta2 must be in [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
            
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
    
    def step(self, params: List[float], gradients: List[float]) -> List[float]:
        """
        Perform a single Adam optimization step.
        
        Args:
            params: Current parameter values
            gradients: Gradients of the loss function
            
        Returns:
            Updated parameter values
        """
        if len(params) != len(gradients):
            raise ValueError("params and gradients must have the same length")
        
        # Initialize moment estimates on first step
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)
        
        self.iterations += 1
        
        # Update biased first and second moment estimates
        self.m = [self.beta1 * m_i + (1 - self.beta1) * g 
                  for m_i, g in zip(self.m, gradients)]
        self.v = [self.beta2 * v_i + (1 - self.beta2) * g * g 
                  for v_i, g in zip(self.v, gradients)]
        
        # Compute bias-corrected moment estimates
        m_hat = [m_i / (1 - self.beta1 ** self.iterations) for m_i in self.m]
        v_hat = [v_i / (1 - self.beta2 ** self.iterations) for v_i in self.v]
        
        # Update parameters
        updated_params = [
            p - self.learning_rate * m / (v ** 0.5 + self.epsilon)
            for p, m, v in zip(params, m_hat, v_hat)
        ]
        
        return updated_params
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current optimizer parameters.
        
        Returns:
            Dictionary containing optimizer state
        """
        return {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'iterations': self.iterations,
            'm': self.m,
            'v': self.v
        }
