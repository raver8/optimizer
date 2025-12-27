"""
Gradient Descent Optimizer Implementation

This module provides a flexible implementation of gradient descent optimization
with support for batch, stochastic, and mini-batch variants.
"""

import numpy as np
from typing import Callable, Optional, List, Dict, Any
from .learning_rate import LearningRateSchedule, ConstantLearningRate


class GradientDescentOptimizer:
    """
    Gradient Descent Optimizer
    
    Implements gradient descent optimization with support for:
    - Batch gradient descent
    - Stochastic gradient descent (SGD)
    - Mini-batch gradient descent
    - Momentum
    - Learning rate schedules
    
    Args:
        learning_rate: Initial learning rate or LearningRateSchedule object
        batch_size: Size of mini-batches (None for batch GD, 1 for SGD)
        momentum: Momentum factor (0 for no momentum)
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance for stopping criterion
        verbose: Whether to print progress information
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: Optional[int] = None,
        momentum: float = 0.0,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = False
    ):
        if isinstance(learning_rate, (int, float)):
            self.lr_schedule = ConstantLearningRate(learning_rate)
        elif isinstance(learning_rate, LearningRateSchedule):
            self.lr_schedule = learning_rate
        else:
            raise ValueError("learning_rate must be a number or LearningRateSchedule")
        
        self.batch_size = batch_size
        self.momentum = momentum
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
        # History tracking
        self.history: Dict[str, List[float]] = {
            'loss': [],
            'learning_rate': []
        }
        
        # Velocity for momentum
        self.velocity: Optional[np.ndarray] = None
    
    def optimize(
        self,
        loss_fn: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        initial_params: np.ndarray,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Optimize parameters using gradient descent.
        
        Args:
            loss_fn: Function that computes the loss given parameters
            gradient_fn: Function that computes the gradient given parameters
            initial_params: Initial parameter values
            X: Optional input data for mini-batch/stochastic variants
            y: Optional target data for mini-batch/stochastic variants
            
        Returns:
            Optimized parameters
        """
        params = initial_params.copy()
        self.velocity = np.zeros_like(params)
        
        n_samples = len(X) if X is not None else None
        
        for iteration in range(self.max_iterations):
            # Get learning rate for this iteration
            lr = self.lr_schedule.get_learning_rate(iteration)
            
            # Compute gradient based on batch type
            if self.batch_size is None:
                # Batch gradient descent
                gradient = gradient_fn(params)
                loss = loss_fn(params)
            else:
                # Mini-batch or stochastic gradient descent
                if X is None or y is None:
                    raise ValueError("X and y must be provided for mini-batch/SGD")
                
                # Sample mini-batch
                if self.batch_size == 1:
                    # Stochastic - single random sample
                    idx = np.random.randint(0, n_samples)
                    batch_X = X[idx:idx+1]
                    batch_y = y[idx:idx+1]
                else:
                    # Mini-batch - random subset
                    # Use replace=False if batch_size <= n_samples, otherwise replace=True
                    replace = self.batch_size > n_samples
                    actual_batch_size = min(self.batch_size, n_samples)
                    indices = np.random.choice(n_samples, actual_batch_size, replace=replace)
                    batch_X = X[indices]
                    batch_y = y[indices]
                
                # Compute gradient on mini-batch
                gradient = gradient_fn(params, batch_X, batch_y)
                loss = loss_fn(params, batch_X, batch_y)
            
            # Update velocity with momentum
            self.velocity = self.momentum * self.velocity + lr * gradient
            
            # Update parameters
            params = params - self.velocity
            
            # Track history
            self.history['loss'].append(loss)
            self.history['learning_rate'].append(lr)
            
            # Verbose output
            if self.verbose and (iteration % 100 == 0 or iteration == self.max_iterations - 1):
                print(f"Iteration {iteration}: Loss = {loss:.6f}, LR = {lr:.6f}")
            
            # Check convergence
            if len(self.history['loss']) > 1:
                loss_change = abs(self.history['loss'][-1] - self.history['loss'][-2])
                if loss_change < self.tolerance:
                    if self.verbose:
                        print(f"Converged at iteration {iteration}")
                    break
        
        return params
    
    def get_history(self) -> Dict[str, List[float]]:
        """
        Get optimization history.
        
        Returns:
            Dictionary with loss and learning rate history
        """
        return self.history
    
    def reset_history(self) -> None:
        """Reset the optimization history."""
        self.history = {'loss': [], 'learning_rate': []}
