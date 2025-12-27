"""
Learning Rate Schedules

This module provides various learning rate scheduling strategies for gradient descent.
"""

from abc import ABC, abstractmethod
import numpy as np


class LearningRateSchedule(ABC):
    """Base class for learning rate schedules."""
    
    @abstractmethod
    def get_learning_rate(self, iteration: int) -> float:
        """
        Get the learning rate for a given iteration.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Learning rate for this iteration
        """
        pass


class ConstantLearningRate(LearningRateSchedule):
    """
    Constant learning rate (no decay).
    
    Args:
        learning_rate: Fixed learning rate value
    """
    
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
    
    def get_learning_rate(self, iteration: int) -> float:
        return self.learning_rate


class ExponentialDecay(LearningRateSchedule):
    """
    Exponential learning rate decay.
    
    learning_rate = initial_lr * decay_rate^(iteration / decay_steps)
    
    Args:
        initial_learning_rate: Initial learning rate
        decay_rate: Decay multiplier (e.g., 0.96)
        decay_steps: Number of steps for one decay period
    """
    
    def __init__(
        self,
        initial_learning_rate: float,
        decay_rate: float = 0.96,
        decay_steps: int = 100
    ):
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
    
    def get_learning_rate(self, iteration: int) -> float:
        return self.initial_learning_rate * (self.decay_rate ** (iteration / self.decay_steps))


class StepDecay(LearningRateSchedule):
    """
    Step-wise learning rate decay.
    
    Reduces learning rate by a factor every N steps.
    
    Args:
        initial_learning_rate: Initial learning rate
        drop_rate: Factor to multiply learning rate by (e.g., 0.5)
        drop_every: Number of iterations between drops
    """
    
    def __init__(
        self,
        initial_learning_rate: float,
        drop_rate: float = 0.5,
        drop_every: int = 100
    ):
        self.initial_learning_rate = initial_learning_rate
        self.drop_rate = drop_rate
        self.drop_every = drop_every
    
    def get_learning_rate(self, iteration: int) -> float:
        drop_count = iteration // self.drop_every
        return self.initial_learning_rate * (self.drop_rate ** drop_count)
