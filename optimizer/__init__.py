"""
Gradient Descent Optimizer Package

A Python implementation of gradient descent optimization algorithms.
"""

from .gradient_descent import GradientDescentOptimizer
from .learning_rate import (
    LearningRateSchedule,
    ConstantLearningRate,
    ExponentialDecay,
    StepDecay
)

__version__ = "1.0.0"

__all__ = [
    "GradientDescentOptimizer",
    "LearningRateSchedule",
    "ConstantLearningRate",
    "ExponentialDecay",
    "StepDecay"
]
