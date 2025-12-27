# Gradient Descent Optimizer

A flexible and easy-to-use Python implementation of gradient descent optimization algorithms.

## Features

- **Multiple Variants**: Batch, Stochastic (SGD), and Mini-batch gradient descent
- **Momentum Support**: Accelerate convergence with momentum
- **Learning Rate Schedules**: Constant, exponential decay, and step decay
- **History Tracking**: Monitor loss and learning rate during optimization
- **Flexible API**: Easy to use with custom loss and gradient functions

## Installation

```bash
pip install -r requirements.txt
```

Or install the package:

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from optimizer import GradientDescentOptimizer

# Define your loss function
def loss_fn(params):
    return (params[0] - 3)**2 + (params[1] + 1)**2

# Define the gradient
def gradient_fn(params):
    return np.array([2*(params[0] - 3), 2*(params[1] + 1)])

# Create optimizer
optimizer = GradientDescentOptimizer(
    learning_rate=0.1,
    max_iterations=100,
    tolerance=1e-6
)

# Optimize
initial_params = np.array([0.0, 0.0])
result = optimizer.optimize(loss_fn, gradient_fn, initial_params)
print(f"Optimized parameters: {result}")
```

## Usage Examples

### Batch Gradient Descent

```python
from optimizer import GradientDescentOptimizer

optimizer = GradientDescentOptimizer(
    learning_rate=0.01,
    batch_size=None,  # None for batch gradient descent
    max_iterations=1000
)
```

### Stochastic Gradient Descent (SGD)

```python
optimizer = GradientDescentOptimizer(
    learning_rate=0.01,
    batch_size=1,  # 1 for stochastic gradient descent
    max_iterations=1000
)

# Requires X and y data
result = optimizer.optimize(loss_fn, gradient_fn, initial_params, X, y)
```

### Mini-Batch Gradient Descent

```python
optimizer = GradientDescentOptimizer(
    learning_rate=0.01,
    batch_size=32,  # Mini-batch size
    momentum=0.9,
    max_iterations=1000
)

result = optimizer.optimize(loss_fn, gradient_fn, initial_params, X, y)
```

### Learning Rate Schedules

```python
from optimizer import GradientDescentOptimizer
from optimizer.learning_rate import ExponentialDecay, StepDecay

# Exponential decay
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.5,
    decay_rate=0.96,
    decay_steps=100
)

optimizer = GradientDescentOptimizer(learning_rate=lr_schedule)

# Step decay
lr_schedule = StepDecay(
    initial_learning_rate=0.5,
    drop_rate=0.5,
    drop_every=100
)

optimizer = GradientDescentOptimizer(learning_rate=lr_schedule)
```

## API Reference

### GradientDescentOptimizer

Main optimizer class for gradient descent.

**Parameters:**
- `learning_rate` (float or LearningRateSchedule): Learning rate or schedule
- `batch_size` (int or None): Size of mini-batches (None for batch GD, 1 for SGD)
- `momentum` (float): Momentum factor (0-1, default: 0)
- `max_iterations` (int): Maximum number of iterations (default: 1000)
- `tolerance` (float): Convergence tolerance (default: 1e-6)
- `verbose` (bool): Print progress information (default: False)

**Methods:**
- `optimize(loss_fn, gradient_fn, initial_params, X=None, y=None)`: Run optimization
- `get_history()`: Get optimization history
- `reset_history()`: Reset the history

### Learning Rate Schedules

**ConstantLearningRate**: Fixed learning rate throughout training

**ExponentialDecay**: Exponential decay schedule
- `initial_learning_rate`: Starting learning rate
- `decay_rate`: Decay multiplier (e.g., 0.96)
- `decay_steps`: Steps per decay period

**StepDecay**: Step-wise decay schedule
- `initial_learning_rate`: Starting learning rate
- `drop_rate`: Multiplicative factor (e.g., 0.5)
- `drop_every`: Iterations between drops

## Examples

First, install the package in development mode:

```bash
pip install -e .
```

Then run the included examples:

```bash
# Simple quadratic optimization
python examples/simple_example.py

# Linear regression with mini-batch GD
python examples/linear_regression.py
```

## Running Tests

```bash
pytest tests/
```

## License

MIT License
