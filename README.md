# optimizer

A Python library for power optimization algorithms.

## Features

- **Base Optimizer Class**: Abstract base class for implementing custom optimizers
- **Adam Optimizer**: Adaptive Moment Estimation optimizer for efficient optimization
- **Easy to Use**: Simple API for parameter optimization
- **Extensible**: Easy to add new optimization algorithms

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from optimizer import AdamOptimizer

# Initialize optimizer
optimizer = AdamOptimizer(learning_rate=0.1)

# Define your parameters and gradients
params = [0.0]
gradients = [2.0]

# Perform optimization step
updated_params = optimizer.step(params, gradients)
```

## Example

See `example.py` for a complete example of optimizing a quadratic function:

```bash
python example.py
```

## Running Tests

```bash
python -m unittest discover tests
```

## API Reference

### AdamOptimizer

The Adam (Adaptive Moment Estimation) optimizer combines momentum and RMSProp.

```python
optimizer = AdamOptimizer(
    learning_rate=0.001,  # Step size for parameter updates
    beta1=0.9,           # Exponential decay rate for first moment
    beta2=0.999,         # Exponential decay rate for second moment
    epsilon=1e-8         # Small constant for numerical stability
)
```

**Methods:**
- `step(params, gradients)`: Perform one optimization step
- `get_params()`: Get current optimizer state

## License

This project is licensed under the MIT License.
