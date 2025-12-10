"""
Simple example demonstrating gradient descent optimization.

This example optimizes a simple quadratic function: f(x, y) = x^2 + y^2
The minimum is at (0, 0).
"""

import numpy as np
import sys
import os

# Add parent directory to path to import optimizer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimizer import GradientDescentOptimizer


def main():
    """Run simple gradient descent example."""
    print("=" * 60)
    print("Simple Gradient Descent Example")
    print("=" * 60)
    print("\nOptimizing f(x, y) = x^2 + y^2")
    print("Expected minimum: (0, 0)\n")
    
    # Define the loss function
    def loss_fn(params):
        """Loss function: f(x, y) = x^2 + y^2"""
        return params[0]**2 + params[1]**2
    
    # Define the gradient function
    def gradient_fn(params):
        """Gradient: [2x, 2y]"""
        return np.array([2 * params[0], 2 * params[1]])
    
    # Initialize optimizer
    optimizer = GradientDescentOptimizer(
        learning_rate=0.1,
        max_iterations=100,
        tolerance=1e-8,
        verbose=True
    )
    
    # Starting point
    initial_params = np.array([5.0, 3.0])
    print(f"Starting point: ({initial_params[0]:.2f}, {initial_params[1]:.2f})")
    print(f"Initial loss: {loss_fn(initial_params):.6f}\n")
    
    # Optimize
    result = optimizer.optimize(loss_fn, gradient_fn, initial_params)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Optimized parameters: ({result[0]:.6f}, {result[1]:.6f})")
    print(f"Final loss: {loss_fn(result):.6f}")
    
    history = optimizer.get_history()
    print(f"Iterations: {len(history['loss'])}")
    print(f"Loss improvement: {history['loss'][0]:.6f} -> {history['loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
