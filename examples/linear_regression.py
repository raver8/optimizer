"""
Linear regression example using mini-batch gradient descent.

This example fits a linear model to synthetic data using mini-batch gradient descent.
"""

import numpy as np
from optimizer import GradientDescentOptimizer
from optimizer.learning_rate import ExponentialDecay


def main():
    """Run linear regression example with mini-batch gradient descent."""
    print("=" * 60)
    print("Linear Regression with Mini-Batch Gradient Descent")
    print("=" * 60)
    
    # Generate synthetic data: y = 3x + 5 + noise
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 1) * 2
    true_weight = 3.0
    true_bias = 5.0
    y = true_weight * X + true_bias + 0.5 * np.random.randn(n_samples, 1)
    
    print(f"\nGenerated {n_samples} samples")
    print(f"True parameters: weight={true_weight}, bias={true_bias}\n")
    
    # Define loss function (Mean Squared Error)
    def loss_fn(params, X_batch, y_batch):
        """MSE loss function."""
        predictions = X_batch @ params[:-1].reshape(-1, 1) + params[-1]
        return np.mean((predictions - y_batch) ** 2)
    
    # Define gradient function
    def gradient_fn(params, X_batch, y_batch):
        """Gradient of MSE loss."""
        predictions = X_batch @ params[:-1].reshape(-1, 1) + params[-1]
        error = predictions - y_batch
        grad_w = 2 * np.mean(X_batch * error, axis=0)
        grad_b = 2 * np.mean(error)
        return np.append(grad_w, grad_b)
    
    # Learning rate schedule with exponential decay
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.1,
        decay_rate=0.96,
        decay_steps=50
    )
    
    # Initialize optimizer with mini-batch size
    optimizer = GradientDescentOptimizer(
        learning_rate=lr_schedule,
        batch_size=32,
        momentum=0.5,
        max_iterations=500,
        tolerance=1e-6,
        verbose=False
    )
    
    # Initialize parameters
    initial_params = np.array([0.0, 0.0])
    
    # Optimize
    print("Training...")
    result = optimizer.optimize(loss_fn, gradient_fn, initial_params, X, y)
    
    # Evaluate on full dataset
    final_loss = loss_fn(result, X, y)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Learned weight: {result[0]:.6f} (true: {true_weight})")
    print(f"Learned bias: {result[1]:.6f} (true: {true_bias})")
    print(f"Final MSE loss: {final_loss:.6f}")
    
    history = optimizer.get_history()
    print(f"Iterations: {len(history['loss'])}")
    print(f"\nLearning rate schedule:")
    print(f"  Initial: {history['learning_rate'][0]:.6f}")
    print(f"  Final: {history['learning_rate'][-1]:.6f}")


if __name__ == "__main__":
    main()
