"""
Example usage of the optimizer package.

This example demonstrates how to use the AdamOptimizer to minimize
a simple quadratic function: f(x) = (x - 3)^2
"""

from optimizer import AdamOptimizer


def quadratic_function(x: float) -> float:
    """Simple quadratic function f(x) = (x - 3)^2"""
    return (x - 3) ** 2


def gradient_quadratic(x: float) -> float:
    """Gradient of f(x) = (x - 3)^2, which is 2(x - 3)"""
    return 2 * (x - 3)


def main():
    print("Optimizing f(x) = (x - 3)^2 using Adam Optimizer")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = AdamOptimizer(learning_rate=0.1)
    
    # Initial parameter
    params = [0.0]
    
    print(f"Initial: x = {params[0]:.4f}, f(x) = {quadratic_function(params[0]):.4f}")
    
    # Optimization loop
    for i in range(50):
        # Compute gradient
        gradients = [gradient_quadratic(params[0])]
        
        # Update parameters
        params = optimizer.step(params, gradients)
        
        # Print progress every 10 iterations
        if (i + 1) % 10 == 0:
            loss = quadratic_function(params[0])
            print(f"Iteration {i+1}: x = {params[0]:.4f}, f(x) = {loss:.6f}")
    
    print("=" * 50)
    print(f"Final: x = {params[0]:.4f}, f(x) = {quadratic_function(params[0]):.6f}")
    print(f"Expected minimum: x = 3.0, f(x) = 0.0")


if __name__ == "__main__":
    main()
