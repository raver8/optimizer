"""
Tests for gradient descent optimizer.
"""

import numpy as np
import pytest
from optimizer import GradientDescentOptimizer
from optimizer.learning_rate import ExponentialDecay, StepDecay


class TestGradientDescentOptimizer:
    """Test cases for GradientDescentOptimizer."""
    
    def test_simple_quadratic_optimization(self):
        """Test optimization of a simple quadratic function."""
        # f(x) = (x - 3)^2, minimum at x = 3
        def loss_fn(params):
            return (params[0] - 3) ** 2
        
        def gradient_fn(params):
            return np.array([2 * (params[0] - 3)])
        
        optimizer = GradientDescentOptimizer(
            learning_rate=0.1,
            max_iterations=100,
            tolerance=1e-6
        )
        
        initial_params = np.array([0.0])
        result = optimizer.optimize(loss_fn, gradient_fn, initial_params)
        
        # Check that we converged close to the minimum
        assert abs(result[0] - 3.0) < 0.01
        
        # Check that loss decreased
        history = optimizer.get_history()
        assert history['loss'][0] > history['loss'][-1]
    
    def test_2d_quadratic_optimization(self):
        """Test optimization of a 2D quadratic function."""
        # f(x, y) = (x - 2)^2 + (y + 1)^2, minimum at (2, -1)
        def loss_fn(params):
            return (params[0] - 2) ** 2 + (params[1] + 1) ** 2
        
        def gradient_fn(params):
            return np.array([
                2 * (params[0] - 2),
                2 * (params[1] + 1)
            ])
        
        optimizer = GradientDescentOptimizer(
            learning_rate=0.1,
            max_iterations=200,
            tolerance=1e-6
        )
        
        initial_params = np.array([0.0, 0.0])
        result = optimizer.optimize(loss_fn, gradient_fn, initial_params)
        
        # Check convergence to minimum
        assert abs(result[0] - 2.0) < 0.01
        assert abs(result[1] - (-1.0)) < 0.01
    
    def test_momentum(self):
        """Test that momentum improves convergence."""
        def loss_fn(params):
            return (params[0] - 5) ** 2
        
        def gradient_fn(params):
            return np.array([2 * (params[0] - 5)])
        
        # Without momentum
        optimizer_no_momentum = GradientDescentOptimizer(
            learning_rate=0.05,
            momentum=0.0,
            max_iterations=50,
            tolerance=1e-10
        )
        result1 = optimizer_no_momentum.optimize(
            loss_fn, gradient_fn, np.array([0.0])
        )
        iterations_no_momentum = len(optimizer_no_momentum.get_history()['loss'])
        
        # With momentum (lower learning rate to prevent overshooting)
        optimizer_with_momentum = GradientDescentOptimizer(
            learning_rate=0.03,
            momentum=0.9,
            max_iterations=50,
            tolerance=1e-10
        )
        result2 = optimizer_with_momentum.optimize(
            loss_fn, gradient_fn, np.array([0.0])
        )
        iterations_with_momentum = len(optimizer_with_momentum.get_history()['loss'])
        
        # With momentum should converge faster
        assert iterations_with_momentum <= iterations_no_momentum
        # Both should find the minimum
        assert abs(result1[0] - 5.0) < 0.1
        assert abs(result2[0] - 5.0) < 0.5
    
    def test_learning_rate_schedule_exponential(self):
        """Test exponential learning rate decay."""
        def loss_fn(params):
            return (params[0] - 1) ** 2
        
        def gradient_fn(params):
            return np.array([2 * (params[0] - 1)])
        
        lr_schedule = ExponentialDecay(
            initial_learning_rate=0.5,
            decay_rate=0.9,
            decay_steps=10
        )
        
        optimizer = GradientDescentOptimizer(
            learning_rate=lr_schedule,
            max_iterations=50
        )
        
        result = optimizer.optimize(loss_fn, gradient_fn, np.array([0.0]))
        
        # Check that learning rate decreased over time
        history = optimizer.get_history()
        assert history['learning_rate'][0] > history['learning_rate'][-1]
        assert abs(result[0] - 1.0) < 0.1
    
    def test_learning_rate_schedule_step(self):
        """Test step learning rate decay."""
        def loss_fn(params):
            return (params[0] - 1) ** 2
        
        def gradient_fn(params):
            return np.array([2 * (params[0] - 1)])
        
        lr_schedule = StepDecay(
            initial_learning_rate=0.1,
            drop_rate=0.5,
            drop_every=10
        )
        
        optimizer = GradientDescentOptimizer(
            learning_rate=lr_schedule,
            max_iterations=50,
            tolerance=1e-10
        )
        
        result = optimizer.optimize(loss_fn, gradient_fn, np.array([0.0]))
        
        # Check that learning rate decreased in steps
        history = optimizer.get_history()
        # Should run long enough to see at least one drop
        assert len(history['learning_rate']) > 10
        assert history['learning_rate'][0] > history['learning_rate'][-1]
        # At iteration 10, learning rate should have dropped
        assert history['learning_rate'][9] > history['learning_rate'][10]
    
    def test_mini_batch_gradient_descent(self):
        """Test mini-batch gradient descent with linear regression."""
        # Create synthetic data: y = 2x + 1
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)
        
        # Loss function: MSE
        def loss_fn(params, X_batch, y_batch):
            predictions = X_batch @ params[:-1].reshape(-1, 1) + params[-1]
            return np.mean((predictions - y_batch) ** 2)
        
        # Gradient function
        def gradient_fn(params, X_batch, y_batch):
            predictions = X_batch @ params[:-1].reshape(-1, 1) + params[-1]
            error = predictions - y_batch
            grad_w = 2 * np.mean(X_batch * error, axis=0)
            grad_b = 2 * np.mean(error)
            return np.append(grad_w, grad_b)
        
        optimizer = GradientDescentOptimizer(
            learning_rate=0.1,
            batch_size=10,
            max_iterations=200
        )
        
        initial_params = np.array([0.0, 0.0])
        result = optimizer.optimize(loss_fn, gradient_fn, initial_params, X, y)
        
        # Should find parameters close to [2, 1]
        assert abs(result[0] - 2.0) < 0.3
        assert abs(result[1] - 1.0) < 0.3
    
    def test_stochastic_gradient_descent(self):
        """Test stochastic gradient descent (batch_size=1)."""
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = 3 * X + 2 + 0.1 * np.random.randn(50, 1)
        
        def loss_fn(params, X_batch, y_batch):
            predictions = X_batch @ params[:-1].reshape(-1, 1) + params[-1]
            return np.mean((predictions - y_batch) ** 2)
        
        def gradient_fn(params, X_batch, y_batch):
            predictions = X_batch @ params[:-1].reshape(-1, 1) + params[-1]
            error = predictions - y_batch
            grad_w = 2 * np.mean(X_batch * error, axis=0)
            grad_b = 2 * np.mean(error)
            return np.append(grad_w, grad_b)
        
        optimizer = GradientDescentOptimizer(
            learning_rate=0.01,
            batch_size=1,
            max_iterations=300
        )
        
        initial_params = np.array([0.0, 0.0])
        result = optimizer.optimize(loss_fn, gradient_fn, initial_params, X, y)
        
        # Should find parameters reasonably close to [3, 2]
        assert abs(result[0] - 3.0) < 0.5
        assert abs(result[1] - 2.0) < 0.5
    
    def test_convergence_tolerance(self):
        """Test that optimizer stops when convergence tolerance is met."""
        def loss_fn(params):
            return (params[0] - 1) ** 2
        
        def gradient_fn(params):
            return np.array([2 * (params[0] - 1)])
        
        optimizer = GradientDescentOptimizer(
            learning_rate=0.3,
            max_iterations=1000,
            tolerance=1e-4
        )
        
        result = optimizer.optimize(loss_fn, gradient_fn, np.array([0.0]))
        
        # Should converge in much fewer than 1000 iterations
        history = optimizer.get_history()
        assert len(history['loss']) < 100
    
    def test_history_tracking(self):
        """Test that optimizer tracks history correctly."""
        def loss_fn(params):
            return params[0] ** 2
        
        def gradient_fn(params):
            return np.array([2 * params[0]])
        
        optimizer = GradientDescentOptimizer(
            learning_rate=0.1,
            max_iterations=10
        )
        
        optimizer.optimize(loss_fn, gradient_fn, np.array([5.0]))
        
        history = optimizer.get_history()
        assert 'loss' in history
        assert 'learning_rate' in history
        assert len(history['loss']) == len(history['learning_rate'])
        assert len(history['loss']) > 0
    
    def test_reset_history(self):
        """Test that history can be reset."""
        def loss_fn(params):
            return params[0] ** 2
        
        def gradient_fn(params):
            return np.array([2 * params[0]])
        
        optimizer = GradientDescentOptimizer(learning_rate=0.1, max_iterations=10)
        optimizer.optimize(loss_fn, gradient_fn, np.array([5.0]))
        
        assert len(optimizer.get_history()['loss']) > 0
        
        optimizer.reset_history()
        
        assert len(optimizer.get_history()['loss']) == 0
        assert len(optimizer.get_history()['learning_rate']) == 0
    
    def test_invalid_learning_rate_type(self):
        """Test that invalid learning rate type raises error."""
        with pytest.raises(ValueError, match="learning_rate must be a number or LearningRateSchedule"):
            GradientDescentOptimizer(learning_rate="invalid")
    
    def test_mini_batch_without_data_raises_error(self):
        """Test that mini-batch mode without data raises error."""
        def loss_fn(params):
            return params[0] ** 2
        
        def gradient_fn(params):
            return np.array([2 * params[0]])
        
        optimizer = GradientDescentOptimizer(
            learning_rate=0.1,
            batch_size=10
        )
        
        with pytest.raises(ValueError, match="X and y must be provided"):
            optimizer.optimize(loss_fn, gradient_fn, np.array([1.0]))
