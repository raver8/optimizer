"""Tests for Adam optimizer."""

import unittest
from optimizer import AdamOptimizer


class TestAdamOptimizer(unittest.TestCase):
    """Test cases for AdamOptimizer."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = AdamOptimizer(learning_rate=0.001)
        self.assertEqual(optimizer.learning_rate, 0.001)
        self.assertEqual(optimizer.beta1, 0.9)
        self.assertEqual(optimizer.beta2, 0.999)
        self.assertEqual(optimizer.epsilon, 1e-8)
        self.assertEqual(optimizer.iterations, 0)
    
    def test_initialization_with_custom_params(self):
        """Test optimizer initialization with custom parameters."""
        optimizer = AdamOptimizer(
            learning_rate=0.01,
            beta1=0.8,
            beta2=0.99,
            epsilon=1e-7
        )
        self.assertEqual(optimizer.learning_rate, 0.01)
        self.assertEqual(optimizer.beta1, 0.8)
        self.assertEqual(optimizer.beta2, 0.99)
        self.assertEqual(optimizer.epsilon, 1e-7)
    
    def test_invalid_learning_rate(self):
        """Test that negative learning rate raises ValueError."""
        with self.assertRaises(ValueError):
            AdamOptimizer(learning_rate=-0.01)
    
    def test_invalid_beta1(self):
        """Test that invalid beta1 raises ValueError."""
        with self.assertRaises(ValueError):
            AdamOptimizer(beta1=1.5)
        with self.assertRaises(ValueError):
            AdamOptimizer(beta1=-0.1)
    
    def test_invalid_beta2(self):
        """Test that invalid beta2 raises ValueError."""
        with self.assertRaises(ValueError):
            AdamOptimizer(beta2=1.5)
        with self.assertRaises(ValueError):
            AdamOptimizer(beta2=-0.1)
    
    def test_invalid_epsilon(self):
        """Test that non-positive epsilon raises ValueError."""
        with self.assertRaises(ValueError):
            AdamOptimizer(epsilon=-1e-8)
        with self.assertRaises(ValueError):
            AdamOptimizer(epsilon=0)
    
    def test_step_single_parameter(self):
        """Test single optimization step with one parameter."""
        optimizer = AdamOptimizer(learning_rate=0.1)
        params = [1.0]
        gradients = [2.0]
        
        updated_params = optimizer.step(params, gradients)
        
        # Parameter should move in negative gradient direction
        self.assertLess(updated_params[0], params[0])
        self.assertEqual(optimizer.iterations, 1)
    
    def test_step_multiple_parameters(self):
        """Test single optimization step with multiple parameters."""
        optimizer = AdamOptimizer(learning_rate=0.1)
        params = [1.0, 2.0, 3.0]
        gradients = [1.0, -1.0, 0.5]
        
        updated_params = optimizer.step(params, gradients)
        
        self.assertEqual(len(updated_params), 3)
        # First param should decrease (positive gradient)
        self.assertLess(updated_params[0], params[0])
        # Second param should increase (negative gradient)
        self.assertGreater(updated_params[1], params[1])
        # Third param should decrease (positive gradient)
        self.assertLess(updated_params[2], params[2])
    
    def test_step_mismatched_lengths(self):
        """Test that mismatched params and gradients raise ValueError."""
        optimizer = AdamOptimizer()
        params = [1.0, 2.0]
        gradients = [1.0]
        
        with self.assertRaises(ValueError):
            optimizer.step(params, gradients)
    
    def test_multiple_steps(self):
        """Test multiple optimization steps."""
        optimizer = AdamOptimizer(learning_rate=0.1)
        params = [5.0]
        
        for _ in range(10):
            # Gradient pointing towards zero
            gradients = [params[0]]
            params = optimizer.step(params, gradients)
        
        # After multiple steps, parameter should move towards zero
        self.assertLess(abs(params[0]), 5.0)
        self.assertEqual(optimizer.iterations, 10)
    
    def test_get_params(self):
        """Test get_params method."""
        optimizer = AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999)
        
        # Before any steps
        params_dict = optimizer.get_params()
        self.assertEqual(params_dict['learning_rate'], 0.01)
        self.assertEqual(params_dict['beta1'], 0.9)
        self.assertEqual(params_dict['beta2'], 0.999)
        self.assertEqual(params_dict['iterations'], 0)
        self.assertIsNone(params_dict['m'])
        self.assertIsNone(params_dict['v'])
        
        # After a step
        optimizer.step([1.0], [0.5])
        params_dict = optimizer.get_params()
        self.assertEqual(params_dict['iterations'], 1)
        self.assertIsNotNone(params_dict['m'])
        self.assertIsNotNone(params_dict['v'])
    
    def test_convergence_quadratic(self):
        """Test convergence on a simple quadratic function."""
        optimizer = AdamOptimizer(learning_rate=0.1)
        
        # Minimize (x - 3)^2, gradient is 2(x - 3)
        params = [0.0]
        target = 3.0
        
        for _ in range(100):
            gradient = 2 * (params[0] - target)
            params = optimizer.step(params, [gradient])
        
        # Should converge close to the minimum at x=3
        self.assertAlmostEqual(params[0], target, delta=0.1)


if __name__ == '__main__':
    unittest.main()
