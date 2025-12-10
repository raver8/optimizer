"""Tests for base optimizer class."""

import unittest
from optimizer.base import Optimizer


class ConcreteOptimizer(Optimizer):
    """Concrete implementation for testing."""
    
    def step(self, params, gradients):
        """Simple gradient descent step."""
        return [p - self.learning_rate * g for p, g in zip(params, gradients)]
    
    def get_params(self):
        """Return optimizer parameters."""
        return {'learning_rate': self.learning_rate, 'iterations': self.iterations}


class TestBaseOptimizer(unittest.TestCase):
    """Test cases for base Optimizer class."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = ConcreteOptimizer(learning_rate=0.01)
        self.assertEqual(optimizer.learning_rate, 0.01)
        self.assertEqual(optimizer.iterations, 0)
    
    def test_default_learning_rate(self):
        """Test default learning rate."""
        optimizer = ConcreteOptimizer()
        self.assertEqual(optimizer.learning_rate, 0.01)
    
    def test_invalid_learning_rate(self):
        """Test that invalid learning rate raises ValueError."""
        with self.assertRaises(ValueError):
            ConcreteOptimizer(learning_rate=0)
        with self.assertRaises(ValueError):
            ConcreteOptimizer(learning_rate=-0.01)
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with self.assertRaises(TypeError):
            # Cannot instantiate abstract class
            Optimizer(learning_rate=0.01)


if __name__ == '__main__':
    unittest.main()
