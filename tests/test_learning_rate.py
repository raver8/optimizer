"""
Tests for learning rate schedules.
"""

import pytest
from optimizer.learning_rate import (
    ConstantLearningRate,
    ExponentialDecay,
    StepDecay
)


class TestConstantLearningRate:
    """Test cases for ConstantLearningRate."""
    
    def test_constant_value(self):
        """Test that learning rate stays constant."""
        lr_schedule = ConstantLearningRate(0.1)
        
        for iteration in range(100):
            assert lr_schedule.get_learning_rate(iteration) == 0.1
    
    def test_different_values(self):
        """Test different constant values."""
        for value in [0.001, 0.01, 0.1, 1.0]:
            lr_schedule = ConstantLearningRate(value)
            assert lr_schedule.get_learning_rate(0) == value
            assert lr_schedule.get_learning_rate(100) == value


class TestExponentialDecay:
    """Test cases for ExponentialDecay."""
    
    def test_decay_over_time(self):
        """Test that learning rate decays exponentially."""
        lr_schedule = ExponentialDecay(
            initial_learning_rate=1.0,
            decay_rate=0.9,
            decay_steps=10
        )
        
        lr_0 = lr_schedule.get_learning_rate(0)
        lr_10 = lr_schedule.get_learning_rate(10)
        lr_20 = lr_schedule.get_learning_rate(20)
        
        # Learning rate should decrease
        assert lr_0 > lr_10 > lr_20
        
        # Check specific values
        assert abs(lr_0 - 1.0) < 1e-6
        assert abs(lr_10 - 0.9) < 1e-6
        assert abs(lr_20 - 0.81) < 1e-6
    
    def test_different_decay_rates(self):
        """Test different decay rates."""
        lr_fast = ExponentialDecay(
            initial_learning_rate=1.0,
            decay_rate=0.5,
            decay_steps=10
        )
        
        lr_slow = ExponentialDecay(
            initial_learning_rate=1.0,
            decay_rate=0.95,
            decay_steps=10
        )
        
        # Faster decay should result in lower learning rate
        assert lr_fast.get_learning_rate(50) < lr_slow.get_learning_rate(50)
    
    def test_decay_steps(self):
        """Test that decay_steps controls decay period."""
        lr_schedule = ExponentialDecay(
            initial_learning_rate=1.0,
            decay_rate=0.5,
            decay_steps=100
        )
        
        lr_0 = lr_schedule.get_learning_rate(0)
        lr_100 = lr_schedule.get_learning_rate(100)
        
        # After decay_steps iterations, should be initial_lr * decay_rate
        assert abs(lr_0 - 1.0) < 1e-6
        assert abs(lr_100 - 0.5) < 1e-6


class TestStepDecay:
    """Test cases for StepDecay."""
    
    def test_step_decay(self):
        """Test that learning rate drops in steps."""
        lr_schedule = StepDecay(
            initial_learning_rate=1.0,
            drop_rate=0.5,
            drop_every=10
        )
        
        # Before first drop
        for i in range(10):
            assert lr_schedule.get_learning_rate(i) == 1.0
        
        # After first drop
        for i in range(10, 20):
            assert lr_schedule.get_learning_rate(i) == 0.5
        
        # After second drop
        for i in range(20, 30):
            assert lr_schedule.get_learning_rate(i) == 0.25
    
    def test_different_drop_rates(self):
        """Test different drop rates."""
        lr_schedule = StepDecay(
            initial_learning_rate=1.0,
            drop_rate=0.1,
            drop_every=5
        )
        
        assert lr_schedule.get_learning_rate(0) == 1.0
        assert lr_schedule.get_learning_rate(5) == 0.1
        assert abs(lr_schedule.get_learning_rate(10) - 0.01) < 1e-9
    
    def test_drop_frequency(self):
        """Test different drop frequencies."""
        lr_frequent = StepDecay(
            initial_learning_rate=1.0,
            drop_rate=0.5,
            drop_every=5
        )
        
        lr_infrequent = StepDecay(
            initial_learning_rate=1.0,
            drop_rate=0.5,
            drop_every=20
        )
        
        # More frequent drops should result in lower learning rate
        assert lr_frequent.get_learning_rate(30) < lr_infrequent.get_learning_rate(30)
    
    def test_exact_drop_boundaries(self):
        """Test learning rate at exact drop boundaries."""
        lr_schedule = StepDecay(
            initial_learning_rate=2.0,
            drop_rate=0.5,
            drop_every=10
        )
        
        # Just before drop
        assert lr_schedule.get_learning_rate(9) == 2.0
        # At drop point
        assert lr_schedule.get_learning_rate(10) == 1.0
        # Just after drop
        assert lr_schedule.get_learning_rate(11) == 1.0
