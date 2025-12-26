"""Unit tests for SparseExecutionEngine."""

import pytest
import torch
from dpsn.core.parameter_pool import ParameterPool
from dpsn.core.executor import SparseExecutionEngine


class TestSparseExecutionEngine:
    """Test suite for SparseExecutionEngine component."""

    def test_initialization(self):
        """Test basic initialization."""
        executor = SparseExecutionEngine(input_dim=64)
        assert executor.input_dim == 64
        assert executor.use_residual == True

    def test_forward_output_shape(self):
        """Test that forward pass returns correct shape."""
        pool = ParameterPool(pool_size=1000, param_dim=64)
        executor = SparseExecutionEngine(input_dim=64)

        batch_size = 4
        budget = 10
        x = torch.randn(batch_size, 64)
        indices = torch.randint(0, 1000, (batch_size, budget))
        weights = torch.softmax(torch.randn(batch_size, budget), dim=1)

        output = executor(x, indices, weights, pool)

        assert output.shape == x.shape

    def test_activation_functions(self):
        """Test different activation functions."""
        pool = ParameterPool(pool_size=100, param_dim=32)

        for activation in ["tanh", "relu", "gelu"]:
            executor = SparseExecutionEngine(input_dim=32, activation=activation)

            x = torch.randn(2, 32)
            indices = torch.randint(0, 100, (2, 5))
            weights = torch.softmax(torch.randn(2, 5), dim=1)

            output = executor(x, indices, weights, pool)
            assert output.shape == x.shape

    def test_invalid_activation(self):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError):
            SparseExecutionEngine(input_dim=64, activation="invalid")

    def test_residual_connection(self):
        """Test that residual connection works correctly."""
        pool = ParameterPool(pool_size=100, param_dim=32)

        # With residual
        executor_with = SparseExecutionEngine(input_dim=32, use_residual=True)
        # Without residual
        executor_without = SparseExecutionEngine(input_dim=32, use_residual=False)

        x = torch.randn(2, 32)
        indices = torch.randint(0, 100, (2, 5))
        weights = torch.softmax(torch.randn(2, 5), dim=1)

        # Zero out the pool to test residual clearly
        with torch.no_grad():
            pool.embedding.weight.zero_()

        output_with = executor_with(x, indices, weights, pool)
        output_without = executor_without(x, indices, weights, pool)

        # With residual should equal input (since pool is zero)
        assert torch.allclose(output_with, x, atol=1e-6)
        # Without residual should be close to zero
        assert torch.allclose(output_without, torch.zeros_like(x), atol=1e-6)

    def test_gradient_flow(self):
        """Test that gradients flow through executor."""
        pool = ParameterPool(pool_size=100, param_dim=32, sparse=True)
        executor = SparseExecutionEngine(input_dim=32)

        x = torch.randn(2, 32, requires_grad=True)
        indices = torch.randint(0, 100, (2, 5))
        weights = torch.softmax(torch.randn(2, 5), dim=1)

        output = executor(x, indices, weights, pool)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert pool.embedding.weight.grad is not None

    def test_forward_with_details(self):
        """Test detailed forward pass."""
        pool = ParameterPool(pool_size=100, param_dim=32)
        executor = SparseExecutionEngine(input_dim=32)

        x = torch.randn(2, 32)
        indices = torch.randint(0, 100, (2, 5))
        weights = torch.softmax(torch.randn(2, 5), dim=1)

        result = executor.forward_with_details(x, indices, weights, pool)

        assert "output" in result
        assert "selected_params" in result
        assert "activations" in result
        assert "weighted_activations" in result
        assert "products" in result

        # Check shapes
        assert result["output"].shape == x.shape
        assert result["selected_params"].shape == (2, 5, 32)
        assert result["activations"].shape == (2, 1, 5)

    def test_parameter_usage(self):
        """Test that executor actually uses selected parameters."""
        pool = ParameterPool(pool_size=100, param_dim=32)
        executor = SparseExecutionEngine(input_dim=32)

        x = torch.randn(2, 32)
        indices = torch.randint(0, 100, (2, 5))
        weights = torch.softmax(torch.randn(2, 5), dim=1)

        # Run once
        output1 = executor(x, indices, weights, pool)

        # Modify the selected parameters
        with torch.no_grad():
            pool.embedding.weight[indices] *= 2.0

        # Run again - output should be different
        output2 = executor(x, indices, weights, pool)

        assert not torch.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
