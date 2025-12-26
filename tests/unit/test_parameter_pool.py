"""Unit tests for ParameterPool."""

import pytest
import torch
from dpsn.core.parameter_pool import ParameterPool


class TestParameterPool:
    """Test suite for ParameterPool component."""

    def test_initialization(self):
        """Test basic initialization."""
        pool = ParameterPool(pool_size=1000, param_dim=64)
        assert pool.pool_size == 1000
        assert pool.param_dim == 64
        assert pool.params.shape == (1000, 64)

    def test_get_params_shape(self):
        """Test parameter fetching returns correct shapes."""
        pool = ParameterPool(pool_size=1000, param_dim=64)
        batch_size = 4
        budget = 10

        indices = torch.randint(0, 1000, (batch_size, budget))
        params = pool.get_params(indices)

        assert params.shape == (batch_size, budget, 64)

    def test_sparse_gradients(self):
        """Test that only selected parameters receive gradients."""
        pool = ParameterPool(pool_size=1000, param_dim=64, sparse=True)

        # Select only indices 0, 1, 2
        indices = torch.tensor([[0, 1, 2]])
        params = pool.get_params(indices)

        # Compute loss and backward
        loss = params.sum()
        loss.backward()

        # Check that embedding has sparse gradient
        assert pool.embedding.weight.grad.is_sparse

    def test_get_param_statistics(self):
        """Test parameter statistics calculation."""
        pool = ParameterPool(pool_size=100, param_dim=32)
        stats = pool.get_param_statistics()

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "norm" in stats

        # Check that values are reasonable
        assert isinstance(stats["mean"], float)
        assert stats["std"] > 0

    def test_reset_parameters_all(self):
        """Test resetting all parameters."""
        pool = ParameterPool(pool_size=100, param_dim=32)
        original_params = pool.params.clone()

        pool.reset_parameters(init_std=0.1)

        # Parameters should be different
        assert not torch.allclose(original_params, pool.params)

    def test_reset_parameters_selective(self):
        """Test resetting specific parameters."""
        pool = ParameterPool(pool_size=100, param_dim=32)

        # Reset only indices 0, 1, 2
        indices = torch.tensor([0, 1, 2])
        original_params = pool.params[indices].clone()
        unchanged_params = pool.params[3:].clone()

        pool.reset_parameters(indices=indices, init_std=0.1)

        # Selected params should change
        assert not torch.allclose(original_params, pool.params[indices])
        # Other params should remain unchanged
        assert torch.allclose(unchanged_params, pool.params[3:])

    def test_different_devices(self):
        """Test parameter pool on different devices."""
        pool = ParameterPool(pool_size=100, param_dim=32)

        if torch.cuda.is_available():
            pool_cuda = pool.cuda()
            indices = torch.tensor([[0, 1, 2]], device="cuda")
            params = pool_cuda.get_params(indices)
            assert params.device.type == "cuda"
        else:
            # CPU test
            indices = torch.tensor([[0, 1, 2]])
            params = pool.get_params(indices)
            assert params.device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
