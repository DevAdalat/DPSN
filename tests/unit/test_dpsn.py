"""Unit tests for complete DPSN module."""

import pytest
import torch
from dpsn.core.dpsn import DPSN
from config.hyperparameters import DPSNConfig


class TestDPSN:
    """Test suite for complete DPSN module."""

    def test_initialization_default(self):
        """Test basic initialization with defaults."""
        model = DPSN(input_dim=64, pool_size=1000)
        assert model.pool.pool_size == 1000
        assert model.pool.param_dim == 64

    def test_initialization_with_config(self):
        """Test initialization with config object."""
        config = DPSNConfig(
            input_dim=128, pool_size=5000, min_params=50, max_params=500
        )
        model = DPSN(config=config)

        assert model.pool.param_dim == 128
        assert model.pool.pool_size == 5000
        assert model.router.min_params == 50
        assert model.router.max_params == 500

    def test_forward_pass(self):
        """Test complete forward pass."""
        model = DPSN(input_dim=64, pool_size=1000)
        x = torch.randn(4, 64)

        result = model(x)

        # Check all expected keys
        assert "output" in result
        assert "complexity_score" in result
        assert "parameters_used" in result
        assert "indices" in result
        assert "weights" in result
        assert "execution_time_ms" in result

        # Check shapes
        assert result["output"].shape == x.shape
        assert result["complexity_score"].shape == (4, 1)

    def test_forward_with_details(self):
        """Test forward pass with detailed execution info."""
        model = DPSN(input_dim=64, pool_size=1000)
        x = torch.randn(2, 64)

        result = model(x, return_details=True)

        assert "execution_details" in result
        assert "selected_params" in result["execution_details"]

    def test_gradient_flow_end_to_end(self):
        """Test gradients flow through entire DPSN."""
        model = DPSN(input_dim=64, pool_size=1000)
        x = torch.randn(2, 64)

        result = model(x)
        loss = result["output"].sum()
        loss.backward()

        # Check gradients in all components
        assert any(p.grad is not None for p in model.router.parameters())
        assert model.pool.embedding.weight.grad is not None

    def test_adaptive_computation(self):
        """Test that model adapts computation based on input."""
        model = DPSN(input_dim=64, pool_size=1000, min_params=50, max_params=200)
        model.eval()  # Deterministic mode

        # Create inputs that should have different complexities
        # Note: actual complexity depends on learned router weights
        x1 = torch.randn(4, 64) * 0.1  # Small magnitude
        x2 = torch.randn(4, 64) * 10.0  # Large magnitude

        result1 = model(x1)
        result2 = model(x2)

        # Both should use budgets within range
        assert result1["parameters_used"] >= 50
        assert result1["parameters_used"] <= 200
        assert result2["parameters_used"] >= 50
        assert result2["parameters_used"] <= 200

    def test_get_statistics(self):
        """Test statistics gathering."""
        model = DPSN(input_dim=64, pool_size=1000)
        x = torch.randn(4, 64)

        stats = model.get_statistics(x)

        assert "pool" in stats
        assert "routing" in stats
        assert "execution" in stats

        assert "mean" in stats["pool"]
        assert "budget" in stats["routing"]
        assert "budget_ratio" in stats["execution"]

    def test_count_parameters(self):
        model = DPSN(input_dim=768, pool_size=1_000_000)

        param_count = model.count_parameters()

        assert "total" in param_count
        assert "pool" in param_count
        assert "router" in param_count
        assert "pool_percentage" in param_count

        assert param_count["total"] > 0
        assert param_count["pool"] > 0
        assert param_count["router"] > 0
        assert param_count["pool"] > param_count["router"]

        param_count = model.count_parameters()

        assert "total" in param_count
        assert "pool" in param_count
        assert "router" in param_count
        assert "pool_percentage" in param_count

        # Pool should dominate parameter count
        assert param_count["pool"] > param_count["router"]
        assert param_count["pool_percentage"] > 50

    def test_training_mode_consistency(self):
        """Test model behavior in train vs eval mode."""
        model = DPSN(input_dim=64, pool_size=1000)
        x = torch.randn(2, 64)

        # Training mode
        model.train()
        result_train = model(x)
        assert model.training

        # Eval mode
        model.eval()
        result_eval1 = model(x)
        result_eval2 = model(x)
        assert not model.training

        # In eval mode, results should be deterministic
        assert torch.equal(result_eval1["indices"], result_eval2["indices"])

    def test_different_batch_sizes(self):
        """Test model with various batch sizes."""
        model = DPSN(input_dim=64, pool_size=1000)

        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 64)
            result = model(x)
            assert result["output"].shape == (batch_size, 64)
            assert result["indices"].shape[0] == batch_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
