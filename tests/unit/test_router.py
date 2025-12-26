"""Unit tests for RouteGeneratorModel."""

import pytest
import torch
from dpsn.core.router import RouteGeneratorModel


class TestRouteGeneratorModel:
    """Test suite for RouteGeneratorModel component."""

    def test_initialization(self):
        """Test basic initialization."""
        router = RouteGeneratorModel(
            input_dim=64, pool_size=1000, min_params=10, max_params=100
        )
        assert router.pool_size == 1000
        assert router.min_params == 10
        assert router.max_params == 100

    def test_forward_output_shapes(self):
        """Test that forward pass returns correct shapes."""
        router = RouteGeneratorModel(
            input_dim=64, pool_size=1000, min_params=10, max_params=100
        )

        batch_size = 4
        x = torch.randn(batch_size, 64)

        indices, budget, weights, complexity = router(x)

        # Check shapes
        assert indices.shape == (batch_size, budget)
        assert isinstance(budget, int)
        assert weights.shape == (batch_size, budget)
        assert complexity.shape == (batch_size, 1)

        # Check value ranges
        assert budget >= router.min_params
        assert budget <= router.max_params
        assert (complexity >= 0).all() and (complexity <= 1).all()

    def test_budget_determination(self):
        """Test budget determination logic."""
        router = RouteGeneratorModel(
            input_dim=64, pool_size=1000, min_params=100, max_params=500
        )

        # Test with different complexity scores
        low_complexity = torch.tensor([[0.1], [0.1], [0.1]])
        high_complexity = torch.tensor([[0.9], [0.9], [0.9]])

        low_budget = router.determine_budget(low_complexity)
        high_budget = router.determine_budget(high_complexity)

        # Higher complexity should result in higher budget
        assert high_budget > low_budget
        assert low_budget >= router.min_params
        assert high_budget <= router.max_params

    def test_training_vs_eval_mode(self):
        """Test that training mode adds noise while eval mode is deterministic."""
        router = RouteGeneratorModel(input_dim=64, pool_size=1000, noise_std=0.1)

        x = torch.randn(2, 64)

        # Training mode - might produce different results
        router.train()
        indices1, _, _, _ = router(x)
        indices2, _, _, _ = router(x)
        # With noise, indices might differ (not guaranteed but likely)

        # Eval mode - should be deterministic
        router.eval()
        indices3, _, _, _ = router(x)
        indices4, _, _, _ = router(x)
        assert torch.equal(indices3, indices4)

    def test_weights_sum_to_one(self):
        """Test that returned weights are properly normalized."""
        router = RouteGeneratorModel(input_dim=64, pool_size=1000)
        x = torch.randn(4, 64)

        _, _, weights, _ = router(x)

        # Weights should sum to 1 along dim=1
        weight_sums = weights.sum(dim=1)
        assert torch.allclose(weight_sums, torch.ones(4), atol=1e-6)

    def test_complexity_exponent_effect(self):
        """Test that complexity exponent affects budget scaling."""
        router_linear = RouteGeneratorModel(
            input_dim=64,
            pool_size=1000,
            min_params=100,
            max_params=500,
            complexity_exponent=1.0,  # Linear
        )

        router_quadratic = RouteGeneratorModel(
            input_dim=64,
            pool_size=1000,
            min_params=100,
            max_params=500,
            complexity_exponent=2.0,  # Quadratic
        )

        mid_complexity = torch.tensor([[0.5], [0.5]])

        budget_linear = router_linear.determine_budget(mid_complexity)
        budget_quadratic = router_quadratic.determine_budget(mid_complexity)

        # Quadratic should be lower for mid-range complexity
        assert budget_quadratic < budget_linear

    def test_get_routing_statistics(self):
        """Test routing statistics calculation."""
        router = RouteGeneratorModel(input_dim=64, pool_size=1000)
        x = torch.randn(4, 64)

        stats = router.get_routing_statistics(x)

        assert "mean_complexity" in stats
        assert "budget" in stats
        assert "budget_ratio" in stats
        assert stats["budget_ratio"] <= 1.0
        assert stats["budget"] > 0

    def test_indices_within_pool_size(self):
        """Test that all selected indices are valid."""
        router = RouteGeneratorModel(input_dim=64, pool_size=1000)
        x = torch.randn(8, 64)

        indices, _, _, _ = router(x)

        # All indices should be in [0, pool_size)
        assert (indices >= 0).all()
        assert (indices < router.pool_size).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
