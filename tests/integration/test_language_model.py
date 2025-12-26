"""Integration tests for DPSN Language Model."""

import pytest
import torch
from dpsn.models.language_model import DPSNLanguageModel
from config.hyperparameters import LanguageModelConfig


class TestDPSNLanguageModel:
    """Test suite for DPSN Language Model integration."""

    def test_initialization(self):
        """Test model initialization."""
        config = LanguageModelConfig(
            vocab_size=1000,
            n_embd=128,
            n_head=4,
            n_layer=2,
            block_size=64,
            pool_size=5000,
        )
        model = DPSNLanguageModel(config)

        assert model.config.vocab_size == 1000
        assert len(model.blocks) == 2

    def test_forward_pass(self):
        """Test complete forward pass."""
        config = LanguageModelConfig(
            vocab_size=1000,
            n_embd=128,
            n_head=4,
            n_layer=2,
            block_size=64,
            pool_size=5000,
        )
        model = DPSNLanguageModel(config)

        batch_size = 2
        seq_len = 16
        idx = torch.randint(0, 1000, (batch_size, seq_len))

        logits, loss, stats = model(idx)

        assert logits.shape == (batch_size, seq_len, 1000)
        assert loss is None  # No targets provided
        assert len(stats) == 2  # One per layer

    def test_forward_with_targets(self):
        """Test forward pass with loss calculation."""
        config = LanguageModelConfig(
            vocab_size=1000,
            n_embd=128,
            n_head=4,
            n_layer=2,
            block_size=64,
            pool_size=5000,
        )
        model = DPSNLanguageModel(config)

        batch_size = 2
        seq_len = 16
        idx = torch.randint(0, 1000, (batch_size, seq_len))
        targets = torch.randint(0, 1000, (batch_size, seq_len))

        logits, loss, stats = model(idx, targets)

        assert logits.shape == (batch_size, seq_len, 1000)
        assert loss is not None
        assert isinstance(loss.item(), float)

    def test_generation(self):
        """Test text generation."""
        config = LanguageModelConfig(
            vocab_size=100,
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=32,
            pool_size=1000,
        )
        model = DPSNLanguageModel(config)
        model.eval()

        # Start with a single token
        start_idx = torch.tensor([[5]])
        max_new_tokens = 10

        generated = model.generate(start_idx, max_new_tokens=max_new_tokens)

        assert generated.shape == (1, 1 + max_new_tokens)
        assert (generated >= 0).all()
        assert (generated < 100).all()

    def test_generation_with_temperature(self):
        """Test generation with different temperatures."""
        config = LanguageModelConfig(
            vocab_size=100,
            n_embd=64,
            n_head=4,
            n_layer=1,
            block_size=32,
            pool_size=1000,
        )
        model = DPSNLanguageModel(config)
        model.eval()

        start_idx = torch.tensor([[5]])

        # Low temperature (more deterministic)
        gen_low = model.generate(start_idx, max_new_tokens=5, temperature=0.1)
        # High temperature (more random)
        gen_high = model.generate(start_idx, max_new_tokens=5, temperature=2.0)

        assert gen_low.shape == gen_high.shape == (1, 6)

    def test_generation_with_top_k(self):
        """Test generation with top-k sampling."""
        config = LanguageModelConfig(
            vocab_size=100,
            n_embd=64,
            n_head=4,
            n_layer=1,
            block_size=32,
            pool_size=1000,
        )
        model = DPSNLanguageModel(config)
        model.eval()

        start_idx = torch.tensor([[5]])

        generated = model.generate(start_idx, max_new_tokens=10, top_k=10)

        assert generated.shape == (1, 11)

    def test_gradient_flow_through_model(self):
        """Test that gradients flow through entire model."""
        config = LanguageModelConfig(
            vocab_size=100,
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=32,
            pool_size=1000,
        )
        model = DPSNLanguageModel(config)

        idx = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))

        logits, loss, stats = model(idx, targets)
        loss.backward()

        # Check gradients in various components
        assert model.token_embedding_table.weight.grad is not None
        assert any(p.grad is not None for p in model.blocks[0].sa.parameters())
        assert model.blocks[0].dpsn.pool.embedding.weight.grad is not None

    def test_get_model_statistics(self):
        """Test model statistics gathering."""
        config = LanguageModelConfig(
            vocab_size=100,
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=32,
            pool_size=1000,
        )
        model = DPSNLanguageModel(config)

        idx = torch.randint(0, 100, (2, 16))
        stats = model.get_model_statistics(idx)

        assert "num_blocks" in stats
        assert "total_params_used" in stats
        assert "avg_complexity" in stats
        assert stats["num_blocks"] == 2

    def test_count_parameters(self):
        """Test parameter counting."""
        config = LanguageModelConfig(
            vocab_size=100,
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=32,
            pool_size=1000,
        )
        model = DPSNLanguageModel(config)

        param_count = model.count_parameters()

        assert "total" in param_count
        assert "embeddings" in param_count
        assert "dpsn_pools" in param_count
        assert "attention" in param_count

        # DPSN pools should be significant portion
        assert param_count["dpsn_pools"] > 0

    def test_multiple_layers_execute(self):
        """Test that multiple DPSN layers execute correctly."""
        config = LanguageModelConfig(
            vocab_size=100,
            n_embd=64,
            n_head=4,
            n_layer=4,  # Multiple layers
            block_size=32,
            pool_size=1000,
        )
        model = DPSNLanguageModel(config)

        idx = torch.randint(0, 100, (2, 16))
        logits, loss, stats = model(idx)

        assert len(stats) == 4
        # Each layer should have produced output
        for stat in stats:
            assert "parameters_used" in stat
            assert stat["parameters_used"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
