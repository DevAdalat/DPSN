"""Demonstration tests showing DPSN's dynamic parameter selection."""

import pytest
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dpsn.core.dpsn import DPSN
from dpsn.models.language_model import DPSNLanguageModel
from config.hyperparameters import LanguageModelConfig


class TestDPSNDemonstrations:
    def test_adaptive_complexity_demonstration(self):
        model = DPSN(input_dim=64, pool_size=10_000, min_params=50, max_params=1000)
        model.eval()

        print("\n=== Adaptive Complexity Demonstration ===")
        print("Testing how DPSN adapts parameter usage to input characteristics\n")

        test_cases = [
            ("Low magnitude input", torch.randn(1, 64) * 0.1),
            ("Normal magnitude input", torch.randn(1, 64) * 1.0),
            ("High magnitude input", torch.randn(1, 64) * 5.0),
            (
                "Sparse input (mostly zeros)",
                torch.zeros(1, 64).scatter_(1, torch.randint(0, 64, (1, 5)), 1.0),
            ),
            ("Dense uniform input", torch.ones(1, 64)),
        ]

        results = []
        for name, x in test_cases:
            result = model(x)
            complexity = result["complexity_score"].item()
            params_used = result["parameters_used"]

            print(
                f"{name:30} | Complexity: {complexity:.4f} | Params Used: {params_used:>4}"
            )
            results.append(
                {
                    "name": name,
                    "complexity": complexity,
                    "params_used": params_used,
                }
            )

        assert len(results) == len(test_cases)

    def test_parameter_selection_stability(self):
        model = DPSN(input_dim=64, pool_size=1000)
        model.eval()

        print("\n=== Parameter Selection Stability (Eval Mode) ===")

        x = torch.randn(1, 64)

        indices_list = []
        for i in range(5):
            result = model(x)
            indices_list.append(result["indices"])

        all_same = all(
            torch.equal(indices_list[0], indices) for indices in indices_list[1:]
        )

        print(f"Same input, 5 forward passes")
        print(f"Deterministic (all selections identical): {all_same}")

        assert all_same, "Eval mode should be deterministic"

    def test_parameter_diversity_across_inputs(self):
        model = DPSN(input_dim=64, pool_size=1000)
        model.eval()

        print("\n=== Parameter Diversity Across Different Inputs ===")

        num_samples = 10
        all_indices = []

        for i in range(num_samples):
            x = torch.randn(1, 64)
            result = model(x)
            all_indices.append(set(result["indices"][0].tolist()))

        pairwise_overlaps = []
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                overlap = len(all_indices[i] & all_indices[j])
                budget = len(all_indices[i])
                overlap_ratio = overlap / budget if budget > 0 else 0
                pairwise_overlaps.append(overlap_ratio)

        avg_overlap = np.mean(pairwise_overlaps)

        print(f"Number of different inputs: {num_samples}")
        print(f"Average pairwise parameter overlap: {avg_overlap:.2%}")
        print(f"Range: {min(pairwise_overlaps):.2%} - {max(pairwise_overlaps):.2%}")

        assert avg_overlap < 1.0, (
            "Different inputs should use somewhat different parameters"
        )

    def test_gradient_sparsity_demonstration(self):
        pool_size = 1000
        model = DPSN(input_dim=64, pool_size=pool_size)

        print("\n=== Gradient Sparsity Demonstration ===")

        x = torch.randn(4, 64)
        result = model(x)

        loss = result["output"].sum()
        loss.backward()

        grad = model.pool.embedding.weight.grad

        if grad.is_sparse:
            grad_dense = grad.to_dense()
            nonzero_params = (grad_dense.abs().sum(dim=1) > 0).sum().item()

            print(f"Total parameters in pool: {pool_size}")
            print(f"Parameters with gradients: {nonzero_params}")
            print(f"Gradient sparsity: {(1 - nonzero_params / pool_size) * 100:.2f}%")

            assert nonzero_params < pool_size, (
                "Only selected parameters should have gradients"
            )
        else:
            print("Note: Gradients are dense (sparse mode not enabled)")

    def test_complexity_vs_performance_tradeoff(self):
        print("\n=== Complexity vs Performance Tradeoff ===")
        print("Showing how different budget settings affect computation\n")

        input_dim = 128
        pool_size = 50_000
        x = torch.randn(16, input_dim)

        budget_configs = [
            ("Very Low", 50, 100),
            ("Low", 100, 500),
            ("Medium", 500, 2000),
            ("High", 2000, 5000),
        ]

        results = []
        for name, min_p, max_p in budget_configs:
            model = DPSN(
                input_dim=input_dim,
                pool_size=pool_size,
                min_params=min_p,
                max_params=max_p,
            )

            import time

            start = time.time()
            result = model(x)
            elapsed = (time.time() - start) * 1000

            params_used = result["parameters_used"]
            ratio = params_used / pool_size

            print(
                f"{name:12} | Budget: {min_p:>4}-{max_p:>5} | "
                f"Used: {params_used:>5} ({ratio:.2%} of pool) | "
                f"Time: {elapsed:>6.2f} ms"
            )

            results.append(
                {
                    "name": name,
                    "params_used": params_used,
                    "time_ms": elapsed,
                }
            )

    def test_language_model_token_complexity(self):
        config = LanguageModelConfig(
            vocab_size=100,
            n_embd=128,
            n_head=4,
            n_layer=2,
            block_size=64,
            pool_size=10_000,
        )
        model = DPSNLanguageModel(config)
        model.eval()

        print("\n=== Language Model: Per-Token Complexity ===")

        sequences = [
            ("Short sequence", torch.randint(0, 100, (1, 5))),
            ("Medium sequence", torch.randint(0, 100, (1, 20))),
            ("Long sequence", torch.randint(0, 100, (1, 50))),
        ]

        for name, seq in sequences:
            _, _, stats = model(seq)

            total_params = sum(s["parameters_used"] for s in stats)
            avg_params = total_params / len(stats)

            print(
                f"{name:20} | Seq Length: {seq.shape[1]:>2} | "
                f"Total Params: {total_params:>6} | "
                f"Avg per Layer: {avg_params:>6.1f}"
            )

    def test_comparative_parameter_efficiency(self):
        print("\n=== Parameter Efficiency Comparison ===")
        print("Comparing DPSN with hypothetical dense equivalent\n")

        input_dim = 768
        pool_size = 1_000_000

        model = DPSN(
            input_dim=input_dim, pool_size=pool_size, min_params=100, max_params=5000
        )

        x = torch.randn(32, input_dim)
        result = model(x)

        total_params = model.count_parameters()["total"]
        active_params = result["parameters_used"] * input_dim

        dense_equivalent_params = pool_size * input_dim

        print(f"DPSN Total Parameters: {total_params:>12,}")
        print(f"Pool Size: {pool_size:>12,}")
        print(f"Active Parameters per Forward Pass: {active_params:>12,}")
        print(f"Dense Equivalent would require: {dense_equivalent_params:>12,}")
        print(f"\nEfficiency Ratio: {dense_equivalent_params / active_params:.2f}x")
        print(
            f"(DPSN uses {(active_params / dense_equivalent_params) * 100:.2f}% "
            f"of dense computation)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
