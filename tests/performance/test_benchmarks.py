"""Performance benchmarking suite for DPSN."""

import pytest
import torch
import time
import psutil
import os
from dpsn.core.dpsn import DPSN
from dpsn.models.language_model import DPSNLanguageModel
from config.hyperparameters import DPSNConfig, LanguageModelConfig


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class TestDPSNPerformance:
    @pytest.mark.benchmark
    def test_forward_pass_speed(self, benchmark):
        model = DPSN(input_dim=768, pool_size=100_000)
        x = torch.randn(32, 768)

        result = benchmark(model, x)
        assert result["output"].shape == x.shape

    @pytest.mark.benchmark
    def test_scaling_with_pool_size(self):
        batch_size = 16
        input_dim = 768
        x = torch.randn(batch_size, input_dim)

        results = []
        for pool_size in [1_000, 10_000, 100_000, 1_000_000]:
            model = DPSN(input_dim=input_dim, pool_size=pool_size)

            start_time = time.time()
            _ = model(x)
            elapsed = (time.time() - start_time) * 1000

            results.append(
                {
                    "pool_size": pool_size,
                    "time_ms": elapsed,
                }
            )

        print("\n=== Scaling with Pool Size ===")
        for r in results:
            print(f"Pool Size: {r['pool_size']:>10,} | Time: {r['time_ms']:>8.2f} ms")

    @pytest.mark.benchmark
    def test_scaling_with_batch_size(self):
        pool_size = 100_000
        input_dim = 768

        results = []
        for batch_size in [1, 4, 16, 64, 256]:
            model = DPSN(input_dim=input_dim, pool_size=pool_size)
            x = torch.randn(batch_size, input_dim)

            start_time = time.time()
            _ = model(x)
            elapsed = (time.time() - start_time) * 1000

            results.append(
                {
                    "batch_size": batch_size,
                    "time_ms": elapsed,
                    "time_per_sample_ms": elapsed / batch_size,
                }
            )

        print("\n=== Scaling with Batch Size ===")
        for r in results:
            print(
                f"Batch: {r['batch_size']:>3} | Total: {r['time_ms']:>8.2f} ms | "
                f"Per Sample: {r['time_per_sample_ms']:>6.2f} ms"
            )

    @pytest.mark.benchmark
    def test_memory_usage_with_sparse_gradients(self):
        pool_size = 1_000_000
        input_dim = 768
        batch_size = 32

        mem_before = get_memory_usage()

        model_sparse = DPSN(input_dim=input_dim, pool_size=pool_size)
        x = torch.randn(batch_size, input_dim)

        result = model_sparse(x)
        loss = result["output"].sum()

        mem_after_forward = get_memory_usage()

        loss.backward()

        mem_after_backward = get_memory_usage()

        print(f"\n=== Memory Usage (1M Pool, Sparse Gradients) ===")
        print(f"Before:         {mem_before:>8.2f} MB")
        print(
            f"After Forward:  {mem_after_forward:>8.2f} MB (+{mem_after_forward - mem_before:.2f} MB)"
        )
        print(
            f"After Backward: {mem_after_backward:>8.2f} MB (+{mem_after_backward - mem_after_forward:.2f} MB)"
        )

    @pytest.mark.benchmark
    def test_language_model_generation_speed(self):
        config = LanguageModelConfig(
            vocab_size=1000,
            n_embd=256,
            n_head=8,
            n_layer=4,
            block_size=128,
            pool_size=50_000,
        )
        model = DPSNLanguageModel(config)
        model.eval()

        start_idx = torch.tensor([[5]])

        tokens_to_generate = [10, 50, 100]
        results = []

        for num_tokens in tokens_to_generate:
            start_time = time.time()
            _ = model.generate(start_idx, max_new_tokens=num_tokens)
            elapsed = (time.time() - start_time) * 1000

            results.append(
                {
                    "tokens": num_tokens,
                    "time_ms": elapsed,
                    "tokens_per_sec": num_tokens / (elapsed / 1000),
                }
            )

        print("\n=== Language Model Generation Speed ===")
        for r in results:
            print(
                f"Tokens: {r['tokens']:>3} | Time: {r['time_ms']:>8.2f} ms | "
                f"Speed: {r['tokens_per_sec']:>6.2f} tok/s"
            )

    @pytest.mark.benchmark
    def test_parameter_budget_overhead(self):
        pool_size = 100_000
        input_dim = 768
        batch_size = 32
        x = torch.randn(batch_size, input_dim)

        results = []
        for max_params in [100, 500, 1000, 5000]:
            model = DPSN(
                input_dim=input_dim,
                pool_size=pool_size,
                min_params=max_params // 10,
                max_params=max_params,
            )

            start_time = time.time()
            result = model(x)
            elapsed = (time.time() - start_time) * 1000

            results.append(
                {
                    "max_budget": max_params,
                    "actual_budget": result["parameters_used"],
                    "time_ms": elapsed,
                }
            )

        print("\n=== Parameter Budget vs Speed ===")
        for r in results:
            print(
                f"Max Budget: {r['max_budget']:>5} | Actual: {r['actual_budget']:>5} | "
                f"Time: {r['time_ms']:>8.2f} ms"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
