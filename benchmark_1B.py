import torch
import torch.nn as nn
import time
import psutil
import os
from src.dpsn import DPSN


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3  # GB


def benchmark_1B_model():
    print("=== DPSN 1 Billion Parameter Benchmark (CPU) ===")

    # Configuration for ~1B Parameters
    # Pool: 1,000,000 slots * 1024 dim = 1,024,000,000 parameters
    pool_size = 1_000_000
    input_dim = 1024

    print(f"Configuration:")
    print(f"  - Input Dim: {input_dim}")
    print(f"  - Pool Size: {pool_size}")
    print(f"  - Target Params: ~1.02 Billion in Pool")
    print(f"  - Device: CPU")

    # 1. Initialization
    print("\n[Phase 1] Initializing Model...")
    mem_start = get_process_memory()
    start_time = time.time()

    model = DPSN(input_dim=input_dim, pool_size=pool_size)

    end_time = time.time()
    mem_end = get_process_memory()

    print(f"  - Init Time: {end_time - start_time:.2f} seconds")
    print(f"  - Memory Usage: {mem_end - mem_start:.2f} GB (Total: {mem_end:.2f} GB)")

    # Calculate exact parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Exact Parameter Count: {total_params:,}")

    # 2. Forward Pass Benchmark
    print("\n[Phase 2] Forward Pass Benchmark (Inference)...")
    batch_size = 1
    dummy_input = torch.randn(batch_size, input_dim)

    # Warmup
    _ = model(dummy_input)

    iterations = 10
    start_time = time.time()
    for i in range(iterations):
        _ = model(dummy_input)
    end_time = time.time()

    avg_fwd = (end_time - start_time) / iterations
    print(f"  - Avg Forward Time: {avg_fwd * 1000:.2f} ms / sample")
    print(f"  - Est. Throughput: {1 / avg_fwd:.2f} samples/sec")

    # 3. Training Step Benchmark (Forward + Backward + Update)
    print("\n[Phase 3] Training Step Benchmark...")
    criterion = nn.MSELoss()

    # Setup Optimizers (as per sparse training requirement)
    pool_params = list(model.pool.parameters())
    pool_ids = list(map(id, pool_params))
    router_params = filter(lambda p: id(p) not in pool_ids, model.parameters())

    opt_router = torch.optim.AdamW(router_params, lr=1e-3)
    opt_pool = torch.optim.SGD(pool_params, lr=1e-2)  # SGD for sparsity

    dummy_target = torch.randn(batch_size, input_dim)

    start_time = time.time()
    for i in range(iterations):
        opt_router.zero_grad()
        opt_pool.zero_grad()

        out = model(dummy_input)["output"]
        loss = criterion(out, dummy_target)
        loss.backward()

        opt_router.step()
        opt_pool.step()

        print(f"    Step {i + 1}/{iterations} completed", end="\r")

    end_time = time.time()

    avg_train = (end_time - start_time) / iterations
    print(f"\n  - Avg Training Step Time: {avg_train:.2f} sec / step")

    print("\n=== Benchmark Complete ===")

    # Verdict
    if avg_train < 1.0:
        print("Verdict: EXCELLENT. Your machine can train this very fast.")
    elif avg_train < 5.0:
        print("Verdict: GOOD. Training is feasible.")
    else:
        print("Verdict: SLOW. It works, but full training will take time.")


if __name__ == "__main__":
    benchmark_1B_model()
