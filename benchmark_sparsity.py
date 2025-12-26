import torch
import torch.nn as nn
import time
import psutil


def get_ram_usage():
    return psutil.virtual_memory().percent


def benchmark_dense_vs_sparse():
    print("=== Benchmarking Backward Pass Speed ===")

    # Configuration mimicking your setup
    POOL_SIZE = 1_300_000
    PARAM_DIM = 768
    BATCH_SIZE = 4
    BUDGET = 1000  # Number of params picked

    print(f"Pool Size: {POOL_SIZE:,} x {PARAM_DIM}")
    print(f"Total Params: {POOL_SIZE * PARAM_DIM:,}")
    print(f"Initial RAM: {get_ram_usage()}%")

    # --- Scenario 1: Current Implementation (Dense nn.Parameter) ---
    print("\n1. Testing Current Implementation (nn.Parameter)...")
    try:
        # Create a large tensor like in your code
        params = nn.Parameter(torch.randn(POOL_SIZE, PARAM_DIM))
        indices = torch.randint(0, POOL_SIZE, (BATCH_SIZE, BUDGET))

        # Forward
        t0 = time.time()
        selected = params[indices]  # [Batch, Budget, Dim]
        loss = selected.sum()
        print(f"   Forward Time: {time.time() - t0:.4f}s")

        # Backward
        t0 = time.time()
        loss.backward()
        dt = time.time() - t0
        print(f"   Backward Time: {dt:.4f}s")

        if params.grad is not None:
            print(f"   Gradient Type: {type(params.grad)}")
            print(f"   Gradient Shape: {params.grad.shape} (Allocated full matrix!)")
        else:
            print("   Gradient is None!")

        # Cleanup to free RAM
        del params, indices, selected, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        print(f"   FAILED: {e}")

    # --- Scenario 2: Proposed Solution (nn.Embedding sparse=True) ---
    print("\n2. Testing Proposed Solution (nn.Embedding sparse=True)...")
    try:
        # Use Embedding with sparse=True
        embedding = nn.Embedding(POOL_SIZE, PARAM_DIM, sparse=True)
        indices = torch.randint(0, POOL_SIZE, (BATCH_SIZE, BUDGET))

        # Forward
        t0 = time.time()
        selected = embedding(indices)
        loss = selected.sum()
        print(f"   Forward Time: {time.time() - t0:.4f}s")

        # Backward
        t0 = time.time()
        loss.backward()
        dt = time.time() - t0
        print(f"   Backward Time: {dt:.4f}s")

        if embedding.weight.grad is not None:
            print(f"   Gradient Type: {type(embedding.weight.grad)}")
            print(f"   Gradient is Coalesced: {embedding.weight.grad.is_coalesced()}")
            print(f"   Gradient Indices Size: {embedding.weight.grad._indices().shape}")

    except Exception as e:
        print(f"   FAILED: {e}")


if __name__ == "__main__":
    benchmark_dense_vs_sparse()
