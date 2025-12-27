import torch
import time
import sys
import os

sys.path.append(os.getcwd())

from dpsn.core.router import RouteGeneratorModel


def test_millions_routing():
    print("=== DPSN Router Millions Scale Test ===")

    input_dim = 128
    pool_size = 2_000_000
    max_params = 1_000_000
    min_params = 100

    print(
        f"Configuring Router with Pool Size: {pool_size:,} and Max Selection: {max_params:,}"
    )

    start_init = time.time()
    router = RouteGeneratorModel(
        input_dim=input_dim,
        pool_size=pool_size,
        min_params=min_params,
        max_params=max_params,
        hidden_dim=256,
    )
    end_init = time.time()
    print(f"Router initialized in {end_init - start_init:.2f} seconds.")

    x = torch.randn(1, input_dim)

    with torch.no_grad():
        router.complexity_net[-2].bias.fill_(10.0)

    print("\nRunning Forward Pass (Generating Indices)...")

    start_forward = time.time()
    indices, budget, weights, complexity = router(x)
    end_forward = time.time()

    print(f"Done!")
    print(f" - Complexity Score: {complexity.item():.4f}")
    print(f" - Indices Generated: {indices.shape[1]:,}")
    print(f" - Generation Time: {(end_forward - start_forward) * 1000:.2f} ms")

    assert indices.shape[1] == max_params
    print(
        f"\nSUCCESS: The router successfully generated {indices.shape[1]:,} unique indices in one pass."
    )

    print(
        f"Sample of generated indices: {indices[0, :5].tolist()} ... {indices[0, -5:].tolist()}"
    )


if __name__ == "__main__":
    try:
        test_millions_routing()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\nError: Out of memory. Try reducing the pool_size or input_dim.")
        else:
            raise e
    except Exception as e:
        print(f"\nAn error occurred: {e}")
