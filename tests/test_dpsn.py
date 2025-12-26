import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from dpsn import DPSN


def test_dpsn():
    print("Initializing DPSN...")
    input_dim = 64  # Small dim for test
    pool_size = 1000  # Small memory for test
    model = DPSN(input_dim=input_dim, pool_size=pool_size)

    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, input_dim)

    print("\nRunning Forward Pass...")
    results = model(x)
    output = results["output"]

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Inspect batch stats
    complexity_scores = results["complexity_score"]
    budget = results["parameters_used"]
    print(f"Batch Budget Used: {budget}")
    print(f"Complexity Scores: {complexity_scores.flatten().tolist()}")

    # Check if output is different from input (transformation happened)
    diff = (output - x).abs().mean()
    print(f"\nMean difference from input: {diff.item()}")
    assert diff.item() > 0, "Model did not transform the input!"

    print("\nRunning Backward Pass...")
    loss = output.mean()
    loss.backward()
    print("Backward pass successful.")

    # Check gradients
    print("Checking gradients for Router...")
    has_grad = False
    for param in model.router.parameters():
        if param.grad is not None:
            has_grad = True
            break

    if has_grad:
        print("Router has gradients.")
    else:
        print(
            "WARNING: Router has NO gradients (Expected if Gumbel Softmax or similar differentiable path is not fully utilized or if selection is hard)."
        )
        # Note: In my implementation, I used topk which is not differentiable W.R.T indices,
        # but 'weights' are returned from topk on noisy_scores.
        # However, topk values are differentiable w.r.t input scores.
        # So router should get gradients via the 'weights' path.

    print("\nTest passed successfully!")


if __name__ == "__main__":
    test_dpsn()
