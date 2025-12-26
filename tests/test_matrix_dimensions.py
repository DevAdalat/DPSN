import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from dpsn import DPSN


def test_matrix_dimensions_after_picking():
    """
    Test to show matrix dimensions after parameter picking during execution.
    This demonstrates the size and number of matrices involved in the computation.
    """
    print("=" * 80)
    print("DPSN Matrix Dimension Analysis After Parameter Picking")
    print("=" * 80)

    # Initialize model
    input_dim = 64
    pool_size = 5000
    batch_size = 4

    model = DPSN(input_dim=input_dim, pool_size=pool_size)
    model.train()

    # Create test input
    x_test = torch.randn(batch_size, input_dim)

    print(f"\n[SETUP]")
    print(f"Pool size: {pool_size:,} parameters available")
    print(f"Input dimension: {input_dim}")
    print(f"Batch size: {batch_size}")

    # Forward pass through the model
    result = model(x_test)
    budget = result["parameters_used"]
    indices = result["indices"]

    print(f"\n[ROUTER OUTPUT]")
    print(f"Budget (parameters selected): {budget}")
    print(f"Selected indices shape: {indices.shape}")
    print(
        f"  -> Meaning: Each of the {batch_size} samples selected {budget} parameters"
    )

    # Get selected parameters
    selected_params = model.pool.get_params(indices)

    print(f"\n{'=' * 80}")
    print(f"AFTER PARAMETER PICKING - MATRICES INVOLVED IN EXECUTION")
    print(f"{'=' * 80}")

    print(f"\n[PICKED PARAMETERS MATRIX]")
    print(f"Shape: {selected_params.shape}")
    print(f"Breakdown: [Batch={batch_size}, Budget={budget}, ParamDim={input_dim}]")
    print(f"Total elements: {selected_params.numel():,}")
    print(f"Memory size: ~{selected_params.numel() * 4 / 1024 / 1024:.2f} MB (float32)")

    # First matrix multiplication
    x_expanded = x_test.unsqueeze(1)
    params_transposed = selected_params.transpose(1, 2)

    print(f"\n{'=' * 80}")
    print(f"MATRIX MULTIPLICATION #1: Input × Selected Parameters")
    print(f"{'=' * 80}")
    print(f"\nMatrix A (Input expanded):")
    print(f"  Shape: {x_expanded.shape}")
    print(f"  Breakdown: [Batch={batch_size}, 1, InputDim={input_dim}]")

    print(f"\nMatrix B (Selected params transposed):")
    print(f"  Shape: {params_transposed.shape}")
    print(f"  Breakdown: [Batch={batch_size}, ParamDim={input_dim}, Budget={budget}]")

    products = torch.matmul(x_expanded, params_transposed)

    print(f"\nResult (Products):")
    print(f"  Shape: {products.shape}")
    print(f"  Breakdown: [Batch={batch_size}, 1, Budget={budget}]")
    print(
        f"\nOperation: [{batch_size}, 1, {input_dim}] @ [{batch_size}, {input_dim}, {budget}] = [{batch_size}, 1, {budget}]"
    )

    # Second matrix multiplication (after activation and weighting)
    activations = torch.tanh(products)
    weights = result["indices"]  # Use actual weights from router
    # Simulate weights
    weights_sim = torch.softmax(torch.randn(batch_size, budget), dim=1)
    weighted_activations = activations * weights_sim.unsqueeze(1)

    print(f"\n{'=' * 80}")
    print(f"MATRIX MULTIPLICATION #2: Weighted Activations × Selected Parameters")
    print(f"{'=' * 80}")
    print(f"\nMatrix A (Weighted activations):")
    print(f"  Shape: {weighted_activations.shape}")
    print(f"  Breakdown: [Batch={batch_size}, 1, Budget={budget}]")

    print(f"\nMatrix B (Selected params - original):")
    print(f"  Shape: {selected_params.shape}")
    print(f"  Breakdown: [Batch={batch_size}, Budget={budget}, ParamDim={input_dim}]")

    output = torch.matmul(weighted_activations, selected_params)

    print(f"\nResult (Output before squeeze):")
    print(f"  Shape: {output.shape}")
    print(f"  Breakdown: [Batch={batch_size}, 1, ParamDim={input_dim}]")
    print(
        f"\nOperation: [{batch_size}, 1, {budget}] @ [{batch_size}, {budget}, {input_dim}] = [{batch_size}, 1, {input_dim}]"
    )

    output_squeezed = output.squeeze(1)
    print(f"\nFinal Output (after squeeze):")
    print(f"  Shape: {output_squeezed.shape}")
    print(f"  Breakdown: [Batch={batch_size}, ParamDim={input_dim}]")

    # Summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nTotal number of matrices after parameter picking: 3")
    print(f"  1. Selected Parameters: [{batch_size}, {budget}, {input_dim}]")
    print(f"  2. Input (expanded): [{batch_size}, 1, {input_dim}]")
    print(f"  3. Weighted Activations: [{batch_size}, 1, {budget}] (intermediate)")

    print(f"\nTotal number of matrix multiplications: 2")
    print(
        f"  MatMul #1: [{batch_size}, 1, {input_dim}] @ [{batch_size}, {input_dim}, {budget}] = [{batch_size}, 1, {budget}]"
    )
    print(
        f"  MatMul #2: [{batch_size}, 1, {budget}] @ [{batch_size}, {budget}, {input_dim}] = [{batch_size}, 1, {input_dim}]"
    )

    print(f"\nKey Dimensions:")
    print(f"  - Batch: {batch_size} (number of samples processed in parallel)")
    print(f"  - Budget: {budget} (DYNAMIC - changes based on input complexity)")
    print(f"  - ParamDim/InputDim: {input_dim} (fixed model dimension)")

    print(f"\nDynamic Budget Range:")
    print(f"  - Minimum: 100 parameters (simple inputs)")
    print(f"  - Maximum: 5000 parameters (complex inputs)")
    print(f"  - Current: {budget} parameters ({budget / 5000 * 100:.1f}% of max)")

    print(f"\n{'=' * 80}")
    print("Test completed successfully!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    test_matrix_dimensions_after_picking()
