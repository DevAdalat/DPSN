import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List
import numpy as np
from dpsn import DPSN


class DPSNTrainer:
    def __init__(
        self,
        model: DPSN,
        learning_rate: float = 0.001,
        sparsity_penalty_weight: float = 0.01,
    ):
        self.model = model
        self.sparsity_penalty_weight = sparsity_penalty_weight

        # We separate parameters for potentially different learning rates
        # 1. Router parameters
        self.router_optimizer = optim.Adam(
            self.model.router.parameters(), lr=learning_rate
        )

        # 2. Pool parameters (The knowledge bank)
        # We use SGD for the pool to ensure strict sparse updates without
        # Adam's momentum updating non-selected params (via moving averages)
        # unless we strictly manage it. For this prototype, SGD is safest
        # to demonstrate "only selected params update".
        self.pool_optimizer = optim.SGD(
            self.model.pool.parameters(),
            lr=learning_rate
            * 10.0,  # Higher LR for pool as they are updated less frequently
        )

        self.criterion = nn.MSELoss()

    def compute_auxiliary_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Encourages the router to not be too confident too early (entropy)
        or other structural constraints.
        For now, we just penalize extremely high budgets if we wanted,
        but let's focus on task loss first.
        """
        return 0.0

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict:
        """
        Performs a single training step.
        """
        self.model.train()
        self.router_optimizer.zero_grad()
        self.pool_optimizer.zero_grad()

        # 1. Forward Pass
        # x: [1, InputDim]
        result = self.model(x)
        prediction = result["output"]
        indices = result["indices"]  # [1, Budget]

        # 2. Compute Loss
        task_loss = self.criterion(prediction, y)

        # Optional: Add penalty if budget is exploding (optional)
        # budget_penalty = result['complexity_score'] * 0.1

        total_loss = task_loss

        # 3. Backward Pass
        total_loss.backward()

        # 4. Sparse Update Verification (For debug/demo)
        # We want to confirm that gradients exist ONLY for selected indices in the pool
        pool_grad = self.model.pool.params.grad

        # In PyTorch, indexing (params[indices]) produces a sparse gradient update
        # structure effectively. The gradient tensor 'pool_grad' will likely be dense
        # but mostly zeros.

        # 5. Optimizer Step
        self.router_optimizer.step()
        self.pool_optimizer.step()

        return {
            "loss": total_loss.item(),
            "budget": result["parameters_used"],
            "complexity": result["complexity_score"],
            "indices": indices.detach().cpu().numpy(),
        }


def generate_dummy_data(batch_size=100, input_dim=64):
    """
    Generates data where different input regions require different transformations.
    """
    x = torch.randn(batch_size, input_dim)
    # Target is a complex non-linear function of x
    # y = sin(x) + x^2 (element wise) roughly
    y = torch.sin(x) + (x**2) * 0.1
    return x, y


def train_loop():
    input_dim = 64
    pool_size = 10_000  # Reduced for quick testing
    model = DPSN(input_dim=input_dim, pool_size=pool_size)
    trainer = DPSNTrainer(model)

    print(f"Start Training: Pool Size={pool_size}, Input Dim={input_dim}")

    # Generate Data
    X, Y = generate_dummy_data(batch_size=200, input_dim=input_dim)

    # Train
    epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0
        used_params_history = []

        # Train sample by sample (or small batches) to simulate the dynamic nature
        for i in range(len(X)):
            sample_x = X[i : i + 1]
            sample_y = Y[i : i + 1]

            stats = trainer.train_step(sample_x, sample_y)
            epoch_loss += stats["loss"]
            used_params_history.append(stats["budget"])

            if i % 50 == 0 and epoch == 0:
                print(f"  Step {i}: Loss={stats['loss']:.4f}, Budget={stats['budget']}")

        avg_loss = epoch_loss / len(X)
        avg_budget = sum(used_params_history) / len(used_params_history)
        print(
            f"Epoch {epoch + 1}: Avg Loss={avg_loss:.4f}, Avg Budget={int(avg_budget)}"
        )

    # Verification of Sparse Learning
    print("\n--- Verification: Gradient Sparsity ---")
    # Run one pass
    sample_x = X[0:1]
    sample_y = Y[0:1]
    trainer.router_optimizer.zero_grad()
    trainer.pool_optimizer.zero_grad()
    result = model(sample_x)
    loss = trainer.criterion(result["output"], sample_y)
    loss.backward()

    grad = model.pool.params.grad
    # Check non-zero gradients
    # Use tolerance because of float precision
    non_zero_grads = (grad.abs() > 1e-8).sum().item()
    selected_count = result["indices"].shape[1]

    print(f"Selected Indices Count: {selected_count}")
    print(f"Parameters with Non-Zero Gradients: {non_zero_grads}")

    # They might not match exactly if multiple indices point to same param (unlikely with topk)
    # or if some gradients mathematically became 0. But should be close.
    # Crucially, it should be << Total Pool Size (10,000)

    assert non_zero_grads < pool_size * 0.5, (
        "Error: Gradients are dense! Too many parameters updating."
    )
    print("Success: Updates are sparse!")


if __name__ == "__main__":
    train_loop()
