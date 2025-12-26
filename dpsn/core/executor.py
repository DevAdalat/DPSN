"""Sparse Execution Engine - The computation worker."""

import torch
import torch.nn as nn
from typing import Optional


class SparseExecutionEngine(nn.Module):
    """Executes computation using dynamically selected parameters.

    The executor is the "worker" of the DPSN architecture. Given an input
    and a set of selected parameters, it performs the actual computation
    through a series of matrix operations:

    1. Compute activation scores: x @ params^T
    2. Apply non-linearity and weights
    3. Reconstruct output: weighted_activations @ params
    4. Add residual connection

    This creates a dynamic computation graph that changes for every input
    based on which parameters were selected by the router.

    Args:
        input_dim: Dimension of input vectors
        activation: Activation function to use ('tanh', 'relu', 'gelu')
        use_residual: Whether to add residual connection

    Attributes:
        input_dim: Input dimension
        activation_fn: The activation function
        use_residual: Whether residual connections are used
    """

    def __init__(
        self,
        input_dim: int,
        activation: str = "tanh",
        use_residual: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_residual = use_residual

        # Select activation function
        if activation == "tanh":
            self.activation_fn = torch.tanh
        elif activation == "relu":
            self.activation_fn = torch.relu
        elif activation == "gelu":
            self.activation_fn = torch.nn.functional.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        pool,  # ParameterPool instance
    ) -> torch.Tensor:
        """Execute sparse computation with selected parameters.

        Mathematical operations:
        1. Fetch: params = pool[indices]              # [B, K, D]
        2. Project: activations = tanh(x @ params^T)  # [B, 1, K]
        3. Weight: weighted = activations * weights    # [B, 1, K]
        4. Reconstruct: output = weighted @ params     # [B, 1, D]
        5. Residual: output = x + output               # [B, D]

        Where:
        - B = batch_size
        - K = budget (number of selected parameters)
        - D = input_dim (parameter dimension)

        Args:
            x: Input tensor of shape [batch_size, input_dim]
            indices: Selected parameter indices [batch_size, budget]
            weights: Softmax weights for parameters [batch_size, budget]
            pool: ParameterPool instance to fetch parameters from

        Returns:
            Output tensor of shape [batch_size, input_dim]
        """
        # Step 1: Fetch selected parameters from pool
        # Using get_params() ensures sparse gradient generation
        selected_params = pool.get_params(indices)  # [B, K, D]

        # Step 2: Compute activation scores
        # x: [B, D] -> [B, 1, D]
        # selected_params: [B, K, D] -> [B, D, K]
        # products: [B, 1, K]
        x_expanded = x.unsqueeze(1)  # [B, 1, D]
        params_transposed = selected_params.transpose(1, 2)  # [B, D, K]
        products = torch.matmul(x_expanded, params_transposed)  # [B, 1, K]

        # Step 3: Apply activation and weighting
        activations = self.activation_fn(products)  # [B, 1, K]
        weighted_activations = activations * weights.unsqueeze(1)  # [B, 1, K]

        # Step 4: Reconstruct output
        # weighted_activations: [B, 1, K]
        # selected_params: [B, K, D]
        # output: [B, 1, D]
        output = torch.matmul(weighted_activations, selected_params)  # [B, 1, D]
        output = output.squeeze(1)  # [B, D]

        # Step 5: Add residual connection
        if self.use_residual:
            output = x + output

        return output

    def forward_with_details(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        pool,
    ) -> dict:
        """Execute computation and return intermediate activations.

        Useful for debugging and analysis.

        Args:
            Same as forward()

        Returns:
            Dictionary containing:
                - output: Final output tensor
                - selected_params: The selected parameter vectors
                - activations: Pre-weighted activation scores
                - weighted_activations: Post-weighted activations
                - products: Raw projection products
        """
        # Fetch parameters
        selected_params = pool.get_params(indices)

        # Compute
        x_expanded = x.unsqueeze(1)
        params_transposed = selected_params.transpose(1, 2)
        products = torch.matmul(x_expanded, params_transposed)
        activations = self.activation_fn(products)
        weighted_activations = activations * weights.unsqueeze(1)
        output = torch.matmul(weighted_activations, selected_params).squeeze(1)

        if self.use_residual:
            output = x + output

        return {
            "output": output,
            "selected_params": selected_params,
            "activations": activations,
            "weighted_activations": weighted_activations,
            "products": products,
        }
