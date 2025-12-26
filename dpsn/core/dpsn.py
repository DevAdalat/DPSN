"""Main DPSN module combining all components."""

import torch
import torch.nn as nn
import time
from typing import Dict, Optional

from dpsn.core.parameter_pool import ParameterPool
from dpsn.core.router import RouteGeneratorModel
from dpsn.core.executor import SparseExecutionEngine
from config.hyperparameters import DPSNConfig


class DPSN(nn.Module):
    """Dynamic Parameter Selection Network.

    A neural network that dynamically selects which parameters to use based
    on input complexity, enabling adaptive computation and efficient scaling.

    The DPSN consists of three main components:
    1. Parameter Pool: Large storage of learnable parameters
    2. Router: Determines complexity and selects relevant parameters
    3. Executor: Performs computation using selected parameters

    Args:
        input_dim: Dimension of input embeddings
        pool_size: Total number of parameters in the pool
        config: Optional DPSNConfig object for detailed configuration
        **kwargs: Additional config parameters (overrides config object)

    Attributes:
        pool: The parameter pool
        router: The route generator
        executor: The execution engine
        config: Configuration object
    """

    def __init__(
        self,
        input_dim: int = 768,
        pool_size: int = 100_000,
        config: Optional[DPSNConfig] = None,
        **kwargs,
    ):
        super().__init__()

        # Extract use_residual before config creation to avoid passing it to DPSNConfig
        use_residual = kwargs.pop("use_residual", True)

        # Build configuration
        if config is None:
            config = DPSNConfig(input_dim=input_dim, pool_size=pool_size, **kwargs)
        else:
            # Override config with any explicit kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self.config = config

        # Initialize components
        self.pool = ParameterPool(
            pool_size=config.pool_size,
            param_dim=config.input_dim,
            sparse=config.sparse_gradients,
        )

        self.router = RouteGeneratorModel(
            input_dim=config.input_dim,
            pool_size=config.pool_size,
            hidden_dim=config.router_hidden_dim,
            min_params=config.min_params,
            max_params=config.max_params,
            complexity_exponent=config.complexity_exponent,
            noise_std=config.noise_std,
        )

        self.executor = SparseExecutionEngine(
            input_dim=config.input_dim,
            activation="tanh",
            use_residual=use_residual,
        )

    def forward(self, x: torch.Tensor, return_details: bool = False) -> Dict:
        """Forward pass through DPSN.

        Args:
            x: Input tensor of shape [batch_size, input_dim]
            return_details: Whether to return detailed execution information

        Returns:
            Dictionary containing:
                - output: Transformed input [batch_size, input_dim]
                - complexity_score: Estimated complexity [batch_size, 1]
                - parameters_used: Number of parameters selected (int)
                - indices: Selected parameter indices [batch_size, budget]
                - weights: Parameter weights [batch_size, budget]
                - execution_time_ms: Time taken for execution (float)
                - (optional) detailed execution info if return_details=True
        """
        # Route: determine which parameters to use
        indices, budget, weights, complexity = self.router(x)

        # Execute: perform computation with selected parameters
        start_time = time.time()

        if return_details:
            exec_result = self.executor.forward_with_details(
                x, indices, weights, self.pool
            )
            output = exec_result["output"]
        else:
            output = self.executor(x, indices, weights, self.pool)
            exec_result = None

        end_time = time.time()

        # Prepare result dictionary
        result = {
            "output": output,
            "complexity_score": complexity,
            "parameters_used": budget,
            "indices": indices,
            "weights": weights,
            "execution_time_ms": (end_time - start_time) * 1000,
        }

        if return_details and exec_result is not None:
            result["execution_details"] = exec_result

        return result

    def get_statistics(self, x: torch.Tensor) -> dict:
        """Get comprehensive statistics about DPSN operation.

        Args:
            x: Input tensor

        Returns:
            Dictionary with pool, routing, and execution statistics
        """
        with torch.no_grad():
            # Get pool statistics
            pool_stats = self.pool.get_param_statistics()

            # Get routing statistics
            routing_stats = self.router.get_routing_statistics(x)

            # Perform forward pass for execution stats
            result = self.forward(x, return_details=True)

            return {
                "pool": pool_stats,
                "routing": routing_stats,
                "execution": {
                    "mean_complexity": result["complexity_score"].mean().item(),
                    "budget": result["parameters_used"],
                    "budget_ratio": result["parameters_used"] / self.config.pool_size,
                    "execution_time_ms": result["execution_time_ms"],
                },
            }

    def count_parameters(self) -> dict:
        """Count total and active parameters in the model.

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        pool_params = self.pool.params.numel()
        router_params = sum(p.numel() for p in self.router.parameters())

        return {
            "total": total_params,
            "pool": pool_params,
            "router": router_params,
            "pool_percentage": (pool_params / total_params) * 100,
        }
