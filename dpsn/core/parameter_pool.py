"""Parameter Pool - The massive, passive storage of knowledge."""

import torch
import torch.nn as nn
from typing import Optional


class ParameterPool(nn.Module):
    """A large pool of learnable parameters accessed via sparse indexing.

    This component acts as the "library" of the DPSN architecture, storing
    potentially millions or billions of parameters that can be selectively
    activated based on input complexity.

    The pool uses sparse embeddings to ensure gradient updates only affect
    the parameters that were actually used, preventing catastrophic forgetting
    and enabling efficient training with massive parameter counts.

    Args:
        pool_size: Total number of parameter vectors in the pool
        param_dim: Dimension of each parameter vector
        sparse: Whether to use sparse gradients (recommended for large pools)
        init_std: Standard deviation for parameter initialization

    Attributes:
        pool_size: Number of parameters in the pool
        param_dim: Dimension of each parameter
        embedding: The underlying embedding layer storing parameters
    """

    def __init__(
        self,
        pool_size: int = 1_000_000,
        param_dim: int = 768,
        sparse: bool = True,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.param_dim = param_dim

        # Use Embedding with sparse=True to ensure gradients are sparse
        # This prevents allocating a dense gradient matrix for the entire pool
        self.embedding = nn.Embedding(pool_size, param_dim, sparse=sparse)

        # Initialize with normal distribution
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)

    @property
    def params(self) -> torch.Tensor:
        """Expose the weight tensor for optimizer access and inspection.

        Returns:
            Tensor of shape [pool_size, param_dim]
        """
        return self.embedding.weight

    def get_params(self, indices: torch.Tensor) -> torch.Tensor:
        """Fetch parameters for the given indices.

        Using the embedding layer forward pass ensures sparse gradients during
        backpropagation, meaning only the selected parameters receive gradient
        updates.

        Args:
            indices: Tensor of shape [batch_size, num_selected] containing
                    parameter indices to fetch

        Returns:
            Tensor of shape [batch_size, num_selected, param_dim] containing
            the selected parameter vectors
        """
        return self.embedding(indices)

    def get_param_statistics(self) -> dict:
        """Get statistics about parameter usage and values.

        Returns:
            Dictionary containing:
                - mean: Mean parameter value
                - std: Standard deviation of parameters
                - min: Minimum parameter value
                - max: Maximum parameter value
                - norm: L2 norm of all parameters
        """
        params = self.params.data
        return {
            "mean": params.mean().item(),
            "std": params.std().item(),
            "min": params.min().item(),
            "max": params.max().item(),
            "norm": params.norm().item(),
        }

    def reset_parameters(
        self, indices: Optional[torch.Tensor] = None, init_std: float = 0.02
    ):
        """Reset specific parameters or all parameters to random values.

        Useful for experimentation or selective parameter refresh.

        Args:
            indices: Optional tensor of indices to reset. If None, resets all.
            init_std: Standard deviation for reinitialization
        """
        if indices is None:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
        else:
            with torch.no_grad():
                self.embedding.weight[indices] = (
                    torch.randn(
                        len(indices),
                        self.param_dim,
                        device=self.embedding.weight.device,
                    )
                    * init_std
                )
