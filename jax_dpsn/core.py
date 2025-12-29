"""Core DPSN components implemented in JAX/Flax NNX."""

from typing import Tuple, Optional, Any, Dict
import jax
import jax.numpy as jnp
from flax import nnx


class ParameterPool(nnx.Module):
    """
    A large pool of learnable parameters accessed via sparse indexing.

    In JAX/TPU, we typically shard this across devices.
    For this implementation, we use a standard Embedding layer.
    """

    def __init__(
        self,
        pool_size: int,
        param_dim: int,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.pool_size = pool_size
        self.param_dim = param_dim
        # We use nnx.Embed for the pool.
        # On TPU, this would be sharded using jax.sharding.PartitionSpec.
        self.embedding = nnx.Embed(
            num_embeddings=pool_size, features=param_dim, rngs=rngs, dtype=dtype
        )

    def get_params(self, indices: jax.Array) -> jax.Array:
        """
        Fetch parameters for the given indices.

        Args:
            indices: [batch_size, num_selected]

        Returns:
            [batch_size, num_selected, param_dim]
        """
        return self.embedding(indices)

    def __call__(self, indices: jax.Array) -> jax.Array:
        return self.get_params(indices)


class Router(nnx.Module):
    """
    Router that determines complexity and selects parameters.

    Implements Soft-Adaptive Computation:
    - Always selects 'max_params' to keep tensor shapes static (TPU requirement).
    - Uses a dynamic mask to zero out parameters beyond the calculated budget.
    """

    def __init__(
        self,
        input_dim: int,
        pool_size: int,
        hidden_dim: int = 256,
        min_params: int = 100,
        max_params: int = 1024,
        complexity_exponent: float = 2.0,
        noise_std: float = 0.1,
        rngs: nnx.Rngs = None,
        dtype=jnp.float32,
    ):
        self.pool_size = pool_size
        self.min_params = min_params
        self.max_params = max_params
        self.complexity_exponent = complexity_exponent
        self.noise_std = noise_std

        # Complexity network: input -> scalar [0, 1]
        self.complexity_net = nnx.Sequential(
            nnx.Linear(input_dim, 128, rngs=rngs, dtype=dtype),
            nnx.relu,
            nnx.Linear(128, 1, rngs=rngs, dtype=dtype),
            nnx.sigmoid,
        )

        # Scoring network: input -> scores [pool_size]
        self.scorer = nnx.Sequential(
            nnx.Linear(input_dim, hidden_dim, rngs=rngs, dtype=dtype),
            nnx.relu,
            nnx.Linear(hidden_dim, pool_size, rngs=rngs, dtype=dtype),
        )

    def determine_budget(self, complexity_scores: jax.Array) -> jax.Array:
        """
        Calculate budget based on complexity.
        Returns the integer budget for each item in batch.
        """
        # complexity_scores: [batch_size, ..., 1]
        scale = jnp.power(complexity_scores, self.complexity_exponent)

        # Raw budget calculation
        raw_budget = self.min_params + (self.max_params - self.min_params) * scale

        # Clip to valid range
        budget = jnp.clip(raw_budget, self.min_params, self.max_params)
        return jnp.round(budget).astype(jnp.int32)  # [batch_size, ..., 1]

    def __call__(
        self, x: jax.Array, rngs: Optional[nnx.Rngs] = None, deterministic: bool = False
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Args:
            x: Input [batch_size, ..., input_dim]

        Returns:
            indices: [batch_size, ..., max_params] (Top-k indices)
            weights: [batch_size, ..., max_params] (Softmax weights)
            mask: [batch_size, ..., max_params] (1.0 for active, 0.0 for masked)
            complexity: [batch_size, ..., 1]
        """
        # 1. Estimate complexity
        complexity = self.complexity_net(x)  # [B, ..., 1]
        budgets = self.determine_budget(complexity)  # [B, ..., 1]

        # 2. Score parameters
        scores = self.scorer(x)  # [B, ..., pool_size]

        # 3. Add noise during training (if rngs provided and not deterministic)
        if not deterministic and rngs is not None and self.noise_std > 0:
            # Use jax.random.normal with the key from rngs.params()
            noise = jax.random.normal(rngs.params(), scores.shape) * self.noise_std
            scores = scores + noise

        # 4. Select Top-K (Static K = max_params)
        # We always select max_params to keep shapes static for TPU
        top_scores, indices = jax.lax.top_k(scores, self.max_params)

        # 5. Compute weights (Softmax over selected)
        weights = nnx.softmax(top_scores, axis=-1)

        # 6. Generate Mask based on dynamic budget
        # range_tensor: [max_params]
        range_tensor = jnp.arange(self.max_params)

        # mask: 1.0 if index < budget, else 0.0
        # budgets: [B, ..., 1]
        # range_tensor broadcasts to [B, ..., max_params]
        mask = (range_tensor < budgets).astype(jnp.float32)

        # Apply mask to weights
        weights = weights * mask

        return indices, weights, mask, complexity


class Executor(nnx.Module):
    """
    Executes computation using selected parameters.
    """

    def __init__(
        self,
        input_dim: int,
        activation: str = "tanh",
        use_residual: bool = True,
    ):
        self.input_dim = input_dim
        self.use_residual = use_residual

        if activation == "tanh":
            self.activation_fn = jnp.tanh
        elif activation == "relu":
            self.activation_fn = nnx.relu
        elif activation == "gelu":
            self.activation_fn = nnx.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def __call__(
        self,
        x: jax.Array,
        indices: jax.Array,
        weights: jax.Array,
        pool: ParameterPool,
    ) -> jax.Array:
        """
        Args:
            x: [batch_size, ..., input_dim]
            indices: [batch_size, ..., max_params]
            weights: [batch_size, ..., max_params] (already masked)
            pool: ParameterPool instance
        """
        # 1. Fetch params: [B, ..., K, D] where K=max_params
        selected_params = pool(indices)

        # 2. Project: x @ params.T
        # Use einsum for robust handling of batch/sequence dimensions
        # x: [..., D], params: [..., K, D] -> products: [..., K]
        products = jnp.einsum("...d,...kd->...k", x, selected_params)

        # 3. Activation & Weighting
        activations = self.activation_fn(products)  # [..., K]
        weighted_activations = activations * weights  # [..., K]

        # 4. Reconstruct: weighted @ params
        # weighted: [..., K], params: [..., K, D] -> output: [..., D]
        output = jnp.einsum("...k,...kd->...d", weighted_activations, selected_params)

        # 5. Residual
        if self.use_residual:
            output = output + x

        return output


class DPSN(nnx.Module):
    """
    Main DPSN module combining Router, Pool, and Executor.
    """

    def __init__(
        self,
        input_dim: int,
        pool_size: int,
        min_params: int = 100,
        max_params: int = 1024,
        rngs: nnx.Rngs = None,
    ):
        self.pool = ParameterPool(pool_size=pool_size, param_dim=input_dim, rngs=rngs)

        self.router = Router(
            input_dim=input_dim,
            pool_size=pool_size,
            min_params=min_params,
            max_params=max_params,
            rngs=rngs,
        )

        self.executor = Executor(
            input_dim=input_dim, activation="tanh", use_residual=True
        )

    def __call__(
        self, x: jax.Array, rngs: Optional[nnx.Rngs] = None, deterministic: bool = False
    ) -> jax.Array:
        # Route
        indices, weights, mask, complexity = self.router(
            x, rngs=rngs, deterministic=deterministic
        )

        # Execute
        output = self.executor(x, indices, weights, self.pool)

        return output
