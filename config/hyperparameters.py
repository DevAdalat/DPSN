"""Hyperparameter configurations for DPSN models."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DPSNConfig:
    """Configuration for Dynamic Parameter Selection Network.

    Attributes:
        input_dim: Dimension of input embeddings
        pool_size: Total number of parameters in the pool
        min_params: Minimum number of parameters to select
        max_params: Maximum number of parameters to select
        router_hidden_dim: Hidden dimension of router network
        complexity_exponent: Exponent for complexity-to-budget scaling (default: 2)
        noise_std: Standard deviation of exploration noise during training
        sparse_gradients: Whether to use sparse gradients for parameter pool
    """

    input_dim: int = 768
    pool_size: int = 100_000
    min_params: int = 100
    max_params: int = 5_000
    router_hidden_dim: int = 256
    complexity_exponent: float = 2.0
    noise_std: float = 0.1
    sparse_gradients: bool = True

    def __post_init__(self):
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.pool_size > 0, "pool_size must be positive"
        assert self.min_params > 0, "min_params must be positive"
        assert self.router_hidden_dim > 0, "router_hidden_dim must be positive"
        assert self.complexity_exponent > 0, "complexity_exponent must be positive"
        assert self.noise_std >= 0, "noise_std must be non-negative"

        if self.max_params > self.pool_size:
            self.max_params = self.pool_size
        if self.min_params > self.max_params:
            self.min_params = self.max_params // 10


@dataclass
class LanguageModelConfig:
    """Configuration for DPSN Language Model.

    Attributes:
        vocab_size: Size of vocabulary
        n_embd: Embedding dimension
        n_head: Number of attention heads
        n_layer: Number of transformer layers
        block_size: Maximum context length
        pool_size: Size of DPSN parameter pool
        dropout: Dropout probability
        use_dpsn: Whether to use DPSN (if False, uses standard FFN)
    """

    vocab_size: int
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    block_size: int = 1024
    pool_size: int = 100_000
    dropout: float = 0.1
    use_dpsn: bool = True
    # DPSN specific params
    router_hidden_dim: int = 256
    min_params: int = 100
    max_params: int = 5000
    complexity_exponent: float = 2.0

    def __post_init__(self):
        """Validate configuration."""
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.n_embd > 0, "n_embd must be positive"
        assert self.n_head > 0, "n_head must be positive"
        assert self.n_layer > 0, "n_layer must be positive"
        assert self.block_size > 0, "block_size must be positive"
        assert self.pool_size > 0, "pool_size must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"


@dataclass
class TrainingConfig:
    """Configuration for training process.

    Attributes:
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        max_iters: Maximum number of training iterations
        eval_interval: How often to evaluate on validation set
        eval_iters: Number of iterations for evaluation
        warmup_iters: Number of warmup iterations
        weight_decay: Weight decay for optimizer
        grad_clip: Gradient clipping threshold (None = no clipping)
    """

    batch_size: int = 64
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    warmup_iters: int = 100
    weight_decay: float = 0.1
    grad_clip: Optional[float] = 1.0

    def __post_init__(self):
        """Validate configuration."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.max_iters > 0, "max_iters must be positive"
        assert self.eval_interval > 0, "eval_interval must be positive"
        assert self.eval_iters > 0, "eval_iters must be positive"
        assert self.warmup_iters >= 0, "warmup_iters must be non-negative"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        if self.grad_clip is not None:
            assert self.grad_clip > 0, "grad_clip must be positive"
