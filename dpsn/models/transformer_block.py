"""Transformer block with DPSN integration."""

import torch
import torch.nn as nn
from typing import Tuple, Dict

from dpsn.models.attention import MultiHeadAttention
from dpsn.core.dpsn import DPSN


class DPSNBlock(nn.Module):
    """Transformer block with DPSN replacing the feedforward layer.

    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> Add ->
        x -> LayerNorm -> DPSN -> Add -> x

    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        block_size: Maximum sequence length
        pool_size: Size of DPSN parameter pool
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        pool_size: int,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        head_size = n_embd // n_head

        # Self-attention for communication
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)

        # DPSN for computation (replaces standard feedforward)
        # Disable internal residual to handle it properly in the block (Pre-LN)
        self.dpsn = DPSN(
            input_dim=n_embd, pool_size=pool_size, use_residual=False, **kwargs
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]

        Returns:
            Tuple of:
                - Output tensor of shape [batch_size, seq_len, n_embd]
                - Dictionary with DPSN statistics
        """
        # Self-attention with residual
        x = x + self.sa(self.ln1(x))

        # DPSN with residual
        # DPSN expects [batch_size, n_embd], so we flatten temporal dimension
        B, T, C = x.shape
        x_flat = x.view(B * T, C)
        x_norm = self.ln2(x_flat)

        # Pass through DPSN
        dpsn_out = self.dpsn(x_norm)
        output_flat = dpsn_out["output"]

        # Reshape back to [batch_size, seq_len, n_embd]
        output = output_flat.view(B, T, C)

        # Residual connection (Pre-LN: x = x + DPSN(LN(x)))
        x = x + output

        return x, dpsn_out
