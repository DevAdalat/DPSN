"""Self-attention mechanisms for transformer architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """Single head of self-attention.

    Args:
        head_size: Dimension of the attention head
        n_embd: Embedding dimension
        block_size: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self, head_size: int, n_embd: int, block_size: int, dropout: float = 0.0
    ):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]

        Returns:
            Attention output of shape [batch_size, seq_len, head_size]
        """
        B, T, C = x.shape
        k = self.key(x)  # [B, T, head_size]
        q = self.query(x)  # [B, T, head_size]

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # [B, T, T]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Weighted aggregation of values
        v = self.value(x)  # [B, T, head_size]
        out = wei @ v  # [B, T, head_size]
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel.

    Args:
        num_heads: Number of attention heads
        head_size: Dimension of each head
        n_embd: Embedding dimension
        block_size: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        n_embd: int,
        block_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]

        Returns:
            Multi-head attention output of shape [batch_size, seq_len, n_embd]
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
