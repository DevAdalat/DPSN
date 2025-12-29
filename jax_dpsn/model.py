"""DPSN Language Model implemented in JAX/Flax NNX."""

from typing import Optional, Tuple, Dict
import jax
import jax.numpy as jnp
from flax import nnx
from .core import DPSN


class DPSNBlock(nnx.Module):
    """
    Transformer block using DPSN for the feed-forward layer.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        pool_size: int,
        min_params: int = 100,
        max_params: int = 1024,
        dropout: float = 0.1,
        rngs: nnx.Rngs = None,
    ):
        self.ln1 = nnx.LayerNorm(n_embd, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=n_head,
            in_features=n_embd,
            qkv_features=n_embd,
            out_features=n_embd,
            decode=False,  # Optimized for training
            rngs=rngs,
        )
        self.ln2 = nnx.LayerNorm(n_embd, rngs=rngs)
        self.dpsn = DPSN(
            input_dim=n_embd,
            pool_size=pool_size,
            min_params=min_params,
            max_params=max_params,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jax.Array:
        # Attention block
        residual = x
        x = self.ln1(x)
        x = self.attn(x, mask=mask)
        x = self.dropout(x, deterministic=deterministic, rngs=rngs)
        x = x + residual

        # DPSN block (replaces FeedForward)
        # Note: DPSN has its own residual connection internally
        x = self.ln2(x)
        x = self.dpsn(x, rngs=rngs, deterministic=deterministic)

        return x


class DPSNLanguageModel(nnx.Module):
    """
    Full Language Model with DPSN blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        pool_size: int,
        min_params: int = 100,
        max_params: int = 1024,
        dropout: float = 0.1,
        rngs: nnx.Rngs = None,
    ):
        self.block_size = block_size
        self.token_embedding = nnx.Embed(vocab_size, n_embd, rngs=rngs)
        self.position_embedding = nnx.Embed(block_size, n_embd, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

        self.blocks = [
            DPSNBlock(
                n_embd=n_embd,
                n_head=n_head,
                pool_size=pool_size,
                min_params=min_params,
                max_params=max_params,
                dropout=dropout,
                rngs=rngs,
            )
            for _ in range(n_layer)
        ]

        self.ln_f = nnx.LayerNorm(n_embd, rngs=rngs)
        self.lm_head = nnx.Linear(n_embd, vocab_size, rngs=rngs)

    def __call__(
        self,
        idx: jax.Array,
        deterministic: bool = False,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jax.Array:
        B, T = idx.shape

        # Embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(jnp.arange(T))
        x = tok_emb + pos_emb
        x = self.dropout(x, deterministic=deterministic, rngs=rngs)

        # Causal Mask (Upper triangular with -inf)
        # nnx.MultiHeadAttention expects mask of shape [B, num_heads, Q, K] or broadcastable
        # We make [1, 1, T, T]
        mask = nnx.make_causal_mask(jnp.ones((B, T), dtype=jnp.bool), dtype=jnp.bool)

        # Blocks
        for block in self.blocks:
            x = block(x, mask=mask, deterministic=deterministic, rngs=rngs)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits
