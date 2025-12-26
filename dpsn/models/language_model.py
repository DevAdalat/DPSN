"""DPSN-based Language Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from dpsn.models.transformer_block import DPSNBlock
from config.hyperparameters import LanguageModelConfig


class DPSNLanguageModel(nn.Module):
    """Language model using DPSN for adaptive computation.

    This model replaces the standard feedforward layers in a transformer
    with DPSN modules, enabling dynamic parameter selection based on
    input complexity.

    Args:
        config: LanguageModelConfig object with model hyperparameters
        vocab_size: Vocabulary size (overrides config if provided)
        **kwargs: Additional config overrides

    Attributes:
        config: Model configuration
        token_embedding_table: Token embeddings
        position_embedding_table: Positional embeddings
        blocks: List of DPSN transformer blocks
        ln_f: Final layer normalization
        lm_head: Language modeling head
    """

    def __init__(
        self, config: LanguageModelConfig, vocab_size: Optional[int] = None, **kwargs
    ):
        super().__init__()

        # Handle config overrides
        if vocab_size is not None:
            config.vocab_size = vocab_size
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config

        # Embeddings
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        # Transformer blocks with DPSN
        self.blocks = nn.ModuleList(
            [
                DPSNBlock(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    block_size=config.block_size,
                    pool_size=config.pool_size,
                    dropout=config.dropout,
                    router_hidden_dim=config.router_hidden_dim,
                    min_params=config.min_params,
                    max_params=config.max_params,
                    complexity_exponent=config.complexity_exponent,
                )
                for _ in range(config.n_layer)
            ]
        )

        # Final layer norm and output head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.block_size = config.block_size

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Dict]]:
        """
        Args:
            idx: Input token indices of shape [batch_size, seq_len]
            targets: Target token indices of shape [batch_size, seq_len]

        Returns:
            Tuple of:
                - logits: Predictions of shape [batch_size, seq_len, vocab_size]
                - loss: Cross-entropy loss (if targets provided, else None)
                - stats: List of DPSN statistics from each block
        """
        B, T = idx.shape

        # Embed tokens and positions
        tok_emb = self.token_embedding_table(idx)  # [B, T, n_embd]
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # [T, n_embd]
        x = tok_emb + pos_emb  # [B, T, n_embd]

        # Pass through DPSN blocks
        stats = []
        for block in self.blocks:
            x, block_stats = block(x)
            stats.append(block_stats)

        # Final layer norm and output projection
        x = self.ln_f(x)  # [B, T, n_embd]
        logits = self.lm_head(x)  # [B, T, vocab_size]

        # Compute loss if targets provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss, stats

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate new tokens autoregressively.

        Args:
            idx: Starting context of shape [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens

        Returns:
            Generated sequence of shape [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.block_size :]

            # Get predictions
            logits, _, _ = self(idx_cond)

            # Focus on last time step
            logits = logits[:, -1, :] / temperature  # [B, vocab_size]

            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)  # [B, seq_len + 1]

        return idx

    def get_model_statistics(self, idx: torch.Tensor) -> Dict:
        """Get comprehensive statistics about model operation.

        Args:
            idx: Input token indices

        Returns:
            Dictionary with statistics from all blocks
        """
        with torch.no_grad():
            _, _, stats = self(idx)

            # Aggregate statistics
            total_params_used = sum(s["parameters_used"] for s in stats)
            avg_complexity = sum(
                s["complexity_score"].mean().item() for s in stats
            ) / len(stats)

            return {
                "num_blocks": len(stats),
                "total_params_used": total_params_used,
                "avg_params_per_block": total_params_used / len(stats),
                "avg_complexity": avg_complexity,
                "block_stats": stats,
            }

    def count_parameters(self) -> Dict:
        """Count parameters in different parts of the model.

        Returns:
            Dictionary with parameter counts
        """
        total = sum(p.numel() for p in self.parameters())

        # Count DPSN parameters
        dpsn_pools = sum(block.dpsn.pool.params.numel() for block in self.blocks)
        dpsn_routers = sum(
            sum(p.numel() for p in block.dpsn.router.parameters())
            for block in self.blocks
        )

        # Count other components
        embeddings = (
            self.token_embedding_table.weight.numel()
            + self.position_embedding_table.weight.numel()
        )
        attention = sum(
            sum(p.numel() for p in block.sa.parameters()) for block in self.blocks
        )

        return {
            "total": total,
            "embeddings": embeddings,
            "dpsn_pools": dpsn_pools,
            "dpsn_routers": dpsn_routers,
            "attention": attention,
            "pool_percentage": (dpsn_pools / total) * 100,
        }
