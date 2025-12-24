import torch
import torch.nn as nn
from torch.nn import functional as F
from src.dpsn import DPSN


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class DPSNBlock(nn.Module):
    """Transformer block: communication followed by computation (DPSN)"""

    def __init__(self, n_embd, n_head, block_size, pool_size, dropout=0.0):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)

        # Here we replace the standard FeedForward with DPSN
        self.dpsn = DPSN(input_dim=n_embd, pool_size=pool_size)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Self-Attention (Communication)
        x = x + self.sa(self.ln1(x))

        # DPSN (Computation)
        # DPSN expects [Batch, Dim], but here we have [Batch, Time, Dim].
        # We need to process each token.
        # Our DPSN implementation handles 2D inputs [Batch*Time, Dim] well if we flatten.
        B, T, C = x.shape
        x_flat = x.view(B * T, C)
        x_norm = self.ln2(x_flat)  # Norm before DPSN

        # Pass through DPSN
        # It returns Dict with 'output'
        dpsn_out = self.dpsn(x_norm)
        output_flat = dpsn_out["output"]

        # Reshape back
        output = output_flat.view(B, T, C)

        # Residual
        x = x + output

        return x, dpsn_out


class DPSNLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd=64,
        n_head=4,
        block_size=128,
        pool_size=100_000,
        dropout=0.0,
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Single block for this demo (or could be sequential)
        self.block = DPSNBlock(n_embd, n_head, block_size, pool_size, dropout)

        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

        # Pass through Block
        x, stats = self.block(x)

        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, stats

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size :]
            # get the predictions
            logits, loss, stats = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
