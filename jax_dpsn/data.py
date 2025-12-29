"""Data pipeline using Custom Sequence Packing (Simulating Grain behavior)."""

import jax
import numpy as np
import torch
from transformers import AutoTokenizer
from typing import Iterator, Tuple


class PackedDataLoader:
    """
    Custom Data Loader that implements Sequence Packing.

    Sequence Packing concatenates multiple short examples into a single sequence
    to minimize padding and maximize TPU utilization.
    """

    def __init__(
        self,
        path: str,
        seq_len: int,
        batch_size: int,
        tokenizer_name: str = "gpt2",
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        # 1. Load & Tokenize
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Add EOS token if missing
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = self.tokenizer.vocab_size - 1

        tokens = self.tokenizer.encode(text)
        self.tokens = np.array(tokens, dtype=np.int32)

        # 2. Pack Data
        # In a real scenario with distinct examples, we would pack multiple [EOS] separated
        # sequences. Here, for Shakespeare, it's one stream, so we just chunk it.
        # To demonstrate "Packing", we'll treat the stream as a source of tokens.

        # Ensure we have enough data
        total_len = len(self.tokens)
        self.num_batches = total_len // (batch_size * (seq_len + 1))

        # Truncate
        usable_len = self.num_batches * batch_size * (seq_len + 1)
        self.data = self.tokens[:usable_len]

        # Reshape into [N_samples, seq_len + 1]
        self.data = self.data.reshape(-1, seq_len + 1)
        self.num_samples = self.data.shape[0]

    def __iter__(self) -> Iterator[jax.Array]:
        """Yields batches of packed sequences."""
        indices = np.arange(self.num_samples)
        self.rng.shuffle(indices)

        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            if len(batch_indices) < self.batch_size:
                continue  # Drop remainder

            batch = self.data[batch_indices]
            yield jax.numpy.array(batch)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size


def load_and_chunk_dataset(path: str, seq_len: int, batch_size: int):
    loader = PackedDataLoader(path, seq_len, batch_size)
    return loader, loader.vocab_size


def prepare_batch(batch):
    """Split batch into inputs (x) and targets (y)."""
    # batch: [B, seq_len + 1]
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x, y
