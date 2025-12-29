"""Training script for JAX DPSN."""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import time
from jax_dpsn.model import DPSNLanguageModel
from jax_dpsn.data import load_and_chunk_dataset, prepare_batch


def train():
    # Hyperparameters
    BATCH_SIZE = 4
    BLOCK_SIZE = 128
    N_EMBD = 128
    N_HEAD = 4
    N_LAYER = 2
    POOL_SIZE = 10000
    MIN_PARAMS = 64
    MAX_PARAMS = 256
    LEARNING_RATE = 3e-4
    MAX_STEPS = 50

    print(f"Device: {jax.devices()[0]}")

    # 1. Data
    print("Loading data...")
    dataloader, vocab_size = load_and_chunk_dataset(
        "data/input.txt", seq_len=BLOCK_SIZE, batch_size=BATCH_SIZE
    )

    # 2. Model
    print("Initializing model...")
    model = DPSNLanguageModel(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        block_size=BLOCK_SIZE,
        pool_size=POOL_SIZE,
        min_params=MIN_PARAMS,
        max_params=MAX_PARAMS,
        rngs=nnx.Rngs(0),
    )

    # 3. Optimizer
    optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE))

    # 4. Train Step
    @nnx.jit
    def train_step(model, optimizer, x, y):
        def loss_fn(model):
            logits = model(x, deterministic=False)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    # 5. Loop
    print("Starting training...")
    iter_data = iter(dataloader)

    for step in range(MAX_STEPS):
        t0 = time.time()
        try:
            batch = next(iter_data)
        except StopIteration:
            iter_data = iter(dataloader)
            batch = next(iter_data)

        x, y = prepare_batch(batch)

        loss = train_step(model, optimizer, x, y)

        t1 = time.time()
        dt = (t1 - t0) * 1000

        if step % 10 == 0:
            print(f"Step {step}: loss {loss:.4f}, time {dt:.2f}ms")


if __name__ == "__main__":
    train()
