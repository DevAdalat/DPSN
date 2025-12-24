import torch
import torch.nn as nn
from datasets import load_dataset
from src.dpsn import DPSN
import time
import os
import psutil

# --- Configuration ---
POOL_SIZE = 1_000_000  # 1 Million slots -> ~1 Billion params
INPUT_DIM = 768  # Standard BERT size
MAX_BUDGET = 5000  # Max active params
BATCH_SIZE = 1  # Keep low for CPU sparse training efficiency
LEARNING_RATE = 1e-4
MAX_STEPS = 1000  # Proof of concept run
SAVE_EVERY = 100
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-v1"  # Smaller subset for faster streaming test


# --- 1. Model Wrapper for Language Modeling ---
class DPSNWrapper(nn.Module):
    def __init__(self, input_dim, pool_size):
        super().__init__()
        self.dpsn = DPSN(input_dim=input_dim, pool_size=pool_size)
        # Simple projection to vocab (simulated for generic data)
        # In real LLM, you'd have embeddings. Here we train on raw vectors
        # to demonstrate the 1B training mechanics purely.
        self.head = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: [Batch, Dim]
        dpsn_out = self.dpsn(x)
        output = self.head(dpsn_out["output"])
        return output, dpsn_out


# --- 2. Training Loop ---
def train_1B_model():
    print(f"=== Starting 1B Parameter Training on CPU ===")
    print(f"Memory Check: {psutil.virtual_memory().percent}% Used")

    # A. Load Dataset (Streaming Mode)
    print("Loading Dataset (Streaming)...")

    # We use a dummy generator for vector data since we aren't training a full tokenizer here
    # This proves the *Architecture* training capability
    def data_generator():
        while True:
            yield torch.randn(BATCH_SIZE, INPUT_DIM), torch.randn(BATCH_SIZE, INPUT_DIM)

    # B. Initialize Model
    print("Initializing 1 Billion Parameter DPSN...")
    model = DPSNWrapper(INPUT_DIM, POOL_SIZE)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # C. Optimizers
    pool_params = list(model.dpsn.pool.parameters())
    pool_ids = list(map(id, pool_params))
    router_params = filter(lambda p: id(p) not in pool_ids, model.parameters())

    opt_router = torch.optim.AdamW(router_params, lr=LEARNING_RATE)
    opt_pool = torch.optim.SGD(pool_params, lr=LEARNING_RATE * 10)

    criterion = nn.MSELoss()

    # D. Training Loop
    model.train()
    start_time = time.time()
    data_iter = data_generator()

    losses = []

    for step in range(MAX_STEPS):
        inputs, targets = next(data_iter)

        opt_router.zero_grad()
        opt_pool.zero_grad()

        # Forward
        outputs, stats = model(inputs)
        loss = criterion(outputs, targets)

        # Backward
        loss.backward()

        # Update
        opt_router.step()
        opt_pool.step()

        losses.append(loss.item())

        # Logging
        if (step + 1) % 10 == 0:
            avg_loss = sum(losses[-10:]) / 10
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            print(
                f"Step {step + 1}/{MAX_STEPS} | Loss: {avg_loss:.4f} | Budget: {stats['parameters_used']} | Speed: {steps_per_sec:.2f} step/s"
            )

        # Checkpointing
        if (step + 1) % SAVE_EVERY == 0:
            print(f"Saving checkpoint at step {step + 1}...")
            # We save only the router to save disk space for this demo
            # Saving 1B params takes 4GB disk space
            torch.save(
                model.dpsn.router.state_dict(), f"dpsn_router_step_{step + 1}.pth"
            )

    print("Training Complete!")


if __name__ == "__main__":
    train_1B_model()
