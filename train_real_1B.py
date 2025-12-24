import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from src.lm import DPSNLanguageModel
import time
import os
import psutil

# --- Configuration for 1B Parameter Model ---
# To get ~1 Billion Parameters:
# We need Pool Size * Embedding Dim ≈ 1,000,000,000
# Standard GPT-2 Small has embedding dim 768.
# 1,300,000 * 768 = 998,400,000 (Approx 1 Billion)

POOL_SIZE = 1_300_000
N_EMBD = 768
N_HEAD = 12  # Standard for this embedding size
BLOCK_SIZE = 128  # Context length (Keep manageable for CPU)
DROPOUT = 0.1
BATCH_SIZE = 4  # Small batch for CPU
GRAD_ACCUM_STEPS = 8  # Accumulate gradients to simulate larger batch
LEARNING_RATE = 3e-4
MAX_STEPS = 10000  # Run for a long time
SAVE_EVERY = 500

DEVICE = "cpu"


def get_ram_usage():
    return psutil.virtual_memory().percent


def train_real_1B():
    print(f"=== Initializing Real 1B Parameter DPSN Training ===")
    print(f"Target: ~1 Billion Parameters on CPU")
    print(f"Dataset: WikiText-103 (Streaming)")

    # 1. Setup Tokenizer
    print("Loading GPT-2 Tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"Vocab Size: {vocab_size}")

    # 2. Initialize Model
    print("Allocating Model in RAM (This may take a moment)...")
    # This will allocate ~4GB for weights alone
    model = DPSNLanguageModel(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        block_size=BLOCK_SIZE,
        pool_size=POOL_SIZE,
        dropout=DROPOUT,
    )

    # Calculate exact parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Created Successfully.")
    print(f"Total Parameter Count: {total_params:,} ({total_params / 1e9:.2f} B)")
    print(f"RAM Usage: {get_ram_usage()}%")

    model.to(DEVICE)
    model.train()

    # 3. Optimizers
    print("Setting up Sparse Optimizers...")
    # Pool gets SGD (Sparse updates), Router gets AdamW
    pool_params = list(model.block.dpsn.pool.parameters())
    pool_ids = list(map(id, pool_params))
    router_params = filter(lambda p: id(p) not in pool_ids, model.parameters())

    opt_router = torch.optim.AdamW(router_params, lr=LEARNING_RATE)
    opt_pool = torch.optim.SGD(pool_params, lr=LEARNING_RATE * 10)

    # 4. Dataset Streaming
    print("Connecting to Hugging Face Stream...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)

    # Data Processing Pipeline
    def data_generator():
        # Buffer to create batches
        batch_texts = []
        for item in dataset:
            text = item["text"]
            if len(text.strip()) < 10:
                continue  # Skip empty/short lines

            batch_texts.append(text)

            if len(batch_texts) == BATCH_SIZE:
                # Tokenize
                encodings = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=BLOCK_SIZE,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encodings["input_ids"].to(DEVICE)

                # Create targets (shifted by 1)
                # We can just use input_ids as targets, the model forward handles shifting logic usually
                # But our DPSNLanguageModel.forward expects (idx, targets)
                # Let's verify src/lm.py:
                #   if targets is not None:
                #      loss = F.cross_entropy(logits, targets)
                # It expects targets to be aligned.
                # Standard LM training: input=x[0:-1], target=x[1:]

                # However, our tokenizer pads. We need to be careful with padding in loss.
                # For simplicity in this script, we'll pass full sequence and let model handle?
                # No, standard is:
                x = input_ids[:, :-1]
                y = input_ids[:, 1:]

                yield x, y
                batch_texts = []

    # 5. Training Loop
    print("\n=== STARTING TRAINING ===")
    start_time = time.time()
    data_iter = data_generator()

    step = 0
    total_loss = 0

    # Create directory for checkpoints
    os.makedirs("checkpoints_1B", exist_ok=True)

    opt_router.zero_grad()
    opt_pool.zero_grad()

    try:
        while step < MAX_STEPS:
            # Gradient Accumulation Loop
            accum_loss = 0
            for _ in range(GRAD_ACCUM_STEPS):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    # Reset iterator if dataset ends (unlikely for wikitext-103 in 10k steps)
                    data_iter = data_generator()
                    x, y = next(data_iter)

                logits, loss, stats = model(x, y)

                # Scale loss for accumulation
                loss = loss / GRAD_ACCUM_STEPS
                loss.backward()
                accum_loss += loss.item()

            # Update Weights
            opt_router.step()
            opt_pool.step()

            opt_router.zero_grad()
            opt_pool.zero_grad()

            step += 1
            total_loss += accum_loss

            # Logging
            if step % 1 == 0:  # Print every step for feedback since CPU is slow
                elapsed = time.time() - start_time
                avg_time = elapsed / step
                print(
                    f"Step {step}: Loss {accum_loss:.4f} | Active Params: {stats['parameters_used']} | Time/Step: {avg_time:.2f}s | RAM: {get_ram_usage()}%"
                )

            # Saving
            if step % SAVE_EVERY == 0:
                print(f"Saving Checkpoint {step}...")
                # We save everything. Warning: Large file (~4GB)
                checkpoint_path = f"checkpoints_1B/dpsn_1B_step_{step}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved to {checkpoint_path}")

    except KeyboardInterrupt:
        print("\nTraining paused by user.")
        save = input("Save current state? (y/n): ")
        if save.lower() == "y":
            torch.save(model.state_dict(), "checkpoints_1B/dpsn_1B_interrupted.pth")
            print("Saved.")

    print("Training finished.")


if __name__ == "__main__":
    train_real_1B()
