import torch
import torch.nn.functional as F
from src.lm import DPSNLanguageModel
import os

# Hyperparameters (must match training)
block_size = 64
n_embd = 64
n_head = 4
pool_size = 20_000
dropout = 0.0
device = "cpu"

# Load data for tokenizer
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])


def debug_generation():
    print("Loading model...")
    model = DPSNLanguageModel(
        vocab_size, n_embd, n_head, block_size, pool_size, dropout
    )
    try:
        model.load_state_dict(torch.load("dpsn_shakespeare.pth", map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model file not found. Please train first.")
        return

    model.eval()
    model.to(device)

    # Start generation
    start_str = "\n"
    idx = torch.tensor(encode(start_str), dtype=torch.long, device=device).unsqueeze(0)

    print("\n--- Detailed Generation Debug ---")
    print(f"Generating 20 tokens...\n")

    print(
        f"{'Token':<10} | {'Budget':<10} | {'Exec Time (ms)':<15} | {'Complexity':<10}"
    )
    print("-" * 55)

    generated_text = start_str

    with torch.no_grad():
        for i in range(20):
            # Crop context
            idx_cond = idx[:, -block_size:]

            # Forward pass (get stats)
            logits, _, stats = model(idx_cond)

            # Stats are from the last forward pass which processed the whole sequence context
            # DPSN processes the flattened sequence [B*T, C]
            # So stats['parameters_used'] is the budget used for this batch of tokens.
            # Since our router uses BATCH MEAN/MAX strategy (in determining budget),
            # the budget is constant for the whole forward pass.

            budget = stats["parameters_used"]
            exec_time = stats["execution_time_ms"]
            complexity = (
                stats["complexity_score"].mean().item()
            )  # Avg complexity of context

            # Sample next token
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            token_char = decode([idx_next.item()])
            generated_text += token_char

            # Display special chars visibly
            display_char = token_char.replace("\n", "\\n").replace(" ", "_")

            print(
                f"{display_char:<10} | {budget:<10} | {exec_time:.4f}          | {complexity:.4f}"
            )

    print("\n--- Full Generated Text ---")
    print(generated_text)
    print("---------------------------")


if __name__ == "__main__":
    debug_generation()
