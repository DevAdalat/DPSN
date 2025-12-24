import torch
import torch.nn as nn
import torch.optim as optim
from src.lm import DPSNLanguageModel
import os

# Hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 64  # what is the maximum context length for predictions?
max_iters = 500
eval_interval = 100
learning_rate = 1e-3
device = "cpu"  # force cpu as requested
eval_iters = 50
n_embd = 64
n_head = 4
pool_size = 20_000  # Smaller pool for quick demo training
dropout = 0.0

# ------------

torch.manual_seed(1337)

# Load data
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenizer (Char level)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train/Val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_shakespeare():
    print(f"Initializing DPSN Language Model...")
    print(f"Vocab Size: {vocab_size}, Embed Dim: {n_embd}, Pool Size: {pool_size}")

    model = DPSNLanguageModel(
        vocab_size, n_embd, n_head, block_size, pool_size, dropout
    )
    m = model.to(device)

    print(
        f"Parameter count (approx): {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M"
    )

    # Initial Generation (Before Training)
    print("\n--- Generating Text (Before Training) ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
    print("---------------------------------------\n")

    # Optimizer
    # We need separate optimizers for sparsity as discussed
    # 1. Router & Attention & Embeddings (Standard AdamW)
    # 2. Pool (SGD for strict sparsity)

    # Identify pool parameters
    pool_params = list(m.block.dpsn.pool.parameters())
    pool_param_ids = list(map(id, pool_params))

    other_params = filter(lambda p: id(p) not in pool_param_ids, m.parameters())

    optimizer_router = torch.optim.AdamW(other_params, lr=learning_rate)
    optimizer_pool = torch.optim.SGD(
        pool_params, lr=learning_rate * 10
    )  # Higher LR for pool

    print("Starting Training Loop...")

    # 50 iterations as requested
    max_iters = 50

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

            # Quick debug on stats
            xb, yb = get_batch("train")
            _, _, stats = model(xb, yb)
            # stats is from DPSNBlock -> dpsn return
            # It's a dict with 'parameters_used' (int) and 'complexity_score' (scalar)
            # But wait, in batch mode (via loop in DPSN.forward), it returns a list of stats if batch > 1?
            # Let's check dpsn.py implementation.
            # In dpsn.py, we updated it to return { ..., 'output': ..., 'indices': ... }
            # Wait, dpsn.py processes ONE sample at a time if loop is used, or batched if vectorized.
            # My dpsn.py implementation has:
            # def forward(self, x):
            #   ...
            #   return { ... 'parameters_used': budget ... }
            # AND it assumes x is [1, InputDim].
            #
            # BUT in lm.py I flatten: x.view(B*T, C).
            # If DPSN forward handles batch as single huge batch?
            # In dpsn.py:
            # def forward(self, x: torch.Tensor) -> Dict:
            #   indices, budget, weights, complexity = self.router(x)
            #
            # Router.forward:
            #   complexity_score = self.complexity_net(x).item() -> .item() implies SCALAR!
            #   It crashes if batch > 1.
            #
            # CRITICAL ISSUE: The current DPSN implementation in dpsn.py assumes BATCH SIZE 1.
            # I must fix dpsn.py to handle Batch inputs for real training.
            pass

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss, stats = model(xb, yb)

        optimizer_router.zero_grad(set_to_none=True)
        optimizer_pool.zero_grad(set_to_none=True)

        loss.backward()

        optimizer_router.step()
        optimizer_pool.step()

    # Final Generate
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- Generating Text ---")
    print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))

    # Save model
    torch.save(model.state_dict(), "dpsn_shakespeare.pth")
    print("Model saved to dpsn_shakespeare.pth")


if __name__ == "__main__":
    train_shakespeare()
