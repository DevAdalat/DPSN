import torch
import torch.nn as nn
import torch.optim as optim
from dpsn.models.language_model import DPSNLanguageModel
from config.hyperparameters import LanguageModelConfig
import os

batch_size = 32
block_size = 64
max_iters = 500
eval_interval = 100
learning_rate = 1e-3

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    print("Warning: GPU not available, falling back to CPU")
    device = "cpu"

eval_iters = 50
n_embd = 64
n_head = 4
dropout = 0.0

pool_size = 100_000
router_hidden_dim = 32
min_params = 100
max_params = 1000

torch.manual_seed(1337)

if not os.path.exists("data/input.txt"):
    print("Error: data/input.txt not found. Please ensure the dataset exists.")
    exit(1)

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
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


def train_shakespeare_gpu():
    print(f"Initializing DPSN Language Model on {device}...")
    print(f"Configuration: Pool Size={pool_size}, Router Hidden={router_hidden_dim}")

    config = LanguageModelConfig(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=4,
        block_size=block_size,
        pool_size=pool_size,
        dropout=dropout,
    )

    model = DPSNLanguageModel(
        config,
        router_hidden_dim=router_hidden_dim,
        min_params=min_params,
        max_params=max_params,
        complexity_exponent=2.0,
    )

    m = model.to(device)

    print(
        f"Parameter count (approx): {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M"
    )

    counts = m.count_parameters()
    print(f"Pool Params: {counts['dpsn_pools']:,}")
    print(f"Router Params: {counts['dpsn_routers']:,}")
    print(f"Pool/Router Ratio: {counts['dpsn_pools'] / counts['dpsn_routers']:.2f}")

    pool_params = []
    for block in m.blocks:
        pool_params.extend(list(block.dpsn.pool.parameters()))
    pool_param_ids = list(map(id, pool_params))

    other_params = filter(lambda p: id(p) not in pool_param_ids, m.parameters())

    optimizer_router = torch.optim.AdamW(other_params, lr=learning_rate)
    optimizer_pool = torch.optim.SGD(pool_params, lr=learning_rate * 10)

    print("Starting Training Loop...")

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch("train")

        logits, loss, stats = model(xb, yb)

        optimizer_router.zero_grad(set_to_none=True)
        optimizer_pool.zero_grad(set_to_none=True)

        loss.backward()

        optimizer_router.step()
        optimizer_pool.step()

    print("\n--- Generating Text ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))

    torch.save(model.state_dict(), "dpsn_shakespeare_gpu.pth")
    print("Model saved to dpsn_shakespeare_gpu.pth")


if __name__ == "__main__":
    train_shakespeare_gpu()
