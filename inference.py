import torch
import torch.nn.functional as F
import argparse
import time
import os
import sys

# Add project root to path to ensure imports work
sys.path.append(os.getcwd())

from dpsn.models.language_model import DPSNLanguageModel
from config.hyperparameters import LanguageModelConfig


def get_args():
    parser = argparse.ArgumentParser(
        description="Inference for DPSN Model with detailed statistics"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="dpsn_shakespeare_cpu.pth",
        help="Path to saved model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/input.txt",
        help="Path to training data for vocab building",
    )
    parser.add_argument("--prompt", type=str, default="\n", help="Initial prompt text")
    parser.add_argument(
        "--max_tokens", type=int, default=100, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--no_details", action="store_true", help="Disable detailed statistics table"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run on",
    )

    # Model Architecture Arguments (Must match training)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--pool_size", type=int, default=100_000)
    parser.add_argument("--router_hidden_dim", type=int, default=32)
    parser.add_argument("--min_params", type=int, default=100)
    parser.add_argument("--max_params", type=int, default=1000)

    return parser.parse_args()


def load_vocab(data_path):
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return stoi, itos, vocab_size


def format_table(rows, headers):
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    # Create format string
    fmt = " | ".join([f"{{:<{w}}}" for w in widths])
    separator = "-+-".join(["-" * w for w in widths])

    lines = []
    lines.append(fmt.format(*headers))
    lines.append(separator)
    for row in rows:
        lines.append(fmt.format(*[str(v) for v in row]))

    return "\n".join(lines)


def main():
    args = get_args()

    # Device setup
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load Vocab
    stoi, itos, vocab_size = load_vocab(args.data_path)
    encode = lambda s: [stoi.get(c, 0) for c in s]  # Handle unknown chars gracefully
    decode = lambda l: "".join([itos.get(i, "") for i in l])

    # Initialize Model
    print(f"Loading model from {args.model_path}...")

    config = LanguageModelConfig(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
        pool_size=args.pool_size,
        dropout=0.0,
    )

    model = DPSNLanguageModel(
        config,
        router_hidden_dim=args.router_hidden_dim,
        min_params=args.min_params,
        max_params=args.max_params,
        complexity_exponent=2.0,
    )

    # Load state dict
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: Model file {args.model_path} not found.")
        sys.exit(1)

    model.to(device)
    model.eval()

    # Prepare Prompt
    start_ids = encode(args.prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    print("\n--- Generating ---")
    print(args.prompt, end="", flush=True)

    stats_history = []

    # Generation Loop
    with torch.no_grad():
        for _ in range(args.max_tokens):
            # Crop context
            x_cond = x[:, -args.block_size :]

            # Forward pass with timing
            t0 = time.perf_counter()
            logits, _, stats = model(x_cond)
            t1 = time.perf_counter()
            step_time_ms = (t1 - t0) * 1000

            # Select next token
            logits = logits[:, -1, :]  # (B, V)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append
            x = torch.cat((x, idx_next), dim=1)

            # Decode and print
            token_id = idx_next.item()
            token_str = decode([token_id])
            print(token_str, end="", flush=True)

            # Collect Stats
            # stats is a list of dicts (one per layer)
            # We aggregate for the whole model for this step

            total_params = sum(layer["parameters_used"] for layer in stats)
            avg_complexity = sum(
                layer["complexity_score"].mean().item() for layer in stats
            ) / len(stats)

            # Assuming 'execution_time_ms' in stats is the internal DPSN computation time
            internal_time = sum(layer["execution_time_ms"] for layer in stats)

            stats_history.append(
                {
                    "token_id": token_id,
                    "token_char": repr(
                        token_str
                    ),  # Use repr to show newlines/spaces clearly
                    "params_used": total_params,
                    "complexity": f"{avg_complexity:.4f}",
                    "internal_time_ms": f"{internal_time:.2f}",
                    "total_time_ms": f"{step_time_ms:.2f}",
                }
            )

    print("\n\n--- Generation Complete ---")

    if not args.no_details:
        print("\n--- Detailed Generation Statistics ---")
        headers = [
            "Token",
            "Params Used",
            "Complexity",
            "DPSN Time (ms)",
            "Total Time (ms)",
        ]
        rows = []
        for s in stats_history:
            rows.append(
                [
                    s["token_char"],
                    s["params_used"],
                    s["complexity"],
                    s["internal_time_ms"],
                    s["total_time_ms"],
                ]
            )

        print(format_table(rows, headers))

        # Summary
        avg_params = sum(s["params_used"] for s in stats_history) / len(stats_history)
        avg_time = sum(float(s["total_time_ms"]) for s in stats_history) / len(
            stats_history
        )
        print(f"\nAverage Params per Token: {avg_params:.0f}")
        print(f"Average Time per Token: {avg_time:.2f} ms")


if __name__ == "__main__":
    main()
