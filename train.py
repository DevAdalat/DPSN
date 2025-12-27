import argparse
import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from dpsn.models.language_model import DPSNLanguageModel
from config.hyperparameters import LanguageModelConfig
from dpsn.utils.data_streaming import get_streaming_dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DPSN Model on HuggingFace Dataset"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g. 'wikitext')",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Dataset config name (e.g. 'wikitext-103-v1')",
    )
    parser.add_argument(
        "--dataset_split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column name containing text data",
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default="gpt2", help="Pretrained tokenizer to use"
    )

    parser.add_argument(
        "--n_layer", type=int, default=12, help="Number of transformer layers"
    )
    parser.add_argument(
        "--n_head", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument(
        "--block_size", type=int, default=1024, help="Context window size"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )

    parser.add_argument(
        "--pool_size",
        type=int,
        default=100_000,
        help="Total number of parameters in the pool",
    )
    parser.add_argument(
        "--router_hidden_dim", type=int, default=256, help="Router hidden dimension"
    )
    parser.add_argument(
        "--min_params",
        type=int,
        default=100,
        help="Minimum parameters selected per token",
    )
    parser.add_argument(
        "--max_params",
        type=int,
        default=5000,
        help="Maximum parameters selected per token",
    )
    parser.add_argument(
        "--complexity_exponent",
        type=float,
        default=2.0,
        help="Exponent for complexity scaling",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--max_steps", type=int, default=10000, help="Total training steps"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Peak learning rate"
    )
    parser.add_argument(
        "--pool_lr_mult",
        type=float,
        default=10.0,
        help="Learning rate multiplier for pool parameters",
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument(
        "--save_interval", type=int, default=1000, help="Checkpoint save interval"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=500, help="Evaluation interval"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (cuda/cpu/mps)"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=5,
        help="Number of batches to prefetch per worker",
    )

    return parser.parse_args()


def get_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def save_checkpoint(
    model, optimizer_router, optimizer_pool, scaler, step, loss, output_dir
):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_router_state_dict": optimizer_router.state_dict(),
        "optimizer_pool_state_dict": optimizer_pool.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "loss": loss,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def log_gpu_memory(label, device):
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        print(
            f"[{label}] Memory: Allocated={allocated:.2f}GB, "
            f"Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB"
        )


def train():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"Loading tokenizer: {args.tokenizer_name}")
    print(f"Loading dataset: {args.dataset_name} (streaming)")

    train_loader, vocab_size = get_streaming_dataloader(
        tokenizer_name=args.tokenizer_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        text_column=args.text_column,
        seq_len=args.block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    print("Initializing DPSN Model...")
    config = LanguageModelConfig(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
        pool_size=args.pool_size,
        dropout=args.dropout,
    )

    model = DPSNLanguageModel(
        config,
        router_hidden_dim=args.router_hidden_dim,
        min_params=args.min_params,
        max_params=args.max_params,
        complexity_exponent=args.complexity_exponent,
    )

    model.to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    pool_params = []
    for block in model.blocks:
        pool_params.extend(list(block.dpsn.pool.parameters()))
    pool_param_ids = list(map(id, pool_params))

    other_params = filter(lambda p: id(p) not in pool_param_ids, model.parameters())

    optimizer_router = optim.AdamW(
        other_params, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    optimizer_pool = optim.SGD(pool_params, lr=args.learning_rate * args.pool_lr_mult)

    scaler = GradScaler()

    start_step = 0
    if args.resume_from:
        print(f"Resuming from {args.resume_from}...")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_router.load_state_dict(checkpoint["optimizer_router_state_dict"])
        optimizer_pool.load_state_dict(checkpoint["optimizer_pool_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_step = checkpoint["step"]

    model.train()
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    step = start_step
    running_loss = 0.0
    accum_steps = 0

    optimizer_router.zero_grad()
    optimizer_pool.zero_grad()

    print("Starting Training...")
    torch.cuda.reset_peak_memory_stats(device) if device == "cuda" else None
    t0 = time.time()

    data_iter = iter(train_loader)

    while step < args.max_steps:
        log_gpu_memory("STEP START", device)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        inputs = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)
        log_gpu_memory("DATA LOADED", device)

        with autocast(
            device_type=device,
            dtype=torch.float16 if device == "cuda" else torch.bfloat16,
        ):
            logits, loss, stats = model(inputs, targets)
            loss = loss / args.gradient_accumulation_steps

        log_gpu_memory("FORWARD PASS", device)

        scaler.scale(loss).backward()
        log_gpu_memory("BACKWARD PASS", device)

        running_loss += loss.item() * args.gradient_accumulation_steps
        accum_steps += 1

        if accum_steps % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer_router)
            scaler.unscale_(optimizer_pool)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer_router)
            scaler.step(optimizer_pool)

            scaler.update()
            log_gpu_memory("OPTIMIZER STEP", device)

            optimizer_router.zero_grad()
            optimizer_pool.zero_grad()

            step += 1

            if step % args.log_interval == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                tokens_per_sec = (
                    args.batch_size * args.block_size * args.log_interval
                ) / dt
                avg_loss = running_loss / args.log_interval

                avg_params = sum(s["parameters_used"] for s in stats) / len(stats)
                avg_complexity = sum(
                    s["complexity_score"].mean().item() for s in stats
                ) / len(stats)

                print(
                    f"Step {step}/{args.max_steps} | Loss: {avg_loss:.4f} | "
                    f"Params: {avg_params:.0f} | Cplx: {avg_complexity:.3f} | "
                    f"Tok/s: {tokens_per_sec:.0f}"
                )

                writer.add_scalar("Train/Loss", avg_loss, step)
                writer.add_scalar("DPSN/Avg_Params", avg_params, step)
                writer.add_scalar("DPSN/Avg_Complexity", avg_complexity, step)
                writer.add_scalar("System/TokensPerSec", tokens_per_sec, step)

                running_loss = 0.0

            if step % args.save_interval == 0:
                save_checkpoint(
                    model,
                    optimizer_router,
                    optimizer_pool,
                    scaler,
                    step,
                    avg_loss,
                    args.output_dir,
                )

    print("Training Complete.")
    save_checkpoint(
        model, optimizer_router, optimizer_pool, scaler, step, 0.0, args.output_dir
    )
    writer.close()


if __name__ == "__main__":
    train()
