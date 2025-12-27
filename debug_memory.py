import torch
import torch.nn as nn
from config.hyperparameters import LanguageModelConfig


def check_memory():
    config = LanguageModelConfig(
        vocab_size=50257,
        n_embd=768,
        n_head=12,
        n_layer=12,
        block_size=1024,
        pool_size=1300000,
        dropout=0.1,
    )

    print(f"Config: n_layer={config.n_layer}, pool_size={config.pool_size}")

    pool_params_per_layer = config.pool_size * config.n_embd
    total_pool_params = pool_params_per_layer * config.n_layer

    print(
        f"Params per pool: {config.pool_size} * {config.n_embd} = {pool_params_per_layer:,}"
    )
    print(f"Total pool params (12 layers): {total_pool_params:,}")
    print(f"Total pool memory (float32): {total_pool_params * 4 / (1024**3):.2f} GB")

    print("\n--- Testing actual memory allocation ---")

    print("Creating single embedding layer...")
    single_emb = nn.Embedding(config.pool_size, config.n_embd, sparse=True)
    single_mem = single_emb.weight.element_size() * single_emb.weight.numel()
    print(f"Single embedding memory: {single_mem / (1024**3):.2f} GB")

    print("\nCreating 12 layers (what your model does)...")
    layers = []
    total_memory = 0
    for i in range(config.n_layer):
        emb = nn.Embedding(config.pool_size, config.n_embd, sparse=True)
        layers.append(emb)
        layer_mem = emb.weight.element_size() * emb.weight.numel()
        total_memory += layer_mem
        print(f"Layer {i + 1}: {layer_mem / (1024**3):.2f} GB")

    print(f"\nTotal memory for all layers: {total_memory / (1024**3):.2f} GB")

    print("\n--- Testing sparse vs dense memory usage ---")
    dense_emb = nn.Embedding(100000, 768, sparse=False)
    sparse_emb = nn.Embedding(100000, 768, sparse=True)

    dense_mem = dense_emb.weight.element_size() * dense_emb.weight.numel()
    sparse_mem = sparse_emb.weight.element_size() * sparse_emb.weight.numel()

    print(f"Dense embedding memory: {dense_mem / (1024**2):.2f} MB")
    print(f"Sparse embedding memory: {sparse_mem / (1024**2):.2f} MB")
    print(f"Memory saved by sparse: {(dense_mem - sparse_mem) / (1024**2):.2f} MB")


if __name__ == "__main__":
    check_memory()
