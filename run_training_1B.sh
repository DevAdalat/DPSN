#!/bin/bash

# Configuration for 1B total parameters model training
# Note: "1B Pool Size" refers to the TOTAL number of parameters in the model.
# Since the pool is replicated across n_layer (12) blocks, we need to divide the total target by n_layer.
# Target: ~1.2B params (1B pool + 0.2B router/others).
# To fit in 15GB VRAM (T4), we need to be careful.
# Params per layer = 83.3M params.
# Pool size (entries) = 83.3M / 768 (input_dim) ≈ 108,500.
# REDUCED to 90,000 to prevent OOM on 16GB cards.
# New Total Params ≈ 1.25B (Safe for T4 with AMP).

# Run training
python train.py \
    --dataset_name "HuggingFaceFW/fineweb" \
    --dataset_config "sample-10BT" \
    --tokenizer_name "gpt2" \
    --text_column "text" \
    --n_layer 12 \
    --n_head 12 \
    --n_embd 768 \
    --block_size 1024 \
    --pool_size 90000 \
    --router_hidden_dim 256 \
    --min_params 100 \
    --max_params 5000 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_steps 50000 \
    --learning_rate 3e-4 \
    --pool_lr_mult 10.0 \
    --output_dir "./checkpoints_1B_fineweb" \
    --num_workers 8 \
    --prefetch_factor 5 \
    --device "cuda"

# Notes on performance flags used:
# --num_workers 8: Uses 8 CPU threads to preprocess/tokenize data in parallel
# --prefetch_factor 5: Each worker prefetches 5 batches ahead (total 40 batches in queue)
# --device "cuda": Forces GPU usage
