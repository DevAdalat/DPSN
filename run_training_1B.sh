#!/bin/bash

# Configuration for 1B total parameters model training
# Note: "1B Pool Size" usually refers to the number of parameter slots.
# Assuming input_dim=768 (default), 1M pool size = 768M params.
# To get approx 1B total parameters, we use pool_size=1,300,000 (~1B params)
# Router is kept small (~200M params with hidden=256).

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
    --pool_size 1300000 \
    --router_hidden_dim 256 \
    --min_params 100 \
    --max_params 5000 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
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
