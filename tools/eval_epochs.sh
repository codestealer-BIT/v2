#!/bin/bash

for param in $(seq 100000 2000 112000); do
    echo "Running with checkpoint-$param..."
    CUDA_VISIBLE_DEVICES=0 python3 evaluate_video_retrieval_tiktok.py \
        --short \
        --pretrained_path "/mnt/bn/yexiaoyu-test/checkpoints/stage_1_large_dataset_formal_adjust_ratio/checkpoint-$param" \
        --output_path "/mnt/bn/yexiaoyu-test/checkpoints/stage_1_large_dataset_formal_adjust_ratio"
done
