#!/bin/bash

for param in $(seq 150000 2000 172000); do
    echo "Running with checkpoint-$param..."
    CUDA_VISIBLE_DEVICES=0 python3 evaluate_video_retrieval_tiktok_fusion.py \
        --pretrained_path "/mnt/bn/yexiaoyu-test/checkpoints/stage_2_formal_training/checkpoint-$param" \
        --output_path "/mnt/bn/yexiaoyu-test/checkpoints/stage_2_formal_training"
done
