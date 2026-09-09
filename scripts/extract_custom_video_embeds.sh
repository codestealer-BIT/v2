#!/bin/bash

GPUS=1  # The gpu number

MODEL_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_music_clip_dedup_v3_large_queue/checkpoint-200000/pytorch_model_ema.bin"  
OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_dedup_v3_large_queue/checkpoint-200000/ema_custom"    # The checkpoint saving dir

# MODEL_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_music_text_clip/checkpoint-200000/pytorch_model_ema.bin" 
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_text_clip/checkpoint-200000/ema_custom"    # The checkpoint saving dir

BATCH_SIZE=8    # It should satisfy batch_size % 4 == 0
NUM_FRAMES=5         # e.g., 16 for 5s, 32 for 10s

ANNO_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/MM-Embedding/demo_video.txt"    # The video annotation file path
CONFIG_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/MM-Embedding/experiments/config_finetune_mmembed_v3.yaml"

MODEL_NAME="video_music_clip"
# MODEL_NAME="video_music_text_clip"
DATASET_NAME="custom_dataset"

# For the 768p version, make sure to add the args:  --gradient_checkpointing

torchrun --nproc_per_node ${GPUS} \
    tools/extract_video_embeds.py \
    --model_name $MODEL_NAME \
    --model_path $MODEL_PATH \
    --model_dtype bf16 \
    --cfg_path $CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --dataset_type $DATASET_NAME \
    --num_workers 16 \
    --max_frames $NUM_FRAMES \
    --anno_dir $ANNO_DIR