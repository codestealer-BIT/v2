#!/bin/bash

GPUS=8  # The gpu number
# MODEL_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_music_clip/checkpoint-70000/pytorch_model.bin"  
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_baseline/non_ema"    # The checkpoint saving dir

# MODEL_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_music_clip/checkpoint-170000/pytorch_model.bin"  
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_baseline/checkpoint-170000/non_ema"    # The checkpoint saving dir

# MODEL_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_music_clip/checkpoint-170000/pytorch_model_ema.bin"  
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_baseline/checkpoint-170000/ema"    # The checkpoint saving dir

# MODEL_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_music_clip/checkpoint-200000/pytorch_model_ema.bin"  
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_baseline/checkpoint-200000/ema"    # The checkpoint saving dir

# MODEL_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_music_clip_dedup_v2/checkpoint-200000/pytorch_model.bin"  
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_baseline_dedup/checkpoint-200000/non_ema"    # The checkpoint saving dir

# MODEL_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_music_clip_dedup_v2/checkpoint-200000/pytorch_model_ema.bin"  
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_baseline_dedup/checkpoint-200000/ema"    # The checkpoint saving dir

# MODEL_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_music_clip_dedup_v3_large_queue/checkpoint-200000/pytorch_model_ema.bin"  
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_dedup_v3_large_queue/checkpoint-200000/ema"    # The checkpoint saving dir

MODEL_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_music_text_clip/checkpoint-200000/pytorch_model_ema.bin"  
OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_text_clip/checkpoint-200000/ema"    # The checkpoint saving dir

# MODEL_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/models/Multimodal-Embedding/model_state_epoch_170000.th"  
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_baseline/origin_model"    # The checkpoint saving dir

BATCH_SIZE=128    # It should satisfy batch_size % 4 == 0
NUM_FRAMES=5         # e.g., 16 for 5s, 32 for 10s
ANNO_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/eval_v1_data"   # The video annotation file path
CONFIG_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/MM-Embedding/experiments/config_finetune_mmembed_v3.yaml"

# MODEL_NAME="video_music_clip"
MODEL_NAME="video_music_text_clip"
DATASET_NAME="batch_eval"

# For the 768p version, make sure to add the args:  --gradient_checkpointing

torchrun --nnodes=${ARNOLD_WORKER_NUM} \
	--node_rank ${ARNOLD_ID}  --master_addr ${METIS_WORKER_0_HOST} --master_port ${METIS_WORKER_0_PORT} \
	--nproc_per_node ${ARNOLD_WORKER_GPU} \
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