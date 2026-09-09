#!/bin/bash

# This script is used for Pyramid-Flow Video Generation Training (Using Temporal Pyramid and autoregressive training)
# It enables the autoregressive video generative training with temporal pyramid
# make sure to set, NUM_FRAMES % VIDEO_SYNC_GROUP == 0; GPUS % VIDEO_SYNC_GROUP == 0

# ps -ef | grep train/train_video_music_clip.py | grep -v grep | cut -c 9-16 | xargs kill -9

GPUS=2  # The gpu number
MODEL_PATH="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_music_clip_dedup_v3_large_queue"
OUTPUT_DIR="/mnt/bn/yexiaoyu-test/checkpoints/debug_long_clip_holding_up_space"

BATCH_SIZE=64
GRAD_ACCU_STEPS=1
NUM_FRAMES=8
# ANNO_FILE="hdfs://harunasg/home/byte_data_tt_m/jinyang.leo/music_rec_data/finetune_v1_data"
ANNO_FILE="hdfs://harunasg/home/byte_data_tt_m/jinyang.leo/music_rec_data/finetune_v1_data_plus"

# For the 768p version, make sure to add the args:  --gradient_checkpointing

torchrun --nnodes=${ARNOLD_WORKER_NUM} \
	--node_rank ${ARNOLD_ID}  --master_addr ${METIS_WORKER_0_HOST} --master_port ${METIS_WORKER_0_PORT} \
	--nproc_per_node ${ARNOLD_WORKER_GPU} \
    train/train_video_siglip2_long_caption.py \
    --model_name video_siglip2 \
    --model_path $MODEL_PATH \
    --model_dtype bf16 \
    --gradient_accumulation_steps $GRAD_ACCU_STEPS \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_workers 16 \
    --max_frames $NUM_FRAMES \
    --max_text_len 196 \
    --anno_file_path $ANNO_FILE \
    --lr_scheduler cosine \
    --queue_scale 0 \
    --opt adamw \
    --gradient_checkpointing \
    --opt_beta1 0.9 \
    --opt_beta2 0.95 \
    --seed 42 \
    --weight_decay 1e-4 \
    --clip_grad 1.0 \
    --lr 1e-4 \
    --warmup_steps 2000 \
    --epochs 100 \
    --iters_per_epoch 2000 \
    --report_to tensorboard \
    --print_freq 40 \
    --save_ckpt_freq 1 \
    --load_vision \
    --load_text \
    --freeze_vision \
    --freeze_text \


# torchrun --nproc_per_node ${GPUS} \