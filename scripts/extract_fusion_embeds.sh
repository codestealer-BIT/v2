#!/bin/bash

# This script is used for Pyramid-Flow Video Generation Training (Using Temporal Pyramid and autoregressive training)
# It enables the autoregressive video generative training with temporal pyramid
# make sure to set, NUM_FRAMES % VIDEO_SYNC_GROUP == 0; GPUS % VIDEO_SYNC_GROUP == 0

# ps -ef | grep train/train_video_music_clip.py | grep -v grep | cut -c 9-16 | xargs kill -9

GPUS=2  # The gpu number
MODEL_PATH="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_music_clip_dedup_v3_large_queue"
BLIP_PATH="/mnt/bn/yexiaoyu-test/models/huggingface/blip2"
OUTPUT_DIR="/mnt/bn/yexiaoyu-test/data/faiss"
PRETRAINED_PATH="/mnt/bn/yexiaoyu-test/checkpoints/stage_2_formal_training/checkpoint-158000"

BATCH_SIZE=64
NUM_FRAMES=8

HDFS_FILE="hdfs://harunasg/home/byte_data_tt_m/jinyang.leo/tiktok_video_data/data"

# For the 768p version, make sure to add the args:  --gradient_checkpointing
# ${ARNOLD_WORKER_NUM} \
# 	--node_rank ${ARNOLD_ID}  --master_addr ${METIS_WORKER_0_HOST} --master_port ${METIS_WORKER_0_PORT} \
# 	--nproc_per_node ${ARNOLD_WORKER_GPU} \z

torchrun --nnodes=${ARNOLD_WORKER_NUM} \
	--node_rank ${ARNOLD_ID}  --master_addr ${METIS_WORKER_0_HOST} --master_port ${METIS_WORKER_0_PORT} \
	--nproc_per_node ${ARNOLD_WORKER_GPU} \
    tools/build_faiss_database_ddp.py \
    --model_name video_siglip2 \
    --model_path $MODEL_PATH \
    --pretrained_path $PRETRAINED_PATH \
    --blip_path $BLIP_PATH \
    --model_dtype bf16 \
    --interpolate 6 \
    --max_short_len 64 \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_workers 8 \
    --max_frames $NUM_FRAMES \
    --hdfs_file_path $HDFS_FILE


#     --freeze_text \
# torchrun --nproc_per_node ${GPUS} \