#!/bin/bash

# This script is used for Pyramid-Flow Video Generation Training (Using Temporal Pyramid and autoregressive training)
# It enables the autoregressive video generative training with temporal pyramid
# make sure to set, NUM_FRAMES % VIDEO_SYNC_GROUP == 0; GPUS % VIDEO_SYNC_GROUP == 0

# ps -ef | grep train/train_video_music_clip.py | grep -v grep | cut -c 9-16 | xargs kill -9

GPUS=2  # The gpu number
MODEL_PATH="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_music_clip_dedup_v3_large_queue"
BLIP_PATH="/mnt/bn/yexiaoyu-test/models/huggingface/blip2"
OUTPUT_DIR="/mnt/bn/yexiaoyu-test/checkpoints/stage_2_final_no_frame_mask"
PRETRAINED_PATH="/mnt/bn/yexiaoyu-test/checkpoints/stage_1_final_no_frame_mask/checkpoint-120000"

BATCH_SIZE=16
GRAD_ACCU_STEPS=2
NUM_FRAMES=8

HDFS_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/final_tiktok_data_for_small_model/finetune_v1_data_plus_files_filtered.json"
HDFS_DICT_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/final_tiktok_data_for_small_model/"
INTERN_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/vd-foundation___InternVid-10M-FLT/intervid_anno/InternVid-10M-FLT-INFO_with_path_long_caption.jsonl"

LARGE_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/final_tiktok_data_for_small_model/pretrain_data_0911_100m_data_files_filtered.json"
FILE_CACHE="/mnt/bn/yexiaoyu-test/data/cache"

# For the 768p version, make sure to add the args:  --gradient_checkpointing
# ${ARNOLD_WORKER_NUM} \
# 	--node_rank ${ARNOLD_ID}  --master_addr ${METIS_WORKER_0_HOST} --master_port ${METIS_WORKER_0_PORT} \
# 	--nproc_per_node ${ARNOLD_WORKER_GPU} \z

torchrun --nnodes=${ARNOLD_WORKER_NUM} \
	--node_rank ${ARNOLD_ID}  --master_addr ${METIS_WORKER_0_HOST} --master_port ${METIS_WORKER_0_PORT} \
	--nproc_per_node ${ARNOLD_WORKER_GPU} \
    train/train_video_siglip2_short_long_blip.py \
    --model_name video_siglip2 \
    --model_path $MODEL_PATH \
    --pretrained_path $PRETRAINED_PATH \
    --blip_path $BLIP_PATH \
    --model_dtype bf16 \
    --interpolate 6 \
    --max_short_len 64 \
    --gradient_accumulation_steps $GRAD_ACCU_STEPS \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_workers 4 \
    --max_frames $NUM_FRAMES \
    --hdfs_file_path $HDFS_FILE \
    --hdfs_dict_path $HDFS_DICT_FILE \
    --internvid_file_path $INTERN_FILE \
    --large_file_path $LARGE_FILE \
    --large_dict_path $HDFS_DICT_FILE \
    --filelist_cache $FILE_CACHE \
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
    --vision_encoder_lr 1e-5 \
    --vision_head_lr 1e-5 \
    --text_encoder_lr 1e-5 \
    --freeze_text


#     --freeze_text \
# torchrun --nproc_per_node ${GPUS} \