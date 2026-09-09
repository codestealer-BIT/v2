#!/bin/bash

GPUS=2  # Number of GPUs on this instance

# Use SigLIP2 base config and a fine-tuned checkpoint
MODEL_PATH="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_baseline/checkpoint-200000/pytorch_model.bin"
# CKPT_PATH='/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_siglip2_for_music_baseline/checkpoint-94000/pytorch_model.bin'
CKPT_PATH='/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_discriminator_bm_0.1_lowq_7_real_residual_add_user_info/checkpoint-100000/pytorch_model.bin'

# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_text_clip/checkpoint-200000/ema_custom_full_library_eval"    # The checkpoint saving dir
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/video_embed_filter_train_fusion_music_ue_v2/train_music_v2_ckpt_84000_ema"
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/video_embed_filter_train_fusion_music_ue_v2/previous_baseline"
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/video_embed_filter_train_fusion_music_ue_v2/train_music_v2_fix_lr_dup_finetune_ckpt_12000_ema"
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/evals_video_emb/1029_jy94000"
OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/evals_video_emb/bm_0.1_lowq_7_real_residual_add_user_info_100000"

BATCH_SIZE=16    # Per-process batch size
NUM_FRAMES=5         # e.g., 5 frames

# ANNO_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/MM-Embedding/demo_video.txt"    # The video annotation file path
ANNO_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/MM-Embedding/music_eval_video_data_17w.jsonl"
CONFIG_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/MM-Embedding/experiments/config_finetune_mmembed_v3.yaml"


DATASET_NAME="custom_dataset"

# For the 768p version, make sure to add the args:  --gradient_checkpointing
MASTER_PORT=$(echo "$METIS_WORKER_0_PORT" | cut -d ',' -f 1)

torchrun --nnodes=${ARNOLD_WORKER_NUM} \
	--node_rank ${ARNOLD_ID}  --master_addr ${METIS_WORKER_0_HOST} --master_port ${MASTER_PORT} \
	--nproc_per_node ${ARNOLD_WORKER_GPU} \
    music_evaluation_tools_add_lang_country/extract_video_embeds_v2.py \
    --model_path $MODEL_PATH \
    --ckpt_path $CKPT_PATH \
    --model_dtype bf16 \
    --cfg_path $CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --dataset_type $DATASET_NAME \
    --num_workers 16 \
    --max_frames $NUM_FRAMES \
    --anno_dir $ANNO_DIR \
    --add_user_lang_and_country_method residual
