#!/bin/bash

GPUS=8  # The gpu number

# MODEL_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_local_service/checkpoint-4000/pytorch_model.bin"  
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/results/video_local_service/checkpoint-4000/non_ema"    # The checkpoint saving dir

MODEL_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_local_service_only_gpt/checkpoint-6000/pytorch_model_ema.bin"  
OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/results/video_local_service_only_gpt/checkpoint-6000/ema"    # The checkpoint saving dir

BATCH_SIZE=16    # It should satisfy batch_size % 4 == 0
NUM_FRAMES=8         # e.g., 16 for 5s, 32 for 10s
ANNO_DIR="hdfs://harunava/user/huangchenzhe.7/LocalService/ls_model/datasets/val_cmb-LS-train"   # The video annotation file path
CONFIG_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/MM-Embedding/experiments/config_finetune_mmembed_v3.yaml"

MODEL_NAME="video_local_service"

# For the 768p version, make sure to add the args:  --gradient_checkpointing

torchrun --nnodes=${ARNOLD_WORKER_NUM} \
	--node_rank ${ARNOLD_ID}  --master_addr ${METIS_WORKER_0_HOST} --master_port ${METIS_WORKER_0_PORT} \
	--nproc_per_node ${ARNOLD_WORKER_GPU} \
    tools/evaluate_local_service.py \
    --model_name $MODEL_NAME \
    --model_path $MODEL_PATH \
    --model_dtype bf16 \
    --cfg_path $CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_workers 16 \
    --max_frames $NUM_FRAMES \
    --anno_dir $ANNO_DIR