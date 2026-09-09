#!/bin/bash

GPUS=8  # The gpu number
MODEL_PATH="/mnt/bn/search-ad-creative-zhijie/wangxiuqi.0601/mm/huggingface/siglip2-base"
PRETRAINED_PATH="/mnt/bn/search-ad-creative-zhijie/wangxiuqi.0601/mm/checkpoints/stage_1_final_frame_mask/checkpoint-100000"
OUTPUT_DIR="/mnt/bn/tt-search-ads-nas/wangxiuqi.0601/checkpoints/video_siglip2_for_music_baseline_retrain_nf"
HISTORICAL_MODEL_PATH="/mnt/bn/search-ad-creative-zhijie/wangxiuqi.0601/mm/checkpoints/video_siglip2_for_music_baseline/checkpoint-94000/pytorch_model.bin"
BATCH_SIZE=64
GRAD_ACCU_STEPS=1
NUM_FRAMES=5
ANNO_FILE="/mnt/bn/search-ad-creative-zhijie/wangxiuqi.0601/mm/llm_judge_pretrain_data_pgc_1023_28m_va.json"

# ${ARNOLD_WORKER_NUM} \
# 	--node_rank ${ARNOLD_ID}  --master_addr ${METIS_WORKER_0_HOST} --master_port ${METIS_WORKER_0_PORT} \
# 	--nproc_per_node ${ARNOLD_WORKER_GPU} \

MASTER_PORT=$(echo "$METIS_WORKER_0_PORT" | cut -d ',' -f 1)


torchrun --nnodes=${ARNOLD_WORKER_NUM} \
	--node_rank ${ARNOLD_ID}  --master_addr "127.0.0.1" --master_port ${MASTER_PORT} \
	--nproc_per_node ${ARNOLD_WORKER_GPU} \
    train/train_video_siglip2_for_music_rec.py \
    --model_name video_siglip2_for_music_negative_filtering \
    --model_path $MODEL_PATH \
    --pretrained_path $PRETRAINED_PATH \
    --historical_model_path $HISTORICAL_MODEL_PATH \
    --historical_score_threshold 0.02 \
    --model_dtype bf16 \
    --interpolate 6 \
    --false_positive_filter False \
    --max_short_len 64 \
    --gradient_accumulation_steps $GRAD_ACCU_STEPS \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_workers 1 \
    --max_frames $NUM_FRAMES \
    --anno_file_path $ANNO_FILE \
    --random_drop_frame \
    --random_rotate_video \
    --low_quality_score 4 \
    --lr_scheduler cosine \
    --queue_scale 0 \
    --opt adamw \
    --opt_beta1 0.9 \
    --opt_beta2 0.95 \
    --seed 42 \
    --weight_decay 1e-4 \
    --clip_grad 1.0 \
    --lr 1e-4 \
    --warmup_steps 2000 \
    --epochs 50 \
    --iters_per_epoch 2000 \
    --report_to tensorboard \
    --print_freq 40 \
    --save_ckpt_freq 1 \
    --vision_encoder_lr 1e-5 \
    --text_encoder_lr 1e-5 \
    --vision_head_lr 3e-5 \
    --freeze_text \
    --use_dwa \
    --use_frame_mask 2>error.log


#     --freeze_text \
# torchrun --nproc_per_node ${GPUS} \