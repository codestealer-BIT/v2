#!/bin/bash

MODEL_PATH="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"
PRETRAINED_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/stage_1_final_frame_mask/checkpoint-100000"
OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_contrastive_n_regre_51m_exp38"

BATCH_SIZE=128
GRAD_ACCU_STEPS=1
NUM_FRAMES=5
# ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/train_data_v5_score_20260331_5000w_n_300w_high_5100w_shuffled.json"
ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/fr_train_data_v5_score_20260331_5000w_n_300w_high_5100w_shuffled.json"

MASTER_PORT=$(echo "$METIS_WORKER_0_PORT" | cut -d ',' -f 1)

torchrun --nnodes=${ARNOLD_WORKER_NUM} \
	--node_rank ${ARNOLD_ID}  --master_addr ${METIS_WORKER_0_HOST} --master_port ${MASTER_PORT} \
	--nproc_per_node ${ARNOLD_WORKER_GPU} \
    train/train_video_siglip2_for_music_rec.py \
    --model_name video_siglip2_for_music_only_match_contrastive \
    --model_path $MODEL_PATH \
    --pretrained_path $PRETRAINED_PATH \
    --model_dtype bf16 \
    --interpolate 6 \
    --max_short_len 64 \
    --gradient_accumulation_steps $GRAD_ACCU_STEPS \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_workers 8 \
    --max_frames $NUM_FRAMES \
    --anno_file_path $ANNO_FILE \
    --random_drop_frame \
    --random_rotate_video \
    --low_quality_score 2 \
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
    --iters_per_epoch 4900 \
    --report_to tensorboard \
    --print_freq 40 \
    --save_ckpt_freq 1 \
    --vision_encoder_lr 1e-5 \
    --text_encoder_lr 1e-5 \
    --vision_head_lr 3e-5 \
    --freeze_text \
    --add_title_text True \
    --embed_lang_caption True \
    --force_rescale_prob 0.5 \
    --enable_llm_v5 True \
    --min_score 0 \
    --regression_loss_weight 0.05 \
    --pos_ins_thresh 4 \
    --mbias_delta 0.0 \
    --filter_music_id_file /mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/data/all_filter_music_ids_10_1.0-1.4_4.6_5.0_100_1.0-1.9_4.0_5.0 \
    --use_frame_mask 2>error.log
