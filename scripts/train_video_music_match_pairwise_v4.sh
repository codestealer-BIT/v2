#!/bin/bash

# GPUS=2  # The gpu number
MODEL_PATH="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"
PRETRAINED_PATH="/mnt/bn/yexiaoyu-test/checkpoints/stage_1_final_frame_mask/checkpoint-100000"
OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/checkpoints/only_match_50m_huber_01_v4"


BATCH_SIZE=128
GRAD_ACCU_STEPS=1
NUM_FRAMES=5
# ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/music_captions/llm_judge_pretrain_data_pgc_1023_28m.json"
# ANNO_FILE='/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/llm_judge_pretrain_data_pgc_1023_28m_w_user_lang.json'
# ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/llm_judge_pretrain_data_pgc_1023_41m_new.json"
# ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/llm_judge_pretrain_data_pgc_1106_61m.json"
# ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/llm_judge_pretrain_data_pgc_1106_60m.json"

ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/train_data_pgc_v5_score_20260116_50m.json"
# ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/data/train_data_pgc_v5_score_20260116_50m_plus_listwise_data_thres_2_pair_200.json"
# ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/data/2000_listwise_data_thres_2_pair_200.json"

# ${ARNOLD_WORKER_NUM} \
# 	--node_rank ${ARNOLD_ID}  --master_addr ${METIS_WORKER_0_HOST} --master_port ${METIS_WORKER_0_PORT} \
# 	--nproc_per_node ${ARNOLD_WORKER_GPU} \

MASTER_PORT=$(echo "$METIS_WORKER_0_PORT" | cut -d ',' -f 1)

torchrun --nnodes=1 \
	--node_rank 0  --master_addr "127.0.0.1" --master_port ${MASTER_PORT} \
	--nproc_per_node ${ARNOLD_WORKER_GPU} \
    train/train_video_siglip2_for_music_rec.py \
    --model_name video_siglip2_for_music_only_match \
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
    --iters_per_epoch 5000 \
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
    --huber_loss_delta 0.1 \
    # --add_user_lang_and_country_code True \
    --filter_music_id_file /mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/data/all_filter_music_ids_thres_10_4.0_2.0 \
    --use_frame_mask &> error.log


#     --freeze_text \
# torchrun --nproc_per_node ${GPUS} \