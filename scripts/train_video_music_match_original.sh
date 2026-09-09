#!/bin/bash

# GPUS=2  # The gpu number
MODEL_PATH="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base" #siglip2原生模型，不需要改
PRETRAINED_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/stage_1_final_frame_mask/checkpoint-100000" #金阳的预训练模型，不重要
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_siglip2_for_music_baseline"   # CHANGE
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_gate_60m_gt7_title_pgc_popularity" # before 162000
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_gate_60m_gt7_title_pgc_popularity_m1000w"
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/checkpoints/video_siglip2_for_music_only_match_huber_loss_delta_1_filter_music_ids_fix_logit"
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_all_lang_new_gate_60m_gt7_rescale_05"
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_50m_xiuqi_branch"
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_51m"
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/guyanhang/checkpoints/video_siglip2_for_music_only_match_54m"
OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/guyanhang/checkpoints/exp42_video_siglip2_for_music_only_match_64m_without_filter_withoout_50m_score1_2"


# HISTORICAL_MODEL_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_binary_classification_pos_ge9_neg_le4/checkpoint-80000/pytorch_model.bin"
HISTORICAL_MODEL_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_siglip2_for_music_baseline/checkpoint-94000/pytorch_model.bin" #对比学习的判别器，不重要


BATCH_SIZE=128
GRAD_ACCU_STEPS=1
NUM_FRAMES=5
# ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/music_captions/llm_judge_pretrain_data_pgc_1023_28m.json"
# ANNO_FILE='/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/llm_judge_pretrain_data_pgc_1023_28m_w_user_lang.json'
# ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/llm_judge_pretrain_data_pgc_1023_41m_new.json"
# ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/llm_judge_pretrain_data_pgc_1106_61m.json"
# ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/llm_judge_pretrain_data_pgc_1106_60m.json"

# ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/train_data_pgc_v5_score_20260116_50m.json"
# ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/train_data_v5_score_20260331_5000w_n_300w_high_5100w_shuffled.json"
# ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/fr_train_data_pgc_v5_score_20260416_54m.json"
# ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/train_data_v5_score_va_20260424_54m.json"
# ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/fr_train_data_pgc_v5_score_20260425_64m.json"
ANNO_FILE="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/fr_train_data_pgc_v5_score_20260507_64m_remove_score_1_2_mixed.json"
# ${ARNOLD_WORKER_NUM} \
# 	--node_rank ${ARNOLD_ID}  --master_addr ${METIS_WORKER_0_HOST} --master_port ${METIS_WORKER_0_PORT} \
# 	--nproc_per_node ${ARNOLD_WORKER_GPU} \

MASTER_PORT=$(echo "$METIS_WORKER_0_PORT" | cut -d ',' -f 1)

torchrun --nnodes=${ARNOLD_WORKER_NUM} \
	--node_rank ${ARNOLD_ID}  --master_addr "127.0.0.1" --master_port ${MASTER_PORT} \
	--nproc_per_node ${ARNOLD_WORKER_GPU} \
    train/train_video_siglip2_for_music_rec.py \
    --model_name video_siglip2_for_music_only_match \
    --model_path $MODEL_PATH \
    --pretrained_path $PRETRAINED_PATH \
    --historical_model_path $HISTORICAL_MODEL_PATH \
    --historical_score_threshold 0.1 \
    --model_dtype bf16 \
    --interpolate 6 \
    --max_short_len 64 \
    --false_positive_filter True \
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
    --mbias_delta 0.0 \
    --use_frame_mask
    # --filter_music_id_file /mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/data/all_filter_music_ids_10_1.0-1.4_4.6_5.0_100_1.0-1.9_4.0_5.0 \


#     --freeze_text \
# torchrun --nproc_per_node ${GPUS} \