GPUS=2  # The gpu number

MASTER_PORT=$(echo "$METIS_WORKER_0_PORT" | cut -d ',' -f 1)
MODEL_PATH="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"

# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_discriminator_bm_0.1_lowq_7_real_residual_add_user_info/checkpoint-100000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_add_user_info_residual_41m/checkpoint-100000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_add_user_info_residual_41m_gt8_top20lang/checkpoint-60000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_add_user_info_residual_41m_gt8_top20lang_maxa/checkpoint-78000"
CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_bm_filtering_41m_gt8/checkpoint-100000"

torchrun --nnodes=${ARNOLD_WORKER_NUM} \
	--node_rank ${ARNOLD_ID}  --master_addr ${METIS_WORKER_0_HOST} --master_port ${MASTER_PORT} \
    --nproc_per_node ${ARNOLD_WORKER_GPU} \
    music_evaluation_tools/encode_music_embedding.py \
    --model_path $MODEL_PATH \
    --data_path $CKPT_PATH \
    --model_dtype bf16 
# torchrun --nproc_per_node ${GPUS} \