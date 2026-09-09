GPUS=8  # The gpu number

MASTER_PORT=$(echo "$METIS_WORKER_0_PORT" | cut -d ',' -f 1)
MODEL_PATH="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_discriminator_bm_0.1_lowq_7_real_residual_add_user_info/checkpoint-100000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_add_user_info_residual_41m/checkpoint-100000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_add_user_info_residual_41m_gt8_top20lang/checkpoint-60000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_add_user_info_residual_41m_gt8_top20lang_maxa/checkpoint-78000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_discriminator_1029_0.08_lowq_7_gate_add_user_info/checkpoint-60000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_residual_41m_gt8_maxa/checkpoint-100000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_residual_41m_gt8_title/checkpoint-40000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_residual_61m_gt8_title/checkpoint-120000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_residual_61m_gt8_title_lang_loss/checkpoint-36000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_residual_60m_gt8_title_lang_loss_pgc_popularity/checkpoint-62000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_residual_60m_gt7_title_lang_loss_pgc_popularity/checkpoint-146000/"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_gate_60m_gt7_title_pgc_popularity/checkpoint-148000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_all_lang_new_gate_60m_gt7_rescale_05/checkpoint-158000" #todo: change to 0.0
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_llm_v5_filter_music_220w_continue_from_39200_unfreeze_text/checkpoint-40000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_llm_v5_300w_continue_from_39200/checkpoint-40000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_llm_v5_300w/checkpoint-30000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_highscore_pointwise_300w_45p_1n_bce/checkpoint-6000"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_cont_from_0129_80w_regression/checkpoint-39500"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_cont_from_0129_80w_regression_p45_n1/checkpoint-39500"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_cont_from_0129_80w_regression_2w2/checkpoint-39500"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_51m/checkpoint-39200"
# CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_contrastive_51m/checkpoint-73500"
CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/guyanhang/checkpoints/exp41_video_siglip2_for_music_only_match_64m_without_filter/checkpoint-220500"
# torchrun --nnodes=${ARNOLD_WORKER_NUM} \
# 	--node_rank ${ARNOLD_ID}  --master_addr ${METIS_WORKER_0_HOST} --master_port ${MASTER_PORT} \
#     --nproc_per_node ${ARNOLD_WORKER_GPU} \
#     music_evaluation_tools_add_lang_country/encode_music_embedding.py \
#     --model_path $MODEL_PATH \
#     --data_path $CKPT_PATH \
#     --model_dtype bf16 \
#     --add_user_lang_and_country_method gate \
#     --add_title_text True \
#     --embed_lang_caption True

# torchrun --nproc_per_node ${GPUS} \


# for deployment batch inferrence
torchrun --nnodes=${ARNOLD_WORKER_NUM} \
	--node_rank ${ARNOLD_ID}  --master_addr "127.0.0.1" --master_port ${MASTER_PORT} \
    --nproc_per_node ${ARNOLD_WORKER_GPU} \
    music_evaluation_tools_add_lang_country/encode_music_embedding_batch.py \
    --model_path $MODEL_PATH \
    --data_path $CKPT_PATH \
    --model_dtype bf16 \
    --add_user_lang_and_country_method gate \
    --add_title_text True \
    --embed_lang_caption True \
    --output_pt True