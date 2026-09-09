# checkpoint_path="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_residual_60m_gt8_title_lang_loss_pgc_popularity/checkpoint-62000"
# checkpoint_path="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_residual_60m_gt7_title_lang_loss_pgc_popularity/checkpoint-66000"
# checkpoint_path="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_residual_60m_gt7_title_lang_loss_pgc_popularity/checkpoint-146000/"
# checkpoint_path="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_residual_60m_gt7_title_pgc_popularity/checkpoint-68000/"
# checkpoint_path="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_gate_60m_gt7_title_pgc_popularity/checkpoint-148000/"
# checkpoint_path="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_all_lang_new_gate_60m_gt7_rescale_05/checkpoint-158000/"
checkpoint_path="/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/checkpoints/video_siglip2_for_music_1029_dis_all_lang_new_gate_v2_60m_gt7_rescale_05_ct_drop_05_llm_v5/checkpoint-60000"

# torchrun --nproc_per_node=4 music_evaluation_tools_add_lang_country/publish_eval_infer.py \
#     --base_model_path /mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base \
#     --checkpoint_path ${checkpoint_path} \
#     --data_path /mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/data/evalset_negative.json \
#     --tokenizer_dir /mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base \
#     --output_dir auc_results/gate_v2_ct_drop_05_20_negative_llm_v5_pred \
#     --batch_size 128 \
#     --num_workers 1 \
#     --max_frame_len 8 \
#     --embed_lang_caption \
#     --add_title_text \
#     --user_info_fusion_method gate \
#     --merge_output

torchrun --nproc_per_node=4 music_evaluation_tools_add_lang_country/publish_eval_infer.py \
    --base_model_path /mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base \
    --checkpoint_path ${checkpoint_path} \
    --data_path /mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/data/evalset_positive.json \
    --tokenizer_dir /mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base \
    --output_dir auc_results/gate_v2_ct_drop_05_20_positive_llm_v5_pred \
    --batch_size 128 \
    --num_workers 1 \
    --max_frame_len 8 \
    --embed_lang_caption \
    --add_title_text \
    --user_info_fusion_method gate \
    --merge_output


