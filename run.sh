# train with tmux
cd /mnt/bn/jiny-ttls-i18n-fr1q/lutong/codebases/xiuqi/VideoSiglip2-music

## train
bash scripts/train_video_siglip2_for_music.sh 2>&1 | tee LT/logs/training_$(date +%Y%m%d_%H%M%S)_41m_p${ARNOLD_ID}.log

bash scripts/train_video_siglip2_for_music_add_lang_top20.sh 2>&1 | tee LT/logs/training_lang_bm_61m_gt8_multi_lang_loss_$(date +%Y%m%d_%H%M%S)_p${ARNOLD_ID}.log

bash scripts/train_video_siglip2_for_music_add_lang_1029_dis_gate.sh 2>&1 | tee LT/logs/training_all_lang_60m_gt7_gate_rescale_1_$(date +%Y%m%d_%H%M%S)_p${ARNOLD_ID}.log

bash scripts/train_video_music_match.sh 2>&1 | tee LT/logs/0129_continue_80w_2w2_$(date +%Y%m%d_%H%M%S)_p${ARNOLD_ID}.log

bash scripts/train_video_music_match_original.sh 2>&1 | tee LT/logs/51m_original_$(date +%Y%m%d_%H%M%S)_p${ARNOLD_ID}.log

bash scripts/train_video_music_match_contrastive.sh  2>&1 | tee LT/logs/51m_constrative_$(date +%Y%m%d_%H%M%S)_p${ARNOLD_ID}.log

bash scripts/train_video_siglip2_for_music_add_lang_1029_dis.sh 2>&1 | tee LT/logs/training_1029_dis_61m_gt7_no_constr_$(date +%Y%m%d_%H%M%S)_p${ARNOLD_ID}.log

## generate music embedding -- distributed (tmux)
bash music_evaluation_tools_add_lang_country/run_encode_music_650w.sh 2>&1 | tee LT/logs/music_encode_650w.log

bash music_evaluation_tools/run_encode_music_650w.sh 2>&1 | tee LT/logs/music_encode_650w.log

## generate video embedding for test cases
bash music_evaluation_tools_add_lang_country/check_cases_top50.sh 2>&1 | tee LT/logs/check_cases_top50.log
bash music_evaluation_tools/check_cases_top50.sh | tee LT/logs/check_cases_top50.log

## recall
python3 -u music_evaluation_tools/calculate_recall.py 2>&1 | tee LT/logs/recall_v2m.log

python3 -u get_mid_info.py 2>&1 | tee LT/logs/get_mid_info.log

# eval top20
bash music_evaluation_tools_add_lang_country/get_eval_video_emb.sh 2>&1 | tee LT/logs/get_eval_video_emb.log
python3 -u music_evaluation_tools_add_lang_country/calculate_recall.py 2>&1 | tee LT/logs/get_top20.log

# eval auc v2m
bash run_evalset.sh  # 有自动存log
bash run_evalset_lang.sh 

## eval publish auc
bash music_evaluation_tools_add_lang_country/run_publish_eval.sh 2>&1 | tee LT/logs/eval_publish_auc.log

export CKPT_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_add_user_info_residual_41m/checkpoint-34000/"
python3 music_evaluation_tools_add_lang_country/encode_music_embedding.py --data_path "${CKPT_PATH}" --add_title_text False --add_lyrics_ue False --add_title_ue False --add_user_lang_and_country_method residual 2>&1 | tee LT/logs/generate_auc_embs.log


# tmux
tmux new -s train1
`ctrl+b` (once) then `d` # to detatch the tmux session
`exit` or `ctrl+d` # to  kill inside the tmux
tmux kill-session -t session_name # kill outside by name

# monitor gpu 
watch -n 1 --color nvidia-smi
watch -n 1 --color gpustat -cpu
gpustat # 彩色的


# see loss with TensorBoard --report_to tensorboard
pip install tensorboard

tensorboard --logdir /mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_music_clip_large_queue_pgc_pugc_100m_stage1 --port 6006

# Then open http://localhost:6006 in your browser 


## hdfs commands
hdfs dfs -put /mnt/bn/jiny-ttls-i18n-fr1q/lutong/codebases/MM-Embedding/LT/music_ids_viking.jsonl hdfs://harunava/home/byte_recommend_anote/data/relevance_model/fine_tune_train/
hdfs dfs -put /mnt/bn/jiny-ttls-i18n-fr1q/lutong/finetune_fix_dup_ckpt_4000_v2m_eval_results hdfs://harunava/home/byte_recommend_anote/data/relevance_model/llm_model/eval/20250923_4000_benchmark_raw
# overwrite
hdfs dfs -put /mnt/bn/jiny-ttls-i18n-fr1q/lutong/evals_top20/1029_jy94000/ hdfs://harunava/home/byte_recommend_anote/data/relevance_model/llm_model/eval/20251029_94000_benchmark_raw 
hdfs dfs -put /mnt/bn/jiny-ttls-i18n-fr1q/MM-Embedding/music_eval_video_data_17w.jsonl hdfs://harunava/user/wangxiuqi.0601/lutong/benchmark/music_eval_video_data_17w.jsonl

hdfs dfs -get hdfs://harunava/home/byte_recommend_anote/data/relevance_model/llm_model/llm_judge/20251030023358_benchmark.tsv /mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/evalset_all_llm_judge_data.tsv
hdfs dfs -get hdfs://harunava/user/wangxiuqi.0601/lutong/llm_judge/20251118061451_benchmark.tsv /mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/evalset_all_llm_judge_data_natural_popularity.tsv

hdfs dfs -get hdfs://harunava/user/wangxiuqi.0601/music/llm_judge_pretrain_data_pgc_1023_28m_va.json /mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/llm_judge_pretrain_data_pgc_1023_28m_va.json
hdfs dfs -get hdfs://harunava/user/wangxiuqi.0601/music/checkpoints/video_siglip2_discriminator_bm_0.1_lowq_7_residual_add_user_info/checkpoint-100000 /mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_discriminator_bm_0.1_lowq_7_residual_add_user_info/checkpoint-100000
hdfs dfs -get hdfs://harunava/user/wangxiuqi.0601/music/checkpoints/video_siglip2_discriminator_bm_0.1_lowq_7_real_residual_add_user_info/checkpoint-100000 /mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_discriminator_bm_0.1_lowq_7_real_residual_add_user_info/checkpoint-100000
hdfs dfs -get hdfs://harunava/user/wangxiuqi.0601/music/checkpoints/video_siglip2_discriminator_1029_0.08_lowq_7_gate_add_user_info/checkpoint-60000 /mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_discriminator_1029_0.08_lowq_7_gate_add_user_info/checkpoint-60000
hdfs dfs -get hdfs://harunava/home/byte_recommend_anote/data/relevance_model/llm_model/llm_judge/20251017063458_training.tsv /mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/
hdfs dfs -get hdfs://harunava/user/wangxiuqi.0601/music/online_update/model_v20251212/batch_input_features_with_ugc/20251213/with_ugc_data_all.json /mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/music/ugc_1213_data_100w.json
hdfs dfs -getmerge hdfs://harunava/user/wangxiuqi.0601/music/online_update/model_v20251222/batch_input_features_with_ugc/20251219/inputs/part* /mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/music/with_ugc_1219_data_150w.jsonl
hdfs dfs -get hdfs://harunava/home/byte_tiktok_music/proj/content_understanding/music/all_song_candidates_1216_7600w /mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/music/all_song_candidates_1216_7600w
hdfs dfs -get hdfs://harunava/user/wangxiuqi.0601/music/data/get_intersection_clip_id_text.csv /mnt/bn/jiny-ttls-i18n-fr1q/lutong/muisc_embs/20260101_1200w
hdfs dfs -get hdfs://harunava/home/byte_tiktok_music/proj/content_understanding/music/deploy/20260104_model/7d_20_30d_137_200w_music_ues  /mnt/bn/jiny-ttls-i18n-fr1q/lutong/muisc_embs/20260107_200w
hdfs dfs -get hdfs://harunava/user/wangxiuqi.0601/lutong/llm_judge/1029_positive_pred_all_llm_v3.jsonl auc_results/llm_v3/
hdfs dfs -get hdfs://harunava/user/wangxiuqi.0601/lutong/llm_judge/human_label_200_20260106/*.json 

hdfs dfs -put /mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/data/1029_negative_pred/predictions_all.jsonl  hdfs://harunava/user/wangxiuqi.0601/lutong/llm_judge/1029_negative_pred_all.jsonl
hdfs dfs -put /mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/data/1029_positive_pred/predictions_all.jsonl  hdfs://harunava/user/wangxiuqi.0601/lutong/llm_judge/1029_positive_pred_all.jsonl
hdfs dfs -put /mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_residual_60m_gt7_title_lang_loss_pgc_popularity/checkpoint-146000/deployment_100w_data/music_ue_vector.jsonl hdfs://harunava/user/wangxiuqi.0601/music/online_update/model_v20251212/batch_input_features_with_ugc/20251213/with_ugc_data_all_ue.json
hdfs dfs -put /mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_gate_60m_gt7_title_pgc_popularity/checkpoint-148000/650w_deploy/music_ue_vector.jsonl hdfs://harunava/user/wangxiuqi.0601/music/online_update/model_v20251212/batch_input_features_with_ugc/20251213/music_ue_vector.jsonl 
hdfs dfs -put /mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_gate_60m_gt7_title_pgc_popularity/checkpoint-148000/7600w_deploy/*rank*.jsonl hdfs://harunava/user/wangxiuqi.0601/music/online_update/model_v20251222/output_embs/7600w_whole/
hdfs dfs -put /mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_all_lang_new_gate_60m_gt7_rescale_05/checkpoint-158000/7600w_deploy/*rank*.jsonl hdfs://harunava/user/wangxiuqi.0601/music/online_update/model_v20260104/output_embs/7600w_whole
hdfs dfs -put /mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_gate_60m_gt7_title_pgc_popularity/checkpoint-148000/7600w_deploy_filtered/music_ue_vector.jsonl hdfs://harunava/user/wangxiuqi.0601/music/online_update/model_v20251222/output_embs/7600w_filtered_music_ue_vector.jsonl
hdfs dfs -put /mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/music/train_1216_1000w_cover_url.jsonl hdfs://harunava/home/byte_tiktok_music/proj/content_understanding/music/train_1216_1000w_cover_url.jsonl 

# merge multiple jsonl from hdfs to local 1 json file
hdfs dfs -getmerge hdfs://harunava/home/byte_recommend_anote/data/relevance_model/llm_model/video/video_frames_1030_1000/part-*.json /mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/evalset_1k_video_data.jsonl

hdfs dfs -getmerge hdfs://harunava/home/byte_recommend_anote/data/relevance_model/llm_model/music/music_caption_benchmark_1029_1w_all/part-*.json /mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/evalset_1w_music_data.jsonl

hdfs dfs -getmerge hdfs://harunava/home/byte_recommend_anote/data/relevance_model/fine_tune_train/music_clip_ids_1009_enriched_650w_all/part-*.json /mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/music_clip_ids_1009_enriched_650w_all.jsonl

hdfs dfs -getmerge hdfs://harunava/home/byte_tiktok_music/proj/content_understanding/music/training_1216_1000w/part-*.json /mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/music/music_train_1216_1000w.jsonl


hdfs dfs -getmerge hdfs://harunava/user/wangxiuqi.0601/lutong/music/music_clip_ids_1009_enriched_650w_final_dedup/part-*.json /mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/music_clip_ids_1009_enriched_650w_final_dedup.jsonl

hdfs dfs -getmerge hdfs://harunava/user/wangxiuqi.0601/lutong/music/music_caption_benchmark_1117_1w_natural_popularity_all/part-*.json /mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/evalset_1w_natural_popularity_music_data.jsonl


## 删除 ckpt -- preview
base="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_cont_from_0129_80w_regression"
find "$base" -type d -name 'checkpoint-*' -print0 \
| while IFS= read -r -d '' d; do
  n="${d##*-}"
  if [ "$n" -gt 40000 ] 2>/dev/null; then
    echo "$d"
  fi
done

## 删除 ckpt
base="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_cont_from_0129_80w_regression_2w2"
find "$base" -type d -name 'checkpoint-*' -print0 \
| while IFS= read -r -d '' d; do
  n="${d##*-}"
  if [ "$n" -gt 40000 ] 2>/dev/null; then
    rm -rf "$d"
  fi
done

## 从音乐数据里根据music id找到相关info
grep -R -h '"clip_id":6640463390428891909' /mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/data/all_song_candidates_200w --include="*.json"