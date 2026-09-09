torchrun --nproc_per_node=8 music_evaluation_tools_add_lang_country/encode_video_embedding_MGR.py \
  --data_path /mnt/bn/jiny-ttls-i18n-fr1q/guyanhang/checkpoints/exp41_video_siglip2_for_music_only_match_64m_without_filter/checkpoint-220500 \
  --input_hdfs_path hdfs://harunava/home/byte_tiktok_music/proj/content_understanding/music/MGR/ft_data_filtered_capsule \
  --output_hdfs_path hdfs://harunava/user/wangxiuqi.0601/guyanhang/data/video_ue_from_capsule_image_b64_onlymatching \
  --create_time 20260518 \
  --batch_size 16 \
  --num_workers 8 \
  --max_frames 5 \
  --shuffle_files