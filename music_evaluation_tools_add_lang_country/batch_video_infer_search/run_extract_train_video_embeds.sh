#!/usr/bin/env bash
set -euo pipefail


# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/checkpoints/video_siglip2_for_music_only_match_huber_loss_delta_1_filter_music_ids/checkpoint-39200"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_llm_v5_filter_music_220w_continue_from_39200_unfreeze_text/checkpoint-40000"
MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_llm_v5_300w_continue_from_39200/checkpoint-40000"


# DATA_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/train_data_pgc_v5_score_20260116_50m.json"
# DATA_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/train_data_v5_score_20260220_220w.json"
DATA_PATH="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/train_data_v5_score_20260301_302w.json"

OUTPUT_DIR="${MODEL_DIR}/302w"

FLUSH_INTERVAL="${FLUSH_INTERVAL:-200}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

torchrun --nproc_per_node=8 music_evaluation_tools_add_lang_country/batch_video_infer_search/extract_video_embeds_from_train_data.py \
  --base_model_path "/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base" \
  --checkpoint_path "${MODEL_DIR}" \
  --data_path "${DATA_PATH}" \
  --tokenizer_dir /mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 256 \
  --num_workers 4 \
  --max_frame_len 5 \
  --user_info_fusion_method gate \
  --flush_interval "${FLUSH_INTERVAL}" \
  --embed_lang_caption \
  --add_title_text \
  --prefetch_factor 2 \
  --dl_timeout 6000 \
  --dist_backend gloo \
  --dist_timeout_min 60 \
  --disable_barrier


  # --filter_video_id_file "/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video/20260209_sample_by_tag_1w.jsonl" \
