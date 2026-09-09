#!/usr/bin/env bash
set -euo pipefail

# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_add_user_info_residual_41m"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_add_user_info_residual_41m_gt8_top20lang"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_add_user_info_residual_41m_gt8_top20lang_maxa"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_discriminator_1029_0.08_lowq_7_gate_add_user_info"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_residual_41m_gt8_maxa"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_residual_61m_gt8_title"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_residual_61m_gt8_title_lang_loss"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_residual_60m_gt7_title_lang_loss_pgc_popularity"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_gate_60m_gt7_title_pgc_popularity"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/checkpoints/video_siglip2_for_music_only_match_huber_loss_delta_1_add_lang"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/checkpoints/video_siglip2_for_music_only_match_huber_loss_delta_1_filter_music_ids"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_cont_from_0129_80w_regression_p45_n1"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_cont_from_0129_80w_regression_2w2"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_51m"
MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_contrastive_51m"

CHECKPOINTS=(73500)
# LANG_COUNTRY_PAIRS=("0:0" "24:1" "1:1" "9:10" "3:30" "22:56") # None:None, zh-Hant:US, en:US, th:TH, ar:AR, ko:KR     # lang:country
# LANG_COUNTRY_PAIRS=("0:0" "20:1" "1:1" "9:7" "3:0" "18:0") # None:None, zh-Hant:US, en:US, th:TH, ar:None, ko:None     # lang:country top20 mapping
# LANG_COUNTRY_PAIRS=("1:1" "1:3" "6:3" "2:9" "10:11" "9:10" "1:24" "1:15" ) # en:US, en:ID, id:ID, es:MX, vi:VN, th:TH, en:PH, en:MY # lang:country all mapping
# LANG_COUNTRY_PAIRS=("1:27" "0:0" "30:27" ) # en:JP, None:None, zh:JP
# LANG_COUNTRY_PAIRS=("1:27" "30:27" "13:27" "1:1" "1:3" "6:3" "2:9" "10:11" "9:10" "1:24" "1:15" ) # en:JP, zh:JP,ja:JP, en:US, en:ID, id:ID, es:MX, vi:VN, th:TH, en:PH, en:MY 
LANG_COUNTRY_PAIRS=("1:1") #  en:US

RESULT_FILE="${MODEL_DIR}/cases_summary_$(date +%Y%m%d_%H%M%S).log"
echo "Start batch evaluation: ${CHECKPOINTS[*]}" | tee "$RESULT_FILE"
echo "Lang/Country pairs: ${LANG_COUNTRY_PAIRS[*]}" | tee -a "$RESULT_FILE"

for step in "${CHECKPOINTS[@]}"; do
  CKPT_PATH="${MODEL_DIR}/checkpoint-${step}/"
  if [ ! -d "$CKPT_PATH" ]; then
    echo "Skip: ${CKPT_PATH} not found" | tee -a "$RESULT_FILE"
    continue
  fi

  for pair in "${LANG_COUNTRY_PAIRS[@]}"; do
    lang_code="${pair%%:*}"
    country_code="${pair##*:}"

    echo "======== Evaluate checkpoint-${step} lang=${lang_code} country=${country_code} ========" | tee -a "$RESULT_FILE"
    echo "Time: $(date)" | tee -a "$RESULT_FILE"
    echo "Path: $CKPT_PATH" | tee -a "$RESULT_FILE"

    TMP_LOG=$(mktemp)
    echo "TMP_LOG path: ${TMP_LOG}" | tee -a "$RESULT_FILE"

    {
      echo "======= Encode video embeddings =======" | tee -a "$RESULT_FILE"
      python3 music_evaluation_tools_add_lang_country/encode_music_embedding.py \
        --data_path "${CKPT_PATH}" \
        --add_title_text True \
        --add_lyrics_ue False \
        --add_title_ue False \
        --add_user_lang_and_country_method gate \
        --user_language_code "${lang_code}" \
        --embed_lang_caption True \
        --user_country_code "${country_code}"

      echo "======= Calculate recall =======" | tee -a "$RESULT_FILE"
      output_csv="${MODEL_DIR}/cases_results/recall_results_step_${step}_lang_${lang_code}_country_${country_code}.csv"
      mkdir -p "$(dirname "$output_csv")"
      python3 music_evaluation_tools_add_lang_country/calculate_recall.py \
        --data_path "${CKPT_PATH}" \
        --output_csv_path "${output_csv}"

    } &> "$TMP_LOG" || echo "❌ checkpoint-${step} lang=${lang_code} country=${country_code} evaluation error, check logs." | tee -a "$RESULT_FILE"

    echo "--- Last 15 lines of output ---" | tee -a "$RESULT_FILE"
    tail -n 15 "$TMP_LOG" | tee -a "$RESULT_FILE"

    echo "Case top20 recall saved: ${MODEL_DIR}/cases_results/recall_results_step_${step}_lang_${lang_code}_country_${country_code}.csv" | tee -a "$RESULT_FILE"

    rm -f "$TMP_LOG"
    echo "============================================" | tee -a "$RESULT_FILE"
  done
done

echo "✅ All evaluations done. Summary: $RESULT_FILE"