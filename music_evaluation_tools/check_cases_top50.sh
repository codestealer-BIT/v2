#!/usr/bin/env bash
set -euo pipefail

# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_bm_filtering_41m_gt8"
MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_siglip2_for_music_baseline"

CHECKPOINTS=(94000)

RESULT_FILE="${MODEL_DIR}/cases_summary_$(date +%Y%m%d_%H%M%S).log"
echo "Start batch evaluation: ${CHECKPOINTS[*]}" | tee "$RESULT_FILE"

for step in "${CHECKPOINTS[@]}"; do
  CKPT_PATH="${MODEL_DIR}/checkpoint-${step}/"
  if [ ! -d "$CKPT_PATH" ]; then
    echo "Skip: ${CKPT_PATH} not found" | tee -a "$RESULT_FILE"
    continue
  fi

  echo "======== Evaluate checkpoint-${step} ========" | tee -a "$RESULT_FILE"
  echo "Time: $(date)" | tee -a "$RESULT_FILE"
  echo "Path: $CKPT_PATH" | tee -a "$RESULT_FILE"

  TMP_LOG=$(mktemp)
  echo "TMP_LOG path: ${TMP_LOG}" | tee -a "$RESULT_FILE"

  {
    echo "======= Encode music embeddings =======" | tee -a "$RESULT_FILE"
    python3 music_evaluation_tools/encode_music_embedding.py \
      --data_path "${CKPT_PATH}" \
      --add_title_text False \
      --add_lyrics_ue False \
      --add_title_ue False

    echo "======= Calculate recall =======" | tee -a "$RESULT_FILE"
    output_csv="${MODEL_DIR}/cases_results/recall_results_step_${step}.csv"
    mkdir -p "$(dirname "$output_csv")"
    python3 music_evaluation_tools_add_lang_country/calculate_recall.py \
      --data_path "${CKPT_PATH}" \
      --output_csv_path "${output_csv}"

  } &> "$TMP_LOG" || echo "❌ checkpoint-${step} evaluation error, check logs." | tee -a "$RESULT_FILE"

  echo "--- Last 15 lines of output ---" | tee -a "$RESULT_FILE"
  tail -n 15 "$TMP_LOG" | tee -a "$RESULT_FILE"

  echo "Case top20 recall saved: ${MODEL_DIR}/cases_results/${output_csv}" | tee -a "$RESULT_FILE"

  rm -f "$TMP_LOG"
  echo "============================================" | tee -a "$RESULT_FILE"

done

echo "✅ All evaluations done. Summary: $RESULT_FILE"