#!/usr/bin/env bash
set -euo pipefail

# ===== 用户配置 =====
# MODEL_DIR="/mnt/bn/search-ad-creative-zhijie/wangxiuqi.0601/checkpoints/video_siglip2_for_music_baseline_retrain_positive_filter_discriminator_0.0_lowq_8_add_user_info"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_discriminator_bm_0.1_lowq_7_residual_add_user_info"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_discriminator_bm_0.1_lowq_7_real_residual_add_user_info"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_bm_filtering_41m_gt8"
MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_siglip2_for_music_baseline"
# CHECKPOINTS=(10000)
CHECKPOINTS=(94000)

RESULT_FILE="${MODEL_DIR}/eval_summary_$(date +%Y%m%d_%H%M%S).log"
# ====================

echo "开始批量评估: ${CHECKPOINTS[@]}"
echo "评估结果将保存到: $RESULT_FILE"
echo "====================================" | tee "$RESULT_FILE"

# 遍历每个 checkpoint
for step in "${CHECKPOINTS[@]}"; do
    CKPT_PATH="${MODEL_DIR}/checkpoint-${step}/"

    if [ ! -d "$CKPT_PATH" ]; then
        echo "⚠️  跳过: ${CKPT_PATH} 不存在" | tee -a "$RESULT_FILE"
        continue
    fi

    echo "" | tee -a "$RESULT_FILE"
    echo "======== 开始评估 checkpoint-${step} ========" | tee -a "$RESULT_FILE"
    echo "时间: $(date)" | tee -a "$RESULT_FILE"
    echo "路径: $CKPT_PATH" | tee -a "$RESULT_FILE"

    # 临时文件用于保存完整 stdout
    TMP_LOG=$(mktemp)
    echo "TMP_LOG path: ${TMP_LOG}" 

    {
        python3 music_evaluation_tools/encode_music_embedding.py --data_path "${CKPT_PATH}" --add_title_text False --add_lyrics_ue False --add_title_ue False --add_user_lang_and_country_code False
        python3 music_evaluation_tools_add_lang_country/calculate_recall.py --data_path "${CKPT_PATH}"
        python3 music_evaluation_tools_add_lang_country/llm_eval_metrics.py --data_path "${CKPT_PATH}"
    } &> "$TMP_LOG" || echo "❌ checkpoint-${step} 评估出现错误，请检查日志。" | tee -a "$RESULT_FILE"

    echo "--- checkpoint-${step} 输出的最后15行 ---" | tee -a "$RESULT_FILE"
    tail -n 15 "$TMP_LOG" | tee -a "$RESULT_FILE"
    echo "============================================" | tee -a "$RESULT_FILE"

    rm -f "$TMP_LOG"
done

echo "✅ 全部评估完成，结果保存在: $RESULT_FILE"
