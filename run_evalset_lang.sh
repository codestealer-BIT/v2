#!/usr/bin/env bash
set -euo pipefail

# ===== 用户配置 =====
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_add_user_info_gate_60m_gt7_title_pgc_popularity"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_all_lang_new_gate_60m_gt7_rescale_1"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_all_lang_new_gate_60m_gt7_rescale_05"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_cont_from_0129_80w_regression"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_cont_from_0129_80w_regression_p45_n1"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_cont_from_0129_80w_regression_2w2"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_50m_regression_scale_5_1000w_music_all"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_51m"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_contrastive_51m"
# MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/guyanhang/checkpoints/video_siglip2_for_music_only_match_54m"
MODEL_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/guyanhang/checkpoints/exp41_video_siglip2_for_music_only_match_64m_without_filter"
# CHECKPOINTS=(80000 148000 158000 160000)
# CHECKPOINTS=(148000 158000)
CHECKPOINTS=(29400 34300 39200 44100)
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
        python3 music_evaluation_tools_add_lang_country/encode_music_embedding.py --data_path "${CKPT_PATH}" --embed_lang_caption True --add_title_text True --add_lyrics_ue False --add_title_ue False --add_user_lang_and_country_method gate
        python3 music_evaluation_tools_add_lang_country/calculate_recall.py --data_path "${CKPT_PATH}"
        python3 music_evaluation_tools_add_lang_country/llm_eval_metrics.py --data_path "${CKPT_PATH}"
    } &> "$TMP_LOG" || echo "❌ checkpoint-${step} 评估出现错误，请检查日志。" | tee -a "$RESULT_FILE"

    echo "--- checkpoint-${step} 输出的最后40行 ---" | tee -a "$RESULT_FILE"
    tail -n 40 "$TMP_LOG" | tee -a "$RESULT_FILE"
    echo "============================================" | tee -a "$RESULT_FILE"

    rm -f "$TMP_LOG"
done

echo "✅ 全部评估完成，结果保存在: $RESULT_FILE"
