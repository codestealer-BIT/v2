
MODEL_PATH="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"
# OUTPUT_DIR="/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_music_clip_dedup_v3_large_queue"
PRETRAINED_PATH="/mnt/bn/yexiaoyu-test/checkpoints/debug_fusion_without_decoder/checkpoint-100000"
FAISS_PATH="/mnt/bn/yexiaoyu-test/data/faiss"

BATCH_SIZE=32
NUM_FRAMES=8
# ANNO_FILE="hdfs://harunasg/home/byte_data_tt_m/jinyang.leo/music_rec_data/finetune_v1_data"
ANNO_FILE_1="hdfs://harunava/recommend/data/muse/jinyang.leo/video_pretrain_data_music/v1_200m_video_vv500_1m/data"
ANNO_FILE_2="hdfs://harunava/recommend/data/muse/jinyang.leo/video_pretrain_data_music/v1_200m_video_vv500_1m/data_remain"
FILE_CACHE="/mnt/bn/yexiaoyu-test/data/cache"

BERT_PATH="/mnt/bn/yexiaoyu-test/models/huggingface/bert_uncased"


python3 faiss_fusion_database.py \
--model_path $MODEL_PATH \
--bert_path $BERT_PATH \
--pretrained_path $PRETRAINED_PATH \
--batch_size $BATCH_SIZE \
--data_root $ANNO_FILE_1 $ANNO_FILE_2 \
--filelist_cache $FILE_CACHE \
--faiss_path $FAISS_PATH \

# --build_faiss \

