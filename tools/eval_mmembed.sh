# DATA_PATH=/mnt/bn/yexiaoyu-test/data/lavis/msr-vtt
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 \
# main_task_retrieval.py --num_thread_reader=0 \
# --train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
# --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
# --data_path ${DATA_PATH}/MSRVTT_data.json \
# --dict_path ${DATA_PATH}/MSRVTT/msrvtt_gpt_test_captions.pickle \
# --features_path ${DATA_PATH}/MSRVTT/videos/all \
# --max_words 64 --max_frames 8 --batch_size_val 32 \
# --datatype msrvtt --expand_msrvtt_sentences  \
# --feature_framerate 1 \
# --output_dir /opt/tiger/MMPretrain/vis \
# --pretrained_model /mnt/bn/yexiaoyu-test/checkpoints/stage_2_mix_training_with_segment/checkpoint-170000

# DATA_PATH=/mnt/bn/yexiaoyu-test/data/lavis/didemo
# python3 main_task_retrieval_mmembed.py --num_thread_reader=2 \
# --data_path ${DATA_PATH}/didemo_anno \
# --features_path ${DATA_PATH}/videos \
# --output_dir /opt/tiger/MMPretrain/vis \
# --max_words 64 --max_frames 8 --batch_size_val 8 \
# --datatype didemo --feature_framerate 1 \
# --freeze_layer_num 1

DATA_PATH=/mnt/bn/yexiaoyu-test/data/lavis/activitynet_captions
python3 main_task_retrieval_mmembed.py --num_thread_reader=2 \
--data_path ${DATA_PATH} \
--dict_path ${DATA_PATH}/activitynet_test_captions.pickle \
--features_path ${DATA_PATH}/Activity_Videos \
--output_dir /opt/tiger/MMPretrain/vis \
--max_words 284 --max_frames 8 --batch_size_val 8 \
--datatype activity --feature_framerate 1 \
--freeze_layer_num 0  --slice_framepos 2