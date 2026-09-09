
for param in $(seq 74000 2000 76000); do
    echo "Running with checkpoint-$param..."
    CUDA_VISIBLE_DEVICES=0 python3 evaluate_video_retrieval.py \
        --model_type short \
        --pretrained_path "/mnt/bn/yexiaoyu-test/checkpoints/stage_1_final_no_frame_mask/checkpoint-$param" \
        --output_path "/mnt/bn/yexiaoyu-test/checkpoints/stage_1_final_no_frame_mask"
done
# --model_path "/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-so400m" \