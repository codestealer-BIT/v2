#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多卡并行推理脚本
使用方法: torchrun --nproc_per_node=8 infer.py --config config.yaml
"""

import os
import json
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# 导入数据集和模型
from dataset import VideoMusicHDFSDataset, create_mm_embed_dataloader
from video_clip import VideoSigLIP2ForMusicNegativeFilteringAddLang, VideoSigLIP2ForMusicNegativeFilteringBMAddLang, VideoSigLIP2ForMusic


def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_model(args, device):
    """加载模型"""
    print(f"Loading model from {args.checkpoint_path}")

    model = VideoSigLIP2ForMusicNegativeFilteringAddLang(
        model_path=args.base_model_path,
        gpuwise_nce=False,  # 推理时不需要
        interpolate=args.interpolate,
        use_frame_mask=True,
        # add_lyrics_ue=args.add_lyrics_ue,
        # add_title_ue=args.add_title_ue,
        user_info_fusion_method=args.user_info_fusion_method,
        language_add_constant=0.0,
    )
    
    # model = VideoSigLIP2ForMusicNegativeFilteringBMAddLang(
    #     model_path=args.base_model_path,
    #     gpuwise_nce=False,  # 推理时不需要
    #     queue_size=0,
    #     interpolate=args.interpolate,
    #     mean_loss=False,
    #     use_frame_mask=True,
    #     low_quality_score=4,
    #     historical_model_path=None,
    #     user_info_fusion_method=args.user_info_fusion_method,
    # )

    # model = VideoSigLIP2ForMusic(
    #     model_path=args.base_model_path,
    #     gpuwise_nce=False,  # 推理时不需要
    #     queue_size=0,
    #     interpolate=args.interpolate,
    #     mean_loss=False,
    #     use_frame_mask=True,
    #     low_quality_score=4
    # )
    
    # 加载checkpoint
    if args.checkpoint_path:
        pretrained_ckpt_path = os.path.join(args.checkpoint_path, 'pytorch_model.bin')
        print(f"Loading checkpoint from {args.checkpoint_path}")
        state_dict = torch.load(pretrained_ckpt_path, map_location='cpu', weights_only=True)
        load_res = model.load_state_dict(state_dict, strict=True)
        print(f"Loading result: {load_res}")

    model.to(device)
    model.eval()
    
    return model


def create_dataloader(args, rank, world_size):
    """创建数据加载器"""
    dataset = VideoMusicHDFSDataset(
        data_path=args.data_path,
        rank=rank,
        world_size=world_size,
        shuffle=False,  # 推理时不打乱
        repeat=False,   # 推理时不重复
        verbose=True,
        buffer_size=args.buffer_size,
        max_frame_len=args.max_frame_len,
        max_text_len=args.max_text_len,
        max_short_len=args.max_short_len,
        tokens_per_frame=args.tokens_per_frame,
        random_drop_frame=False,  # 推理时不随机丢帧
        random_rotate_video=False,  # 推理时不随机旋转
        filter_video_id_file=args.filter_video_id_file,
        tokenizer_dir=args.tokenizer_dir,
        add_title_text=args.add_title_text,
        embed_lang_caption=args.embed_lang_caption,
        force_rescale_prob=1.0, # rescale here!
        enable_llm_v5=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,  # 推理时保留最后一个batch
        collate_fn=lambda x: x if isinstance(x, list) else [x],
    )
    
    return dataloader


def collate_batch(batch):
    """处理batch数据"""
    if not batch:
        return None
    
    # 过滤掉None样本
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # 整理成字典格式
    collated = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]
    
    return collated


@torch.no_grad()
def inference(model, dataloader, args, rank, world_size):
    """执行推理"""
    device = next(model.parameters()).device
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 每个rank写自己的输出文件
    output_file = output_dir / f"predictions_rank_{rank}.jsonl"
    
    results = []
    total_samples = 0
    
    # 使用tqdm显示进度（只在rank 0显示）
    if rank == 0:
        pbar = tqdm(desc="Inference", dynamic_ncols=True)
    
    with open(output_file, 'w') as f:
        for batch_data in dataloader:
            # 处理batch
            batch = collate_batch(batch_data)
            if batch is None:
                continue
            
            try:
                # 将数据移到GPU
                pixel_values = batch['pixel_values'].to(device)
                pixel_attention_mask = batch['pixel_attention_mask'].to(device)
                spatial_shapes = batch['spatial_shapes'].to(device)
                frame_mask = batch['frame_mask'].to(device)
                
                music_content_embedding = batch['music_content_ue_vector'].to(device)
                music_cover_embedding = batch['music_cover_ue_vector'].to(device)
                music_attribute_input_ids = batch['music_attribute_input_ids'].to(device)
                music_caption_input_ids = batch['music_caption_input_ids'].to(device)
                music_dist_code = batch['music_dist_code']
                
                user_lang_code = batch['user_lang_code']
                user_country_code = batch['user_country_code']
                
                # 提取视频embedding
                video_embeds = model.extract_video_embeds(
                    pixel_values=pixel_values,
                    pixel_attention_mask=pixel_attention_mask,
                    spatial_shapes=spatial_shapes,
                    frame_mask=frame_mask,
                    user_lang_code=user_lang_code,
                    user_country_code=user_country_code,
                )
                
                # 提取音乐embedding
                music_embeds = model.extract_music_embeds(
                    music_content_embedding=music_content_embedding,
                    music_cover_embedding=music_cover_embedding,
                    music_attribute_input_ids=music_attribute_input_ids,
                    music_caption_input_ids=music_caption_input_ids,
                    music_dist_code=music_dist_code,
                )
                
                # 计算相似度分数（内积）
                similarity_scores = torch.sum(video_embeds * music_embeds, dim=-1)
                similarity_scores = similarity_scores.cpu().numpy()
                
                # 构造输出结果
                batch_size = len(batch['music_ids'])
                for i in range(batch_size):
                    result = {
                        'music_id': batch['music_ids'][i],
                        'item_id': batch['item_ids'][i],
                        'predict_score': float(similarity_scores[i]),
                        'video_music_match_score': int(batch['video_music_match_score'][i]),
                    }
                    
                    # 写入文件
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    total_samples += 1
                    
                    if rank == 0:
                        pbar.update(1)
                
                # 定期flush
                if total_samples % args.flush_interval == 0:
                    f.flush()
                    
            except Exception as e:
                print(f"[Rank {rank}] Error processing batch: {e}")
                continue
    
    if rank == 0:
        pbar.close()
    
    print(f"[Rank {rank}] Processed {total_samples} samples, saved to {output_file}")
    
    return total_samples


def merge_results(args, world_size):
    """合并所有rank的结果（可选）"""
    output_dir = Path(args.output_dir)
    
    if args.merge_output:
        print("Merging results from all ranks...")
        merged_file = output_dir / "predictions_all.jsonl"
        
        with open(merged_file, 'w') as f_out:
            for rank in range(world_size):
                rank_file = output_dir / f"predictions_rank_{rank}.jsonl"
                if rank_file.exists():
                    with open(rank_file, 'r') as f_in:
                        for line in f_in:
                            f_out.write(line)
        
        print(f"Merged results saved to {merged_file}")

def main():
    parser = argparse.ArgumentParser(description='Video-Music Matching Inference')
    
    # 模型相关参数
    parser.add_argument('--base_model_path', type=str, required=True,
                        help='Base model path (siglip2-base)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Checkpoint path to load')
    parser.add_argument('--interpolate', type=int, default=6)
    parser.add_argument('--user_info_fusion_method', type=str, default='residual',
                        choices=['gate', 'film', 'residual'])
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data annotation json file')
    parser.add_argument('--tokenizer_dir', type=str, required=True,
                        help='Tokenizer directory')
    parser.add_argument('--filter_video_id_file', type=str, default=None)
    parser.add_argument('--add_title_text', action='store_true')
    parser.add_argument('--embed_lang_caption', action='store_true')
    
    # 数据加载参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--buffer_size', type=int, default=-1)
    parser.add_argument('--max_frame_len', type=int, default=8)
    parser.add_argument('--max_text_len', type=int, default=284)
    parser.add_argument('--max_short_len', type=int, default=64)
    parser.add_argument('--tokens_per_frame', type=int, default=32)
    
    # 输出相关参数
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for predictions')
    parser.add_argument('--merge_output', action='store_true',
                        help='Merge outputs from all ranks into one file')
    parser.add_argument('--flush_interval', type=int, default=100,
                        help='Flush interval for writing results')
    
    args = parser.parse_args()
    
    # 设置分布式环境
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        print(f"Starting inference with {world_size} GPUs")
        print(f"Arguments: {args}")
    
    # 加载模型
    model = load_model(args, device)
    
    # 创建数据加载器
    dataloader = create_dataloader(args, rank, world_size)
    
    # 执行推理
    total_samples = inference(model, dataloader, args, rank, world_size)
    
    # 同步所有进程
    if dist.is_initialized():
        dist.barrier()
    
    # 合并结果（只在rank 0执行）
    if rank == 0 and args.merge_output:
        merge_results(args, world_size)
    
    # 清理
    cleanup_distributed()
    
    if rank == 0:
        print("Inference completed!")


if __name__ == "__main__":
    main()