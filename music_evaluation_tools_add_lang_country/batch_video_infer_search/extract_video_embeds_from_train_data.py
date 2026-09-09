#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多卡并行推理脚本
"""

import os
import json
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import sys
import numpy as np
import csv
import datetime

# 导入数据集和模型
from dataset import VideoMusicHDFSDataset, create_mm_embed_dataloader
from video_clip import VideoSigLIP2ForMusicRegressionAddLangV4


def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return 0, 1, 0
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = os.environ.get('DIST_BACKEND', 'nccl')
        timeout_min = int(os.environ.get('DIST_TIMEOUT_MIN', '30'))
        dist.init_process_group(backend=backend, init_method='env://', timeout=datetime.timedelta(minutes=timeout_min))
        dist.barrier()
    else:
        print("WARNING: Distributed mode requires CUDA, but it's not available. Exiting...")
        sys.exit(1)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_model(args, device):
    """加载模型"""
    print(f"Loading model from {args.checkpoint_path}")

    model = VideoSigLIP2ForMusicRegressionAddLangV4(
        args.base_model_path,
        interpolate=args.interpolate,
        use_frame_mask=True,
        add_user_lang_and_country_code = True,
        user_info_fusion_method=args.user_info_fusion_method,
    )
    
    
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
        score_sample_rates=[0, 0.0, 0.0, 1.0, 1.0], # dropping 1,2,3
        force_rescale_prob=1.0 # rescale here!
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,  # 推理时保留最后一个batch
        collate_fn=lambda x: x if isinstance(x, list) else [x],
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        timeout=args.dl_timeout,
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_samples = 0
    chunk_embeds = []
    chunk_meta = []
    chunk_idx = 0
    if rank == 0:
        pbar = tqdm(desc="Inference", dynamic_ncols=True)
    for batch_data in dataloader:
        batch = collate_batch(batch_data)
        if batch is None:
            continue
        try:
            pixel_values = batch['pixel_values'].to(device)
            pixel_attention_mask = batch['pixel_attention_mask'].to(device)
            spatial_shapes = batch['spatial_shapes'].to(device)
            frame_mask = batch['frame_mask'].to(device)
            user_lang_code = batch['user_lang_code']
            user_country_code = batch['user_country_code']
            video_embeds = model.extract_video_embeds(
                pixel_values,
                pixel_attention_mask,
                spatial_shapes,
                frame_mask,
                user_lang_code,
                user_country_code,
            )
            scores = batch['video_music_match_score']
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().tolist()
            music_ids = batch['music_ids']
            item_ids = batch['item_ids']
            emb_np = video_embeds.detach().cpu().numpy()
            bsz = emb_np.shape[0]
            for i in range(bsz):
                chunk_embeds.append(emb_np[i])
                chunk_meta.append((str(music_ids[i]), str(item_ids[i]), int(scores[i])))
                total_samples += 1
                if rank == 0:
                    pbar.update(1)
            if len(chunk_embeds) >= args.write_chunk_size:
                emb_path = output_dir / f"embeddings_rank_{rank}_chunk_{chunk_idx}.npy"
                np.save(emb_path, np.stack(chunk_embeds, axis=0))
                meta_path = output_dir / f"metadata_rank_{rank}_chunk_{chunk_idx}.csv"
                with open(meta_path, 'w', newline='') as mf:
                    w = csv.writer(mf)
                    w.writerow(['idx','music_id','item_id','score'])
                    for j, (mid, iid, sc) in enumerate(chunk_meta):
                        w.writerow([j, mid, iid, sc])
                chunk_embeds = []
                chunk_meta = []
                chunk_idx += 1
        except Exception as e:
            print(f"[Rank {rank}] Error processing batch: {e}")
            continue
    if rank == 0:
        pbar.close()
    if len(chunk_embeds) > 0:
        emb_path = output_dir / f"embeddings_rank_{rank}_chunk_{chunk_idx}.npy"
        np.save(emb_path, np.stack(chunk_embeds, axis=0))
        meta_path = output_dir / f"metadata_rank_{rank}_chunk_{chunk_idx}.csv"
        with open(meta_path, 'w', newline='') as mf:
            w = csv.writer(mf)
            w.writerow(['idx','music_id','item_id','score'])
            for j, (mid, iid, sc) in enumerate(chunk_meta):
                w.writerow([j, mid, iid, sc])
        chunk_embeds = []
        chunk_meta = []
        chunk_idx += 1
    print(f"[Rank {rank}] Processed {total_samples} samples, saved chunks to {output_dir}")
    return total_samples




def main():
    parser = argparse.ArgumentParser(description='Video-Music Matching Inference')
    
    # 模型相关参数
    parser.add_argument('--base_model_path', type=str, required=True,
                        help='Base model path (siglip2-base)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Checkpoint path to load')
    parser.add_argument('--interpolate', type=int, default=6)
    parser.add_argument('--user_info_fusion_method', type=str, default='gate',
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
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--dl_timeout', type=float, default=0.0)
    
    # 输出相关参数
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for predictions')
    parser.add_argument('--flush_interval', type=int, default=100,
                        help='Flush interval for writing results')
    parser.add_argument('--write_chunk_size', type=int, default=1000)
    parser.add_argument('--disable_barrier', action='store_true')
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--dist_timeout_min', type=int, default=30)
    
    args = parser.parse_args()
    
    # 设置分布式环境
    os.environ['DIST_BACKEND'] = args.dist_backend
    os.environ['DIST_TIMEOUT_MIN'] = str(args.dist_timeout_min)
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
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
    inference(model, dataloader, args, rank, world_size)
    
    # 同步所有进程
    if dist.is_initialized() and not args.disable_barrier:
        dist.barrier()
    
    # 清理
    cleanup_distributed()
    
    if rank == 0:
        print("Inference completed!")


if __name__ == "__main__":
    main()
