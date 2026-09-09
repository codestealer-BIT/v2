import os
import torch
import sys
sys.path.append('/mnt/bn/jiny-ttls-i18n-fr1q/ttls_content')

import argparse
import datetime
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import IterableDataset
from collections import OrderedDict, defaultdict

import io
import base64
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import json
from tqdm import tqdm
import pyarrow.fs as pafs
from urllib.parse import urlparse
from torch.utils.data import DataLoader
from trainer_misc import init_distributed_mode
from data_pipeline_utils import TiktokDataFusedPipeline
from music_evaluation_tools_add_lang_country.encode_music_embedding import build_model
from dataset.hdfs_io import hopen, hmkdir, hlist_files
import bytedlogger
bytedlogger.config_default()

data_pipeline = TiktokDataFusedPipeline()


class TikTokVideoDataset(IterableDataset):

    def __init__(self, hdfs_input_path, rank, world_size, max_frames=5, processor=None, tokens_per_frame=32):
        super().__init__()
        self.fs, _ = pafs.FileSystem.from_uri("hdfs://harunava")
        path_without_scheme = urlparse(hdfs_input_path).path
        infos = self.fs.get_file_info(pafs.FileSelector(path_without_scheme, recursive=True))
        self.files = [info.path for info in infos if info.type == pafs.FileType.File and 'part-' in info.path]
        bytedlogger.logging.info(f"Totally {len(self.files)} files in {hdfs_input_path}")

        self.max_frame_len = max_frames
        self.tokens_per_frame = tokens_per_frame
        self.processor = processor
        self.rank = rank
        self.world_size = world_size
        self.generate_video_language_country_mapping()
    
    def generate_video_language_country_mapping(self):
        # method 1: use all langs and countries
        self.lang2idx = defaultdict(lambda: 0)        
        self.country2idx = defaultdict(lambda: 0)

        with open('/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/train_country2idx.jsonl', 'r') as fr:
            for line in fr:
                value = json.loads(line.strip())
                self.country2idx[value['key']] = value['idx']
        with open('/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/train_lang2idx.jsonl', 'r') as fr:
            for line in fr:
                value = json.loads(line.strip())
                self.lang2idx[value['key']] = value['idx']
        
        # # method 2: only use top20 lang and countries
        # lang2idx = {"None":0, "en": 1, "es": 2, "ar": 3, "pt": 4, "fr": 5, "ru": 6, "id": 7, "tr": 8, "th": 9, "vi": 10, "de": 11, "ja": 12, "it": 13, "uk": 14, "ro": 15, "pl": 16, "az": 17, "ko": 18, "ms": 19, "zh-Hant": 20}
        # self.lang2idx = defaultdict(lambda: lang2idx['None'], lang2idx)
        
        # country2idx = {'None': 0, 'US': 1, 'PK': 2, 'BR': 3, 'ID': 4, 'MX': 5, 'TR': 6, 'TH': 7, 'GB': 8, 'VN': 9, 'BD': 10, 'IQ': 11, 'DE': 12, 'FR': 13, 'NP': 14, 'MY': 15, 'SA': 16, 'IT': 17, 'JP': 18, 'UA': 19, 'ES': 20}
        # self.country2idx = defaultdict(lambda: country2idx['None'], country2idx)

        print(f"Totally {len(self.country2idx)} country2idx")
        print(f"Totally {len(self.lang2idx)} lang2idx")

    def read_video_frames(self, video):
        video = [base64.b64decode(video[i]) for i in range(len(video))]
        video = [Image.open(io.BytesIO(image_str)).convert('RGB') for image_str in video]
        # Removed legacy photo check and manual resizing; the SigLIP2 processor handles resizing internally

        inputs = self.processor(images=video, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        pixel_masks = inputs['pixel_attention_mask']
        spatial_shapes = inputs['spatial_shapes']

        now_len = pixel_values.shape[0]
        total_tokens = self.max_frame_len * self.tokens_per_frame
        valid_tokens = now_len * self.tokens_per_frame

        # Create frame-level mask over tokens
        frame_mask = torch.zeros((total_tokens), dtype=pixel_masks.dtype)
        frame_mask[:valid_tokens] = 1

        # Pad to max_frame_len (5) if necessary
        if now_len < self.max_frame_len:
            pad_frames = self.max_frame_len - now_len
            pixel_values = torch.cat([
                pixel_values,
                torch.zeros_like(pixel_values[0].unsqueeze(0)).repeat(pad_frames, *[1]*(pixel_values.dim()-1))], dim=0)
            pixel_masks = torch.cat([
                pixel_masks,
                torch.zeros_like(pixel_masks[0].unsqueeze(0)).repeat(pad_frames, *[1]*(pixel_masks.dim()-1))], dim=0)
            spatial_shapes = torch.cat([
                spatial_shapes,
                torch.ones_like(spatial_shapes[0].unsqueeze(0)).repeat(pad_frames, *[1]*(spatial_shapes.dim()-1)) * 16], dim=0)
        
        return pixel_values, pixel_masks, spatial_shapes, frame_mask

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        shard_count = self.world_size * num_workers
        shard_id = self.rank * num_workers + worker_id
        for idx, file_path in enumerate(self.files):
            if (idx % shard_count) != shard_id:
                continue
            try:
                with self.fs.open_input_stream(file_path) as stream:
                    fr = io.TextIOWrapper(io.BufferedReader(stream), encoding='utf-8')
                    for line in fr:
                        parts = line.strip().split('\t')
                        if len(parts) != 3:
                            continue
                        item_id = int(parts[0])
                        country_code = parts[1]
                        lang_code = parts[2]
                        try:
                            video_meta = data_pipeline.request_all_info(item_id, frame_number=self.max_frame_len)
                            pixel_values, pixel_masks, spatial_shapes, frame_mask = self.read_video_frames(video_meta['video'])
                            user_languages = self.lang2idx.get(lang_code, 0)
                            user_countries = self.country2idx.get(country_code, 0)
                            yield {
                                'pixel_values': pixel_values,
                                'pixel_masks': pixel_masks,
                                'spatial_shapes': spatial_shapes,
                                'frame_mask': frame_mask,
                                'item_ids': item_id,
                                'user_languages': user_languages,
                                'user_countries': user_countries,
                            }
                        except Exception:
                            pass
            except Exception:
                continue


def get_args():
    parser = argparse.ArgumentParser('Pytorch Multi-process script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--anno_dir', type=str, default='', help="The video annotation file")
    parser.add_argument('--model_dtype', default='bf16', type=str, help="The Model Dtype: bf16 or df16")
    parser.add_argument('--max_frames', default=8, type=int, help='number of max video frames')
    parser.add_argument('--model_path', default="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base", type=str, help='The pre-trained weight path')
    parser.add_argument('--input_hdfs_path', type=str, help='Input HDFS path \t seperated')
    parser.add_argument('--output_hdfs_path', type=str, help='Output HDFS directory to write per-rank JSONL')
    parser.add_argument('--create_time', type=str, required=True, help='Create time string, e.g., 20251112')
    parser.add_argument('--max_records_cache', type=int, default=20000)
    return parser.parse_args()


def build_data_loader(args, processor):
    def custom_collate(batch):
        sample_dict = {
            'pixel_values': [],
            'pixel_masks': [],
            'spatial_shapes': [],
            'frame_mask': [],
            'item_ids': [],
            'user_languages': [],
            'user_countries': [],
        }
        for data_dict in batch:
            if data_dict is None:
                continue
            sample_dict['pixel_values'].append(data_dict['pixel_values'])
            sample_dict['pixel_masks'].append(data_dict['pixel_masks'])
            sample_dict['spatial_shapes'].append(data_dict['spatial_shapes'])
            sample_dict['frame_mask'].append(data_dict['frame_mask'])
            sample_dict['item_ids'].append(data_dict['item_ids'])
            sample_dict['user_languages'].append(data_dict['user_languages'])
            sample_dict['user_countries'].append(data_dict['user_countries'])

        sample_dict['pixel_values'] = torch.stack(sample_dict['pixel_values'], dim=0)
        sample_dict['pixel_masks'] = torch.stack(sample_dict['pixel_masks'], dim=0)
        sample_dict['spatial_shapes'] = torch.stack(sample_dict['spatial_shapes'], dim=0)
        sample_dict['frame_mask'] = torch.stack(sample_dict['frame_mask'], dim=0)
        return sample_dict

    dataset = TikTokVideoDataset(
        hdfs_input_path=args.input_hdfs_path,
        max_frames=args.max_frames,
        processor=processor,
        tokens_per_frame=32,
        rank=args.rank,
        world_size=args.world_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=False,
        collate_fn=custom_collate,
        drop_last=False,
        prefetch_factor=4,
    )

    return loader



def write_results_to_hdfs(results, output_date_dir, is_hdfs, batch_id, rank):
    if not results:
        return
    part_name = f"rank_{rank}_batch_{batch_id:05d}.jsonl"
    part_path = os.path.join(output_date_dir, part_name)
    if is_hdfs:
        with pafs.open_output_stream(part_path) as stream:
            stream.write(("\n".join(results) + "\n").encode())
    else:
        with open(part_path, 'w') as fw:
            fw.write("\n".join(results) + "\n")

def main(data_path, input_hdfs_path, output_hdfs_path, create_time):
    args = get_args()
    args.input_hdfs_path = input_hdfs_path
    args.output_hdfs_path = output_hdfs_path
    args.create_time = create_time
    bytedlogger.logging.info(args)

    init_distributed_mode(args)

    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda')
    rank = args.rank
    ckpt_path =  os.path.join(data_path, "pytorch_model.bin")

    model, processor, tokenizer = build_model(args.model_path, ckpt_path, args)
    model = model.eval()
    device = torch.device('cuda')
    model = model.to(device)
    torch_dtype = torch.bfloat16 

    data_loader = build_data_loader(args, processor)
    torch.distributed.barrier()
    
    is_hdfs = args.output_hdfs_path.startswith('hdfs')
    output_date_dir = args.output_hdfs_path
    hmkdir(output_date_dir)
    buffer = []
    batch_id = 0
    for samples in tqdm(data_loader):
        video_ids = samples['item_ids']

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
            video_embed = model.extract_video_embeds(
                samples['pixel_values'].to(device),
                samples['pixel_masks'].to(device),
                samples['spatial_shapes'].to(device),
                samples['frame_mask'].to(device),
                samples['user_languages'],
                samples['user_countries'],
            )
        
        video_embed = video_embed.cpu().clone()
        for i in range(len(video_ids)):
            record = {
                'item_id': int(video_ids[i]),
                'video_ue_vector': video_embed[i].tolist(),
                'create_time': args.create_time,
            }
            buffer.append(json.dumps(record, ensure_ascii=False))
        if len(buffer) >= args.max_records_cache:
            write_results_to_hdfs(buffer, output_date_dir, is_hdfs, batch_id, args.rank)
            buffer = []
            batch_id += 1
    if buffer:
        write_results_to_hdfs(buffer, output_date_dir, is_hdfs, batch_id, args.rank)

    torch.distributed.barrier()


if __name__ == '__main__':
    create_time = "20251110"
    input_hdfs_path = f"hdfs://harunava/home/byte_tiktok_music/proj/content_understanding/video/video_ue_data/date={create_time}"
    output_hdfs_path = f"hdfs://harunava/home/byte_tiktok_music/proj/content_understanding/video/video_ue/date={create_time}"

    data_path = "/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_1029_dis_all_lang_new_gate_60m_gt7_rescale_05/checkpoint-158000"
    main(data_path, input_hdfs_path, output_hdfs_path, create_time)
