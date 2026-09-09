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
from torch.utils.data import Dataset, SequentialSampler
from collections import OrderedDict
from einops import rearrange

import io
import base64
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import json
import jsonlines
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from trainer_misc import init_distributed_mode
from dataset import VideoHDFSDataset, RandomResizeCrop
from data_pipeline_utils import TiktokDataFusedPipeline
from video_clip import VideoMusicCLIP, VideoMusicTextCLIP
from config import cfg


data_pipeline = TiktokDataFusedPipeline()


class TikTokVideoDataset(Dataset):

    def __init__(self, anno_path : str, max_frames=5):
        super().__init__()

        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        image_h = 384
        image_w = 192

        annotation = []
        self.annotation = []

        if anno_path.endswith('.json') or anno_path.endswith('.jsonl'):
            with open(anno_path, 'r') as reader:
                for item in reader:
                    item = json.loads(item.strip())
                    annotation.append(item)
        else:
            with open(anno_path, 'r') as reader:
                for line in reader:
                    item_id = int(line.strip())
                    annotation.append({'item_id': item_id})

        self.transform = transforms.Compose([
            RandomResizeCrop(size=image_h, crop_w=image_w, crop_h=image_h, resize_long_edge=True, is_training=False),
            transforms.Normalize(mean=image_mean, std=image_std),
        ])
        self.max_frame_len = max_frames
        self.annotation = annotation

    def check_is_photo(self, video):
        frame_resolutions = [(frame.width, frame.height) for frame in video]
        if len(set(frame_resolutions)) > 1:
            return True
        else:
            return False

    def read_video_frames(self, video):
        video = [base64.b64decode(video[i]) for i in range(len(video))]
        video = [Image.open(io.BytesIO(image_str)).convert('RGB') for image_str in video]
        is_photo = self.check_is_photo(video)

        if is_photo:
            resize_height = 512
            resize_width = video[0].width * resize_height // video[0].height
            video = [frame.resize((resize_width, resize_height)) for frame in video]

        video_tensors = [torch.from_numpy(np.array(frame)).permute(2, 0, 1) for frame in video]
        frames_mask = [1 for _ in range(len(video_tensors))]
        video_tensors = torch.stack(video_tensors).float() / 255

        if len(video_tensors) < self.max_frame_len:
            frames_mask.extend([0 for _ in range(self.max_frame_len - len(video_tensors))])
            video_tensors = torch.cat([video_tensors, torch.zeros(self.max_frame_len - len(video_tensors), *video_tensors.shape[1:])])
            
        assert video_tensors.shape[0] == self.max_frame_len, f"The input video frame number does not equal to {self.max_frame_len}"
        
        video_tensors = self.transform(video_tensors)
        frames_mask = torch.LongTensor(frames_mask)

        return video_tensors, frames_mask

    def __getitem__(self, index):
        try:
            anno = self.annotation[index]
            item_id = anno['item_id']
            video_meta = data_pipeline.request_all_info(item_id, frame_number=self.max_frame_len)
            video_tensors, frames_mask = self.read_video_frames(video_meta['video'])
            
            res = {
                'frames': video_tensors,
                'frames_mask': frames_mask,
                'item_ids': item_id,
            }
            return res

        except Exception as e:
            return None

    def __len__(self):
        return len(self.annotation)


def get_args():
    parser = argparse.ArgumentParser('Pytorch Multi-process script', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--model_name', default='video_music_clip', choices=['video_music_clip', 'video_music_text_clip'], type=str)
    parser.add_argument('--dataset_type', default='batch_eval', choices=['batch_eval', 'custom_dataset'], type=str, help="The dataset type")
    parser.add_argument('--anno_dir', type=str, default='', help="The video annotation file")
    parser.add_argument('--model_dtype', default='bf16', type=str, help="The Model Dtype: bf16 or df16")
    parser.add_argument('--max_frames', default=8, type=int, help='number of max video frames')
    parser.add_argument('--cfg_path', default='', type=str, help='The config file path')
    parser.add_argument('--model_path', default='', type=str, help='The pre-trained weight path')
    parser.add_argument('--output_dir', type=str, default='',
                        help='path where to save, empty for no saving')
    return parser.parse_args()


def build_data_loader(args):
    def custom_collate(batch):
        sample_dict =  {
            'frames' : [], 
            'frames_mask': [], 
            'item_ids': [], 
        }
        for data_dict in batch:
            if data_dict is None:
                continue
            sample_dict['frames'].append(data_dict['frames'])
            sample_dict['frames_mask'].append(data_dict['frames_mask'])
            sample_dict['item_ids'].append(data_dict['item_ids'])

        sample_dict['frames'] = torch.stack(sample_dict['frames'], dim=0)
        sample_dict['frames_mask'] = torch.stack(sample_dict['frames_mask'], dim=0)
        return sample_dict


    if args.dataset_type == 'batch_eval':
        dataset = VideoHDFSDataset(
            args.anno_dir,
            args.rank,
            args.world_size,
            max_frame_len=args.max_frames,
            shuffle=False,
            repeat=False,
            is_training=False,
        )

        loader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
            pin_memory=True, drop_last=False, collate_fn=default_collate,
            prefetch_factor=4,
        )

    else:
        assert args.dataset_type == 'custom_dataset'
        dataset = TikTokVideoDataset(
            anno_path=args.anno_dir,
            max_frames=args.max_frames,
        )
        sampler = SequentialSampler(dataset)

        loader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=4, pin_memory=False, 
            sampler=sampler, shuffle=False, collate_fn=custom_collate, 
            drop_last=False, prefetch_factor=2,
        )

    return loader


def build_model(args):
    model_dtype = args.model_dtype
    model_path = args.model_path
    cfg_path = args.cfg_path

    if cfg_path:
        cfg.update_cfg(cfg_path)

    if args.model_name == 'video_music_clip':
        model = VideoMusicCLIP(
            config=cfg,
            gpuwise_nce=False,
        )
    else:
        assert args.model_name == 'video_music_text_clip'
        model = VideoMusicTextCLIP(
            config=cfg,
            gpuwise_nce=False,
        )

    if model_path:
        print(f"Loading the pre-trained checkpoint from {model_path}")
        pretrained_checkpoint = torch.load(model_path, map_location='cpu')
        load_res = model.load_state_dict(pretrained_checkpoint, strict=False)
        print(f"Loading result: {load_res}")

    return model


def main():
    args = get_args()
    
    init_distributed_mode(args)

    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda')
    rank = args.rank

    model = build_model(args)
    model.to(device)

    if args.model_dtype == "bf16":
        torch_dtype = torch.bfloat16 
    elif args.model_dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    data_loader = build_data_loader(args)
    torch.distributed.barrier()
    
    args.output_dir = os.path.join(args.output_dir, f"rank_{args.rank}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    cnt = 0

    for samples in tqdm(data_loader):
        video =  samples['frames'].to(device)
        video_mask = samples['frames_mask'].to(device)
        music_ue_vector = samples['music_ue_vector'] if 'music_ue_vector' in samples else None
        video_ids = samples['item_ids']
        music_ids = samples['music_ids'] if 'music_ids' in samples else None

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
            video_embed = model.extract_video_embeds(video, video_mask)
        
        video_embed = video_embed.cpu().clone()
        output_path = os.path.join(args.output_dir, f"video_embed_{cnt}.pt")
        
        if music_ue_vector is None:
            return_item = {
                'video_ids': video_ids,
                'video_embed': video_embed,
            }
        else:
            return_item = {
                'video_ids': video_ids,
                'music_ids': music_ids,
                'video_embed': video_embed,
                'music_embed': music_ue_vector,
            }
        torch.save(return_item, output_path)
        cnt += 1

    torch.distributed.barrier()


if __name__ == '__main__':
    main()