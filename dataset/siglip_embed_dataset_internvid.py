import os
import json
import jsonlines
import torch
import math
import time
import random
import cv2
import io
from typing import List, Any
from tqdm import tqdm
from collections import OrderedDict
import base64
import pickle

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from functools import lru_cache

import numpy as np
import subprocess
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, IterableDataset, DataLoader, RandomSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as F

from transformers import AutoTokenizer, AutoProcessor


try:
    from .siglip2_preprocessor import Siglip2ImageProcessorFast
except:
    from siglip2_preprocessor import Siglip2ImageProcessorFast




from IPython import embed

import re

def clean_caption(caption: str) -> str:
    """
    Removes prefixes like '[xxx:] The|This video|image <first word>' 
    from a caption.
    """
    pattern = r'^\s*(?:[^:]+:\s*)?(?:the|this)\s+(?:video|image)\s+\w+\s+'
    return re.sub(pattern, '', caption, flags=re.IGNORECASE)





class VideoDatasetInternVid(IterableDataset):
    def __init__(
            self, 
            data_path,
            rank: int = 0,
            world_size: int = 1,
            shuffle: bool = False,
            repeat: bool = False,
            verbose: bool = False,
            buffer_size: int = -1,
            max_frame_len: int = 8,
            max_text_len:int=196,#long caption
            max_short_len:int=64,#short caption
            tokens_per_frame: int=32,
            image_w: int = 192,
            image_h: int = 384,
            image_query_num: int = 256,
            image_mean: list = [0.485, 0.456, 0.406],
            image_std: list = [0.229, 0.224, 0.225],
            is_training: bool = True,
            tokenizer_dir=None,
            blip_tokenizer_dir=None,
            stage: int = 1,
        ):
        super().__init__()
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.stage = stage

        self.files = []

        tic = time.time()
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.files.append(line)
        toc = time.time()

        print(f"Load {len(self.files)} annotations from {data_path}, time cost: {toc-tic}")
        
        self.verbose = verbose
        self.repeat = repeat
        self.buffer = []
        self.buffer_size = buffer_size

        self.max_frame_len = max_frame_len
        self.max_text_len = max_text_len
        self.max_short_len = max_short_len
        self.tokens_per_frame = tokens_per_frame
        
        self.segment_map = {
            'caption': 0,
            'title': 0,
            'sticker': 1,
            'ocr': 2,
            'asr': 3
        }

        #processor for auto processor
        processor = AutoProcessor.from_pretrained(blip_tokenizer_dir)
        self.blip_tokenizer = processor.tokenizer
        self.image_query_num = image_query_num
        self.image_token = processor.image_token

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) 
        self.processor = Siglip2ImageProcessorFast()
        
        self.image_mean = image_mean
        self.image_std = image_std

    def split_shard(self, data: List[Any], shard_idx: int, shard_size: int):
        num = len(data)
        if num < shard_size:
            raise RuntimeError("num:{} < shard size:{}".format(num, shard_size))
        start_idx = (num * shard_idx) // shard_size
        end_idx = (num * (shard_idx + 1)) // shard_size
        return data[start_idx: end_idx]

    def generate(self, seed=711):
        if self.shuffle:
            self.files = self.sort_and_shuffle(self.files, seed)
        else:
            self.files.sort()

        if self.world_size == 1 or len(self.files) == 1:
            cur_dataloader_files = self.files
        else:
            cur_dataloader_files = self.split_shard(self.files, self.rank, self.world_size)
        
        while True:
            worker_info = torch.utils.data.get_worker_info()

            if worker_info is not None:
                if len(cur_dataloader_files) % worker_info.num_workers != 0 and self.verbose:
                    print('[DATA]--current dataloader [%s] file num %s cannot split to worker_num %s ' %
                             (self.rank, len(cur_dataloader_files), worker_info.num_workers))
                cur_worker_files = self.split_shard(cur_dataloader_files, worker_info.id, worker_info.num_workers)
            else:
                cur_worker_files = cur_dataloader_files

            if self.shuffle:
                random.shuffle(cur_worker_files)
    
            if self.verbose:
                print(
                    f"[DataLoader] --> Rank:[{self.rank}]  Workers:[{worker_info.id if worker_info else 0}] process file: {len(cur_worker_files)} :{self.get_surfix(cur_worker_files[:3])}  ..."
                )

            for line in cur_worker_files: # each line is a json string now, no need to read it
                try:
                    if self.buffer_size <= 0:
                        yield json.loads(line)
                    elif len(self.buffer) < self.buffer_size:
                        self.buffer.append(json.loads(line))
                    else:
                        yield self.enq_deq_buffer(json.loads(line))
                except Exception as e:
                    print('encounter broken file: %s %s' % (e, line))

            if self.buffer:
                if self.shuffle:
                    random.shuffle(self.buffer)
                for _ in range(len(self.buffer)):
                    yield self.buffer.pop()

            if not self.repeat:
                break

    def enq_deq_buffer(self, sample):
        pop_sample = self.buffer.pop(random.randrange(self.buffer_size)) if self.shuffle else self.buffer.pop(0)  # 注意是空的情况
        self.buffer.append(sample)
        return pop_sample

    def check_is_photo(self, video):
        frame_resolutions = [(frame.width, frame.height) for frame in video]
        if len(set(frame_resolutions)) > 1:
            return True
        else:
            return 
    

    def process_image(self, images):
        
        inputs = self.processor(images=images, return_tensors="pt")

        pixel_values = inputs['pixel_values']
        pixel_masks = inputs['pixel_attention_mask']
        spatial_shapes = inputs['spatial_shapes']

        now_len = pixel_values.shape[0]

        total_tokens = self.max_frame_len * self.tokens_per_frame
    
        # number of valid tokens
        valid_tokens = now_len * self.tokens_per_frame

        # create mask
        frame_mask = torch.zeros((total_tokens), dtype=pixel_masks.dtype)
        frame_mask[:valid_tokens] = 1

        if now_len < self.max_frame_len:
            pixel_values = torch.cat([pixel_values, torch.zeros_like(pixel_values[0].unsqueeze(0)).repeat(self.max_frame_len - now_len, *[1]*(pixel_values.dim()-1))], dim=0)
            pixel_masks = torch.cat([pixel_masks, torch.zeros_like(pixel_masks[0].unsqueeze(0)).repeat(self.max_frame_len - now_len, *[1]*(pixel_masks.dim()-1))], dim=0)
            spatial_shapes = torch.cat([spatial_shapes, torch.ones_like(spatial_shapes[0].unsqueeze(0)).repeat(self.max_frame_len - now_len, *[1]*(pixel_masks.dim()-1)) * 16], dim=0)
        
        
        return pixel_values,pixel_masks,spatial_shapes, frame_mask
    
    # def generate_mask_and_complement(self, attention_mask):
    #     # first, there are padding frames (for image sequence with nums < max_frame_len),filter them out
    #     non_zero_mask = attention_mask.sum(dim=1) > 0  
    #     valid_indices = torch.where(non_zero_mask)[0]  

    #     # get the mask intersection to avoid predicting the padding tokens
    #     valid_mask = attention_mask[valid_indices]
    #     intersection = torch.all(valid_mask == 1, dim=0).int()
    #     ones_indices = torch.where(intersection == 1)[0]
        
    #     # if there are no intersections (i.e., there are not enough frames)
    #     if len(ones_indices) == 0:
    #         raise ValueError("There are not enough frames/non-padding tokens.")
        
    #     # for every frame, we take the same 50% of the patches to mask
    #     shuffled_indices = ones_indices[torch.randperm(len(ones_indices))]
    #     half_size = len(ones_indices) // 2
    #     selected_indices = shuffled_indices[:half_size]
        
    #     # prediction mask
    #     new_mask_base = torch.zeros_like(intersection)
    #     new_mask_base[selected_indices] = 1
        
    #     # to(F, seq_len) where every frame uses the same spatial mask
    #     target_mask = torch.zeros_like(attention_mask)
    #     target_mask[valid_indices] = new_mask_base.unsqueeze(0).repeat(valid_mask.shape[0], 1)
        
    #     # context mask but the padding must be excluded because student encoder uses it
    #     context_mask = 1 - target_mask
    #     context_mask = torch.where(attention_mask == 1, context_mask, 0)
        
    #     return target_mask, context_mask
    

        
    

    def read_video_frames(self, video):
        if len(video) > self.max_frame_len:
            sample_idx = torch.linspace(0, len(video) - 1, self.max_frame_len).round().long().tolist()
            video = [video[i] for i in sample_idx]

        # video = [base64.b64decode(video[i]) for i in range(len(video))]
        # video = [Image.open(io.BytesIO(image_str)).convert('RGB') for image_str in video]
        # is_photo = self.check_is_photo(video)#意思是有多种不同的分辨率

        # video = [video[0]]

        video_tensors,pixel_masks,spatial_shapes,frame_mask = self.process_image(video)

        assert video_tensors.shape[0] == self.max_frame_len, f"The input video frame number does not equal to {self.max_frame_len}"
        
        return video_tensors, pixel_masks, spatial_shapes,frame_mask

    def process_text(self, short_caption, long_caption): 

        short_input_ids = self.tokenizer(short_caption, padding='max_length', truncation=True, max_length=self.max_short_len, return_attention_mask = True, return_tensors="pt").input_ids[0]
        long_input_ids = self.tokenizer(long_caption, padding='max_length', truncation=True, max_length=self.max_text_len, return_attention_mask = True, return_tensors="pt").input_ids[0]


        return short_input_ids, long_input_ids
    
    def get_video_frames(self, video_path):
        try:
            video_capture = cv2.VideoCapture(video_path)
            frames = []

            while True:
                flag, frame = video_capture.read()
                if not flag:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            video_capture.release()

            if len(frames) > self.max_frame_len:
                sample_idx = torch.randperm(len(frames) - 1)[:self.max_frame_len].tolist()
                sample_idx.sort()
                #sample_idx = torch.linspace(0, len(frames) - 1, self.max_frame_len).round().long().tolist()
                frames = [frames[i] for i in sample_idx]

            frames = [Image.fromarray(frames[i]).convert("RGB") for i in range(len(frames))]

            return frames

        except Exception as e:
            print(f"Load video: {video_path} Error, Exception {e}")
            return None

    def __iter__(self):
        for data_item in self.generate():
            try:
                item_id = data_item['YoutubeID']
                video_path = data_item['file_path']
                # process video data
                # print(video_path)
                video_frames = self.get_video_frames(video_path)
                pixel_values,attention_mask,spatial_shapes,frame_mask = self.read_video_frames(video_frames)

                if frame_mask.sum() == 0:
                    print('encounter all padding frames in video %s' % (item_id))
                    continue
                short_caption = data_item['Caption']

                # target_mask, context_mask = self.generate_mask_and_complement(attention_mask)
                # if 'long_caption' not in data_item.keys():
                #     continue
                long_caption = data_item['long_caption']

                if long_caption == '' or short_caption == '':
                    continue
                short_input_ids, long_input_ids = self.process_text(short_caption = short_caption, long_caption = long_caption)

                if (short_input_ids >= self.tokenizer.vocab_size).any() or (long_input_ids >= self.tokenizer.vocab_size).any():
                    print('encounter bad data in tokenizing %s' % (item_id))
                    continue

                if self.stage == 1:
                    res = {
                        'item_ids': str(item_id), 
                        'pixel_values': pixel_values, 
                        'pixel_attention_mask': attention_mask, 
                        'spatial_shapes':spatial_shapes,

                        'frame_mask': frame_mask,

                        'short_input_ids':short_input_ids,# for video-caption ret
                        'long_input_ids':long_input_ids,# for video-caption ret

                        # 'target_mask': target_mask, # for masked video modeling
                        # 'context_mask': context_mask, # for masked video modeling
                    }
                else:
                    res = {
                        'internvid_item_ids': str(item_id), 
                        'internvid_pixel_values': pixel_values, 
                        'internvid_pixel_attention_mask': attention_mask, 
                        'internvid_spatial_shapes':spatial_shapes,

                        'internvid_frame_mask': frame_mask,

                        'internvid_short_input_ids':short_input_ids,# for video-caption ret
                        'internvid_long_input_ids':long_input_ids,# for video-caption ret

                        # 'target_mask': target_mask, # for masked video modeling
                        # 'context_mask': context_mask, # for masked video modeling
                    }
                yield res

            except Exception as e:
                print('encounter broken data %s: %s' % (item_id, e))

    def reset(self, seed):
        del self.buffer
        self.buffer = []
        return self.generate(seed)

    def sort_and_shuffle(self, data, seed):
        data.sort()
        random.Random(seed).shuffle(data)
        return data

    def get_surfix(self, name_list):
        return [n.split('/')[-1] for n in name_list]


class data_prefetcher():
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.loader = iter(self._dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            for key in self.next_input.keys():
                if isinstance(self.next_input[key], torch.Tensor):
                    self.next_input[key] = self.next_input[key].cuda(non_blocking=True)
            
    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input_item = self.next_input
        self.preload()
        return input_item

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)


def create_mm_embed_dataloader(
    dataset, batch_size, num_workers,
    cuda_prefetch=False,
):
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, 
        pin_memory=True, drop_last=True, collate_fn=default_collate,
        prefetch_factor=4,
    )

    if cuda_prefetch:
        loader = data_prefetcher(loader)

    loader = iter(loader)

    return loader


if __name__ == "__main__":
    import time
    # data_path = 'hdfs://harunasg/home/byte_data_tt_m/jinyang.leo/music_rec_data/data'
    # data_path = 'hdfs://harunasg/home/byte_data_tt_m/jinyang.leo/music_rec_data/finetune_v1_data'
    data_path = '/mnt/bn/jiny-ttls-i18n-fr1q/vd-foundation___InternVid-10M-FLT/intervid_anno/InternVid-10M-FLT-INFO_with_path_long_caption.jsonl'
    tokenizer_dir = '/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base'
    blip_tokenizer_dir = "/mnt/bn/yexiaoyu-test/models/huggingface/blip2"
    
    hdfs_dataset = VideoDatasetInternVid(data_path=data_path, shuffle=True, repeat=True, image_query_num = 10, buffer_size=256, max_frame_len=5, tokenizer_dir=tokenizer_dir,blip_tokenizer_dir = blip_tokenizer_dir)
    # torch.cuda.set_device(0)

    loader = create_mm_embed_dataloader(
        hdfs_dataset,
        batch_size=8,
        num_workers=16,
        cuda_prefetch=False,
    )
    for data in loader:
        # print("data['short_input_ids'].shape: ", data['short_input_ids'].shape)
        # print("data['long_input_ids'].shape: ", data['long_input_ids'].shape)
        print("data['frame_mask']", data['frame_mask'].shape)
        break

    # url = "/mnt/bn/yexiaoyu-test/data/000000039769.jpg"
    # image = [Image.open(url) for i in range(2)]
    # pixel_values,pixel_masks,spatial_shapes = hdfs_dataset.process_image(image)
    # print(len(pixel_values))
    # print(len(pixel_masks))
    # print(len(spatial_shapes))
    

