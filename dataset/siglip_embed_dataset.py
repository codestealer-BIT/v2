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

from transformers import AutoTokenizer


try:
    from .siglip2_preprocessor import Siglip2ImageProcessorFast
except:
    from siglip2_preprocessor import Siglip2ImageProcessorFast



try:
    from .hdfs_io import hlist_files, hisdir, hopen
except:
    from hdfs_io import hlist_files, hisdir, hopen



from IPython import embed




# print(f"Loading caption dict")
# tic = time.time()
# with open('/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/tt_captions_for_music.pkl', 'rb') as fr:
#     CAPTION_DICT = pickle.load(fr)
# toc = time.time()
# print(f"Load caption dict done, {len(CAPTION_DICT)} items, time cost: {toc-tic}")

import random

def random_keep_or_drop(text):
    # split by '.' and strip spaces
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    n = len(sentences)
    if n == 0:
        return text
    
    
    min_drop = 0
    num_drop = random.randint(min_drop, n-1)
    drop_indices = set(random.sample(range(n), num_drop))
    kept = [s for i, s in enumerate(sentences) if i not in drop_indices]
    
    return '. '.join(kept) + ('.' if kept else '')



class RandomResizeCrop(object):

    def __init__(self, size, crop_w, crop_h, resize_long_edge=False, is_training=True):
        self.size = size
        self.resize_long_edge = resize_long_edge
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.is_training = is_training

    def __call__(self, image_tensor):
        # resize
        h, w = image_tensor.shape[-2:]
        if self.resize_long_edge:
            ratio = float(self.size / float(max(h, w)))
        else:
            ratio = float(self.size / float(min(h, w)))

        new_w, new_h = round(w* ratio), round(h * ratio)
        new_w = max(self.crop_w, new_w)
        new_h = max(self.crop_h, new_h)

        image_tensor = F.resize(image_tensor, (new_h, new_w), interpolation=InterpolationMode.BICUBIC, antialias=True)

        if self.is_training:
            image_tensor = transforms.RandomCrop((self.crop_h, self.crop_w))(image_tensor)
        else:
            image_tensor = transforms.CenterCrop((self.crop_h, self.crop_w))(image_tensor)
        
        return image_tensor


class VideoHDFSDataset(IterableDataset):
    def __init__(
            self, 
            data_path,
            dict_path,
            rank: int = 0,
            world_size: int = 1,
            shuffle: bool = False,
            repeat: bool = False,
            verbose: bool = True,
            buffer_size: int = -1,
            max_frame_len: int = 8,
            max_text_len:int=64,
            image_w: int = 192,
            image_h: int = 384,
            image_mean: list = [0.485, 0.456, 0.406],
            image_std: list = [0.229, 0.224, 0.225],
            is_training: bool = True,
            tokenizer_dir=None,
        ):
        super(VideoHDFSDataset).__init__()
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size

        self.files = self.multi_thread_get_files(data_path)
        self.files = [f for f in self.files if f.find('_SUCCESS') < 0 and f.find("_temporary") < 0 and f.find(".caption") < 0]
        self.files.sort()
        print(len(self.files))
        if len(self.files) % self.world_size != 0:
            print('[DATA]--Whole dataset file num %s cannot split to worldsize %s ' %
                     (len(self.files), self.world_size))
        self.verbose = verbose
        self.repeat = repeat
        self.buffer = []
        self.buffer_size = buffer_size

        self.max_frame_len = max_frame_len
        self.max_text_len = max_text_len
        
        self.segment_map = {
            'caption': 0,
            'query': 0,
            'title': 0,
            'tier3_label': 0, # new
            'sticker': 2,
            'sticker_texts': 2,#new
            'challenge': 3,
            "challenge_names": 3,#new
            "hash_tags": 3,#new
            'music': 4,
            "music_title": 4,#new
            'user_nickname': 5,
            "nickname": 5,#new
            'ocr': 6,
            'asr': 7
        }

        self.get_caption_dict(dict_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) 
        self.processor = Siglip2ImageProcessorFast()
        
        self.image_mean = image_mean
        self.image_std = image_std

    def get_caption_dict(self, dict_path):
        print(f"Loading caption dict")
        tic = time.time()
        with open(dict_path, 'rb') as fr:
            self.CAPTION_DICT = pickle.load(fr)
        toc = time.time()
        print(f"Load caption dict done, {len(self.CAPTION_DICT)} items, time cost: {toc-tic}")

    def multi_thread_get_files(self, root_dir):
        print(f'fetching annotation files from {root_dir}')
        hdfs_files = hlist_files([root_dir])
        
        def get_file(file):
            if hisdir(file):
                sub_hdfs_files = hlist_files([file])
                return sub_hdfs_files
            else:
                return [file]

        max_thread = 64
        task_queue = []
        all_anno_files = []

        with ThreadPoolExecutor(max_workers=max_thread) as executor:
            for hdfs_file in hdfs_files:
                task_queue.append(executor.submit(get_file, hdfs_file))

            for future in tqdm(futures.as_completed(task_queue)):
                sub_hdfs_files = future.result()
                all_anno_files.extend(sub_hdfs_files)

        return all_anno_files

    def split_shard(self, data: List[Any], shard_idx: int, shard_size: int):
        num = len(data)
        if num < shard_size:
            raise RuntimeError("num:{} < shard size:{}".format(num, shard_size))
        start_idx = (num * shard_idx) // shard_size
        end_idx = (num * (shard_idx + 1)) // shard_size
        return data[start_idx: end_idx]

    def generate(self, seed=1234):
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

            for filepath in cur_worker_files:
                try:
                    with hopen(filepath, 'rb') as reader:
                        for _, line in enumerate(reader):
                            if self.buffer_size <= 0:
                                yield line.decode()
                            elif len(self.buffer) < self.buffer_size:
                                self.buffer.append(line.decode())
                            else:
                                yield self.enq_deq_buffer(line.decode())
                except Exception as e:
                    print('encounter broken file: %s %s' % (e, filepath))

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
        if now_len < self.max_frame_len:
            pixel_values = torch.cat([pixel_values, torch.zeros_like(pixel_values[0].unsqueeze(0)).repeat(self.max_frame_len - now_len, *[1]*(pixel_values.dim()-1))], dim=0)
            pixel_masks = torch.cat([pixel_masks, torch.zeros_like(pixel_masks[0].unsqueeze(0)).repeat(self.max_frame_len - now_len, *[1]*(pixel_masks.dim()-1))], dim=0)
            spatial_shapes = torch.cat([spatial_shapes, torch.ones_like(spatial_shapes[0].unsqueeze(0)).repeat(self.max_frame_len - now_len, *[1]*(pixel_masks.dim()-1)) * 16], dim=0)
        
        
        return pixel_values,pixel_masks,spatial_shapes
    

    def read_video_frames(self, video):
        if len(video) > self.max_frame_len:
            sample_idx = torch.linspace(0, len(video) - 1, self.max_frame_len).round().long().tolist()
            video = [video[i] for i in sample_idx]

        video = [base64.b64decode(video[i]) for i in range(len(video))]
        video = [Image.open(io.BytesIO(image_str)).convert('RGB') for image_str in video]
        # is_photo = self.check_is_photo(video)#意思是有多种不同的分辨率

        # video = [video[0]]

        video_tensors,frame_masks,spatial_shapes = self.process_image(video)

        assert video_tensors.shape[0] == self.max_frame_len, f"The input video frame number does not equal to {self.max_frame_len}"
        
        return video_tensors, frame_masks, spatial_shapes

    def process_text(self,texts):

        text_output = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_text_len, return_tensors="pt")
        text_input_ids = text_output.input_ids[0]
        return text_input_ids

    def __iter__(self):
        for item_str in self.generate():
            try:
                data_item = json.loads(item_str)
                item_id = data_item['item_id']

                # process video data
                video_frames = data_item['video']
                pixel_values,attention_mask,spatial_shapes = self.read_video_frames(video_frames)
                video_caption = data_item['video_caption']

                if video_caption == '':
                    if item_id in self.CAPTION_DICT:
                        video_caption = self.CAPTION_DICT[item_id]
                    else:
                        print(f"item_id {item_id} has no caption")
                        continue
                
                # #random drop 0-n-1 sentences and add <bos>
                # video_caption = '<bos>' + random_keep_or_drop(video_caption)

                text_input_ids = self.process_text([video_caption])

                

                res = {
                    'item_ids': str(item_id), 
                    'pixel_values': pixel_values, 
                    'pixel_attention_mask': attention_mask, 
                    'spatial_shapes':spatial_shapes,
                    'text_input_ids': text_input_ids,
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
    data_path = 'hdfs://harunasg/home/byte_data_tt_m/jinyang.leo/music_rec_data/finetune_v1_data_plus'
    dict_path = '/mnt/bn/yexiaoyu-test/data/tt_captions_for_music.pkl'
    tokenizer_dir = '/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base'
    
    hdfs_dataset = VideoHDFSDataset(data_path=data_path, dict_path = dict_path, shuffle=True, repeat=True, buffer_size=256, max_frame_len=5, tokenizer_dir=tokenizer_dir)
    # torch.cuda.set_device(0)

    loader = create_mm_embed_dataloader(
        hdfs_dataset,
        batch_size=2,
        num_workers=16,
        cuda_prefetch=False,
    )
    for data in loader:
        for key in ['pixel_values','pixel_attention_mask', 'spatial_shapes', 'text_input_ids']:
            print(key, "  ", data[key].shape)
        break

    # url = "/mnt/bn/yexiaoyu-test/data/000000039769.jpg"
    # image = [Image.open(url) for i in range(2)]
    # pixel_values,pixel_masks,spatial_shapes = hdfs_dataset.process_image(image)
    # print(len(pixel_values))
    # print(len(pixel_masks))
    # print(len(spatial_shapes))
    

