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



try:
    from .hdfs_io import hlist_files, hisdir, hopen
except:
    from hdfs_io import hlist_files, hisdir, hopen



from IPython import embed


import re
import random

def clean_caption(caption: str) -> str:
    # strip leading/trailing whitespace first
    text = caption.strip()
    if random.random() < 0.5:
        
        # 1. remove leading 'A ' or 'An ' (case-insensitive, only at start)
        text = re.sub(r'^(a|an)\s+', '', text, flags=re.IGNORECASE)

        # 2. remove trailing '.' (only if it's the very last char)
        text = re.sub(r'\.$', '', text)

        # remove conjunction words surrounded by spaces
        text = re.sub(r'\b(?:and|or|but|while|where|)\b', '', text, flags=re.IGNORECASE)
        # remove commas
        text = text.replace(',', '')
        # clean up multiple spaces after removal
        text = re.sub(r'\s+', ' ', text).strip()

    return text





# print(f"Loading caption dict")
# tic = time.time()
# with open('/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/tt_captions_for_music.pkl', 'rb') as fr:
#     CAPTION_DICT = pickle.load(fr)
# toc = time.time()
# print(f"Load caption dict done, {len(CAPTION_DICT)} items, time cost: {toc-tic}")

# import random
# from PIL import Image

# def reverse_crop(images):
#     """
#     With 50% probability, reverse the aspect ratio (width:height -> height:width)
#     of each image by center cropping vertically, keeping the width unchanged.

#     Args:
#         images (list[PIL.Image.Image]): list of PIL images

#     Returns:
#         list[PIL.Image.Image]: processed list of images
#     """
#     if random.random() < 0.5:  # 50% chance
#         processed = []
#         for img in images:
#             w, h = img.size
#             target_h = int(w * w / h) 
#             if h > target_h:
#                 # center crop vertically
#                 top = (h - target_h) // 2
#                 bottom = top + target_h
#                 cropped = img.crop((0, top, w, bottom))
#                 processed.append(cropped)
#             else:
#                 # if height < width (can't crop), just skip or keep original
#                 processed.append(img)
#         return processed
#     else:
#         return images




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


class VideoHDFSDatasetShortLong(IterableDataset):
    def __init__(
            self, 
            data_path,
            dict_path,
            rank: int = 0,
            world_size: int = 1,
            shuffle: bool = False,
            repeat: bool = False,
            verbose: bool = False,
            buffer_size: int = -1,
            max_frame_len: int = 8,
            max_text_len:int=64,
            max_short_len:int=64,
            tokens_per_frame: int = 32,
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

        self.files = self.get_files(data_path)
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
        self.tokens_per_frame = tokens_per_frame
        self.max_text_len = max_text_len #long text length
        self.max_short_len = max_short_len #short text length
        
        self.segment_map = {
            'caption': 0,
            'title': 0,
            'sticker': 1,
            'ocr': 2,
            'asr': 3
        }

        self.get_caption_dict(dict_path)

        #processor for auto processor
        processor = AutoProcessor.from_pretrained(blip_tokenizer_dir)
        self.blip_tokenizer = processor.tokenizer
        self.image_query_num = image_query_num
        self.image_token = processor.image_token

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) 
        self.processor = Siglip2ImageProcessorFast()
        
        self.image_mean = image_mean
        self.image_std = image_std

    def get_caption_dict(self, dict_path):
        print("getting caption dicts")
        self.caption_dict = {
            "short_caption": os.path.join(dict_path, "short_caption/finetune_v1_data_plus"),
            "long_caption": os.path.join(dict_path, "long_caption/finetune_v1_data_plus"),
            "fusion_caption": os.path.join(dict_path, "fusion_caption/finetune_v1_data_plus")
        }
        print(self.caption_dict)

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
    
    def get_files(self, anno_file):
        with open(anno_file, 'r') as fr:
            all_anno_files = json.load(fr)
        return all_anno_files

    def split_shard(self, data: List[Any], shard_idx: int, shard_size: int):
        num = len(data)
        if num < shard_size:
            raise RuntimeError("num:{} < shard size:{}".format(num, shard_size))
        start_idx = (num * shard_idx) // shard_size
        end_idx = (num * (shard_idx + 1)) // shard_size
        return data[start_idx: end_idx]
    
    def get_local_anno_for_hdfs_file(self, filepath, suffix = "short_caption"):
        local_root_dir = self.caption_dict[suffix]
        subdir_name, file_name = filepath.split('/')[-2:]
        file_name = file_name.split('.')[0] + '-{}.jsonl'.format(suffix)
        cur_anno_file_path = os.path.join(local_root_dir, f'{subdir_name}-{file_name}')
        #print("file_path: ", filepath, "\tcur_anno_file_path: ", cur_anno_file_path)
        if not os.path.exists(cur_anno_file_path):
            return None
        cur_anno_dict = {}
        with jsonlines.open(cur_anno_file_path, 'r') as reader:
            for item in reader:
                cur_anno_dict[item['item_id']] = item
        return cur_anno_dict
    
    

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

            for filepath in cur_worker_files:
                try:
                    short_anno_dict = self.get_local_anno_for_hdfs_file(filepath, "short_caption")
                    long_anno_dict = self.get_local_anno_for_hdfs_file(filepath, "long_caption")
                    fused_anno_dict = self.get_local_anno_for_hdfs_file(filepath, "fusion_caption")
                    if(short_anno_dict is None or long_anno_dict is None or fused_anno_dict is None):
                        continue

                    with hopen(filepath, 'rb') as reader:
                        for _, line in enumerate(reader):
                            data_anno = json.loads(line.decode())
                            item_id = data_anno['item_id']
                            if item_id not in short_anno_dict or item_id not in long_anno_dict or item_id not in fused_anno_dict:
                                continue
                            data_anno.update(short_anno_dict[item_id])
                            data_anno.update(long_anno_dict[item_id])
                            data_anno.update(fused_anno_dict[item_id])
                            if self.buffer_size <= 0:
                                yield data_anno
                            elif len(self.buffer) < self.buffer_size:
                                    self.buffer.append(data_anno)
                            else:
                                yield self.enq_deq_buffer(data_anno)
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
        
    #     # context mask
    #     context_mask = 1 - target_mask
        
    #     return target_mask, context_mask
    

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
    

    def read_video_frames(self, video):
        if len(video) > self.max_frame_len:
            sample_idx = torch.linspace(0, len(video) - 1, self.max_frame_len).round().long().tolist()
            video = [video[i] for i in sample_idx]

        video = [base64.b64decode(video[i]) for i in range(len(video))]
        video = [Image.open(io.BytesIO(image_str)).convert('RGB') for image_str in video]
        is_photo = self.check_is_photo(video)#意思是有多种不同的分辨率

        # video = [video[0]]

        video_tensors,pixel_masks,spatial_shapes,frame_mask = self.process_image(video)

        assert video_tensors.shape[0] == self.max_frame_len, f"The input video frame number does not equal to {self.max_frame_len}"
        
        return video_tensors, pixel_masks, spatial_shapes,frame_mask

    def process_text(self, captions, titles, stickers, ocr_texts, asr_texts): 

        short_caption, long_caption, fusion_caption = captions
        fusion_caption_output = self.tokenizer(fusion_caption, padding='max_length', truncation=True, max_length=self.max_text_len, return_attention_mask = True, return_tensors="pt")
        fusion_caption_input_ids = fusion_caption_output.input_ids[0]
        caption_attention_mask = fusion_caption_output.attention_mask[0]

        short_caption_input_ids = self.tokenizer(short_caption, padding='max_length', truncation=True, max_length=self.max_short_len, return_attention_mask = False, return_tensors="pt").input_ids[0]
        long_caption_input_ids = self.tokenizer(long_caption, padding='max_length', truncation=True, max_length=self.max_text_len, return_attention_mask = False, return_tensors="pt").input_ids[0]

        
        text_encoding = self.blip_tokenizer(fusion_caption + "</s>", padding='max_length', truncation=True, max_length=self.max_text_len, return_attention_mask = True, return_tensors="pt")
        image_tokens = self.image_token.content * self.image_query_num

        image_text_encoding = self.blip_tokenizer(image_tokens, add_special_tokens = False, padding = False, truncation = False, return_attention_mask = True,return_tensors = "pt")
        
        
        blip_input_ids = torch.cat((image_text_encoding['input_ids'],text_encoding['input_ids']),dim =-1)[0]
        blip_attention_mask = torch.cat((image_text_encoding['attention_mask'],text_encoding['attention_mask']),dim = -1)[0]



        title_list = [titles,stickers,ocr_texts,asr_texts]
        title_output = self.tokenizer(title_list, padding='max_length', truncation=True, max_length=self.max_short_len, return_attention_mask = True, return_tensors="pt")
        
        title_input_ids = title_output.input_ids
        title_attention_mask = title_output.attention_mask


        return short_caption_input_ids, long_caption_input_ids, fusion_caption_input_ids, blip_input_ids, blip_attention_mask, title_input_ids,title_attention_mask

    def __iter__(self):
        for data_item in self.generate():
            try:
                item_id = data_item['item_id']

                # process video data
                video_frames = data_item['video']
                pixel_values,attention_mask,spatial_shapes,frame_mask = self.read_video_frames(video_frames)

                if frame_mask.sum() == 0:
                    print('encounter all padding frames in video %s' % (item_id))
                    continue
                #target_mask, context_mask = self.generate_mask_and_complement(attention_mask)

                short_caption = data_item['short_caption']
                short_caption = clean_caption(short_caption)
                #clean_caption for short captions
                long_caption = data_item['long_caption']
                fusion_caption = data_item['fusion_caption']

                sticker = str(data_item['sticker'])
                title = str(data_item['text'])
                ocr_text = str(data_item['ocr_text'])
                asr_text = str(data_item['video_asr'])
                
                if sticker + title + ocr_text + asr_text == '': # no title info
                    continue


                short_caption_input_ids, long_caption_input_ids, fusion_caption_input_ids, blip_input_ids, blip_attention_mask, title_input_ids, title_attention_mask = self.process_text(captions = (short_caption,long_caption,fusion_caption), titles = title, stickers = sticker, ocr_texts = ocr_text, asr_texts = asr_text)

                # pixel_values = samples['pixel_values']
                # pixel_attention_mask = samples['pixel_attention_mask']
                # spatial_shapes = samples['spatial_shapes']

                # caption_input_ids = samples['caption_input_ids']#caption的segmentid是zero可以直接传 
                # caption_attention_mask = samples['caption_attention_mask']#在text encoder的时候不要传

                # title_input_ids = samples['title_input_ids']
                # title_segment_ids = samples['title_segment_ids']
                # title_attention_mask = samples['title_attention_mask']#在text encoder的时候不要传

                if self.stage ==1:
                    res = {
                        'item_ids': str(item_id), 
                        'pixel_values': pixel_values, 
                        'pixel_attention_mask': attention_mask, 
                        'spatial_shapes':spatial_shapes,

                        'frame_mask':frame_mask,

                        'short_input_ids':short_caption_input_ids,# for video-caption ret
                        'long_input_ids':long_caption_input_ids,# for video-caption ret

                        # 'fusion_input_ids':fusion_caption_input_ids,# for fusion-caption ret
                        
                        # 'blip_input_ids': blip_input_ids,
                        # 'blip_attention_mask': blip_attention_mask,#for video-grounded text gen

                        # 'title_input_ids': title_input_ids,
                        # 'title_attention_mask': title_attention_mask,#for video-title-caption ret
                    }
                else:
                    res = {
                        'item_ids': str(item_id), 
                        'pixel_values': pixel_values, 
                        'pixel_attention_mask': attention_mask, 
                        'spatial_shapes':spatial_shapes,

                        'frame_mask':frame_mask,

                        'short_input_ids':short_caption_input_ids,# for video-caption ret
                        'long_input_ids':long_caption_input_ids,# for video-caption ret

                        'fusion_input_ids':fusion_caption_input_ids,# for fusion-caption ret
                        
                        'blip_input_ids': blip_input_ids,
                        'blip_attention_mask': blip_attention_mask,#for video-grounded text gen

                        'title_input_ids': title_input_ids,
                        'title_attention_mask': title_attention_mask,#for video-title-caption ret
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
    data_path = '/mnt/bn/jiny-ttls-i18n-fr1q/final_tiktok_data_for_small_model/finetune_v1_data_plus_files_filtered.json'
    dict_path = '/mnt/bn/jiny-ttls-i18n-fr1q/final_tiktok_data_for_small_model'
    tokenizer_dir = '/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base'
    blip_tokenizer_dir = "/mnt/bn/yexiaoyu-test/models/huggingface/blip2"
    
    hdfs_dataset = VideoHDFSDatasetShortLong(data_path=data_path, dict_path = dict_path, shuffle=True, repeat=True, image_query_num = 10, buffer_size=256, max_frame_len=5, tokenizer_dir=tokenizer_dir,blip_tokenizer_dir = blip_tokenizer_dir)
    # torch.cuda.set_device(0)

    loader = create_mm_embed_dataloader(
        hdfs_dataset,
        batch_size=1,
        num_workers=16,
        cuda_prefetch=False,
    )
    count = 0
    for data in loader:
        # for key in ['short_input_ids','long_input_ids','fusion_input_ids']:
        #     print(key, "  ", data[key])
        print("data['short_input_ids']", data['short_input_ids'].shape)
        #print("data['context_mask']", data['context_mask'].sum())
        count += 1
        if count > 10:
            break

    # url = "/mnt/bn/yexiaoyu-test/data/000000039769.jpg"
    # image = [Image.open(url) for i in range(2)]
    # pixel_values,pixel_masks,spatial_shapes = hdfs_dataset.process_image(image)
    # print(len(pixel_values))
    # print(len(pixel_masks))
    # print(len(spatial_shapes))
    

