
from PIL import Image
import torch
import numpy as np
import random
import decord
from decord import VideoReader
import json
import os
import re
import math
from functools import lru_cache

from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, IterableDataset, DataLoader, RandomSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as F

from transformers import AutoTokenizer,AutoProcessor

import base64
from io import BytesIO

decord.bridge.set_bridge("torch")

def images_to_base64_strings(image_list):
    """
    将PIL Image对象列表转换为JPEG格式的base64字符串，并用逗号拼接
    
    参数:
        image_list: PIL Image对象的列表
        
    返回:
        str: 所有图像的base64字符串用逗号拼接的结果
    """
    base64_strings = []
    
    for img in image_list:
        # 创建一个字节流缓冲区
        buffered = BytesIO()
        
        # 将图像保存为JPEG格式到缓冲区
        # 注意：如果图像是RGBA模式，需要转换为RGB才能保存为JPEG
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        img.save(buffered, format="JPEG")
        
        # 将缓冲区的内容转换为base64字符串
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        base64_strings.append(img_str)
    
    # 用逗号拼接所有base64字符串
    return ','.join(base64_strings)


def pre_caption(caption,max_words=200):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

    
class VideoRetrievalDataset(Dataset):

    def __init__(self, 
                 video_root, 
                 ann_file, 
                 max_text_len = 64,
                 max_frame_len=8, 
                 tokens_per_frame=32,
                 video_fmt='.mp4',
                 image_mean: list = [0.485, 0.456, 0.406],
                 image_std: list = [0.229, 0.224, 0.225],
                 tokenizer_dir=None,
                ):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''        
        with open(ann_file, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            self.annotation = [json.loads(line) for line in lines]
        
        self.max_frame_len = max_frame_len
        self.tokens_per_frame = tokens_per_frame
        self.video_root = video_root
        self.video_fmt = video_fmt

        self.ann_len = len(self.annotation)
        self.txt2video = [i for i in range(len(self.annotation))]
        self.video2txt = self.txt2video 

        self.max_text_len = max_text_len

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) 
        self.processor = AutoProcessor.from_pretrained(tokenizer_dir)
        
        self.image_mean = image_mean
        self.image_std = image_std             
            
            
    def __len__(self):
        return len(self.annotation)
    
    def process_text(self,texts):
        text_output = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_text_len, return_tensors="pt")
        text_input_ids = text_output.input_ids[0]
        return text_input_ids
    

    def __getitem__(self, index):

        ann = self.annotation[index]  

        video_path = os.path.join(self.video_root, ann['clip_name'] + self.video_fmt) 

        video = self._load_video_from_path_decord(video_path)
        video_base_64 = images_to_base64_strings(video)
        text = pre_caption(ann['caption'],200)
        #video[-1].save("/opt/tiger/MMPretrain/vis/{}.png".format("_".join(text.split(" "))))

        inputs = self.processor(images=video, return_tensors="pt")

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


            
        assert pixel_values.shape[0] == self.max_frame_len, f"The input video frame number does not equal to {self.max_frame_len}"

        
        text_input_ids = self.process_text(text)

        res = {
            'item_ids': str(ann['clip_name']), 
            'captions': text,
            'video_b64': video_base_64,
            'pixel_values': pixel_values, 
            'pixel_attention_mask': pixel_masks, 
            'spatial_shapes':spatial_shapes,
            'frame_mask':frame_mask,
            'text_input_ids': text_input_ids
        }
        return res

        

    def _load_video_from_path_decord(self, video_path):
        try:
            vr = VideoReader(video_path)

            vlen = len(vr)

            start_idx, end_idx = 0, vlen
            frame_indices = np.arange(start_idx, end_idx, vlen / self.max_frame_len, dtype=int)
            raw_sample_frms = vr.get_batch(frame_indices)

        except Exception as e:
            return None

        raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2) # (F, H, W, C) -> (F, C, H, W)
        #print("raw_sample_frms.shape",raw_sample_frms.shape)
        frames = [transforms.ToPILImage()(raw_sample_frms[i]).convert('RGB') for i in range(raw_sample_frms.shape[0])]
        #frames[0].save("/opt/tiger/MMPretrain/vis/{}.png".format(video_path.split('/')[-1]))

        return frames


if __name__ == "__main__":
    
    tokenizer_dir = '/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base'
    video_root = "/mnt/bn/yexiaoyu-test/data/lavis/msr-vtt/MSRVTT/videos/all"
    ann_file = "/mnt/bn/yexiaoyu-test/data/lavis/msr-vtt/msrvtt_test.jsonl"
    from transformers import Siglip2Model
    # model = Siglip2Model.from_pretrained(tokenizer_dir)
    # model.to('cuda')
    test_dataset = VideoRetrievalDataset(video_root=video_root, 
                 ann_file=ann_file, 
                 tokenizer_dir=tokenizer_dir)

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=default_collate,
    ) 

    print("test_dataset.ann_len",test_dataset.ann_len)
    count = 0
    for data in test_loader:
        print(data['captions'])
        print(len(data['video_base64'][0].split(',')))
        # for key in ['pixel_values','pixel_attention_mask', 'spatial_shapes', 'text_input_ids']:
        #     print(key, "  ", data[key].shape)
        break

        # output = model(input_ids= data['text_input_ids'].cuda(),
        #     pixel_values = data['pixel_values'][:,-1,:].cuda(),
        #     pixel_attention_mask= data['pixel_attention_mask'][:,-1,:].cuda(),
        #     spatial_shapes = data['spatial_shapes'][:,-1,:].cuda(),
        #     attention_mask = data['text_attention_masks'].cuda(),)
            
        # logits_per_image = output.logits_per_image
        # probs = torch.sigmoid(logits_per_image) # these are the probabilities
        # print("probs: ", probs)

        # count += 1


        