
from PIL import Image
import torch
import numpy as np
import random
import decord
from decord import VideoReader
import json
import os
import io
import base64
import re
import math
from functools import lru_cache

from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, IterableDataset, DataLoader, RandomSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as F

from transformers import AutoTokenizer,AutoProcessor


decord.bridge.set_bridge("torch")


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

    
class TikTokRetrievalDataset(Dataset):

    def __init__(self, 
                 video_root, 
                 max_short_len= 64,
                 max_text_len = 64,
                 max_frame_len=8, 
                 tokens_per_frame=32,
                 tokenizer_dir=None,
                 bert_tokenizer_dir=None
                ):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''        
        
        
        self.max_frame_len = max_frame_len
        self.tokens_per_frame = tokens_per_frame
        # self.video_root = video_root
        # self.data = os.listdir(video_root)
        # self.data.sort()

        self.data = []
        with open(video_root, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(line)
        
        # self.data = self.data[:1000]
        print("load ",len(self.data), " data from dict")

        
        self.txt2video = [i for i in range(len(self.data))]
        self.video2txt = self.txt2video 

        self.max_text_len = max_text_len
        self.max_short_len = max_short_len

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) 
        #self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer_dir) 
        self.processor = AutoProcessor.from_pretrained(tokenizer_dir)
        
            
    def __len__(self):
        return len(self.data)
    
    def process_text(self, captions, titles, stickers, ocr_texts, asr_texts): 

        short_caption, long_caption, fusion_caption = captions
        fusion_caption_output = self.tokenizer(fusion_caption, padding='max_length', truncation=True, max_length=self.max_text_len, return_attention_mask = True, return_tensors="pt")
        fusion_caption_input_ids = fusion_caption_output.input_ids[0]
        caption_attention_mask = fusion_caption_output.attention_mask[0]

        short_caption_input_ids = self.tokenizer(short_caption, padding='max_length', truncation=True, max_length=self.max_short_len, return_attention_mask = False, return_tensors="pt").input_ids[0]
        long_caption_input_ids = self.tokenizer(long_caption, padding='max_length', truncation=True, max_length=self.max_text_len, return_attention_mask = False, return_tensors="pt").input_ids[0]



        title_list = [titles,stickers,ocr_texts,asr_texts]
        title_output = self.tokenizer(title_list, padding='max_length', truncation=True, max_length=self.max_short_len, return_attention_mask = True, return_tensors="pt")
        
        title_input_ids = title_output.input_ids
        title_attention_mask = title_output.attention_mask


        return short_caption_input_ids, long_caption_input_ids, fusion_caption_input_ids, title_input_ids,title_attention_mask
    
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


        video_tensors,pixel_masks,spatial_shapes,frame_mask = self.process_image(video)

        assert video_tensors.shape[0] == self.max_frame_len, f"The input video frame number does not equal to {self.max_frame_len}"
        
        return video_tensors, pixel_masks, spatial_shapes,frame_mask


    def __getitem__(self, index):

        data_item= json.loads(self.data[index])
        video = data_item['video_frames']
        short_caption = data_item['short_caption']
        long_caption = data_item['long_caption']
        fusion_caption = data_item['fusion_caption']

        sticker = str(data_item['sticker']) if 'sticker' in data_item.keys() else ''
        title = str(data_item['title']) if 'title' in data_item.keys() else ''
        ocr_text = str(data_item['ocr_text']) if 'ocr_text' in data_item.keys() else ''
        asr_text = str(data_item['asr_text']) if 'asr_text' in data_item.keys() else ''
        
        short_caption_input_ids, long_caption_input_ids, fusion_caption_input_ids, title_input_ids, title_attention_mask = self.process_text(captions = (short_caption,long_caption,fusion_caption), titles = title, stickers = sticker, ocr_texts = ocr_text, asr_texts = asr_text)

        pixel_values,attention_mask,spatial_shapes,frame_mask = self.read_video_frames(video)

        res = {
            'item_ids': str(data_item['item_ids']),
            'pixel_values': pixel_values, 
            'pixel_attention_mask': attention_mask, 
            'spatial_shapes':spatial_shapes,

            'frame_mask':frame_mask,

            'short_input_ids':short_caption_input_ids,# for video-caption ret
            'long_input_ids':long_caption_input_ids,# for video-caption ret

            'fusion_input_ids':fusion_caption_input_ids,# for fusion-caption ret
        

            'title_input_ids': title_input_ids,
            'title_attention_mask': title_attention_mask,#for video-title-caption ret

            'fusion_caption':str(fusion_caption),
            'short_caption': str(short_caption),
            'video_b64': ",".join(data_item['video_frames'])
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
        frames = [transforms.ToPILImage()(raw_sample_frms[i]) for i in range(raw_sample_frms.shape[0])]
        #frames[0].save("/opt/tiger/MMPretrain/vis/{}.png".format(video_path.split('/')[-1]))

        return frames


if __name__ == "__main__":
    
    tokenizer_dir = '/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base'
    bert_tokenizer_dir = "/mnt/bn/yexiaoyu-test/models/huggingface/bert_uncased"
    video_root = "/mnt/bn/jiny-ttls-i18n-fr1q/tiktok_30k_recaption.jsonl"
    from transformers import Siglip2Model
    # model = Siglip2Model.from_pretrained(tokenizer_dir)
    # model.to('cuda')
    test_dataset = TikTokRetrievalDataset(video_root=video_root, 
                 max_text_len= 64,
                 tokenizer_dir=tokenizer_dir,
                 bert_tokenizer_dir = bert_tokenizer_dir)

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        collate_fn=default_collate,
    ) 

    print("len(test_dataset)",len(test_dataset))
    for data in test_loader:
        for key in ['caption_input_ids']:
            print(key, "  ", data[key])
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


        