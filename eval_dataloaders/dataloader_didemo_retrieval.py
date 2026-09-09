from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json
from .rawvideo_util import RawVideoExtractor
from decord import VideoReader, cpu
from transformers import AutoProcessor,AutoTokenizer

import base64
from io import BytesIO

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

class DiDeMo_DataLoader(Dataset):
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            tokens_per_frame=32,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokens_per_frame = tokens_per_frame
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.image_processor = AutoProcessor.from_pretrained(tokenizer)
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "val", "test"]

        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, "val_list.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")

        video_json_path_dict = {}
        video_json_path_dict["train"] = os.path.join(self.data_path, "train_data.json")
        video_json_path_dict["val"] = os.path.join(self.data_path, "val_data.json")
        video_json_path_dict["test"] = os.path.join(self.data_path, "test_data.json")

        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]
        
        video_ids = [itm.split(".")[0] + ".mp4" for itm in video_ids]
        print(video_ids[:5])

        caption_dict = {}
        with open(video_json_path_dict[self.subset], 'r') as f:
            json_data = json.load(f)
        for itm in json_data:
            description = itm["description"]
            times = itm["times"]
            video = itm["video"].split(".")[0] + ".mp4"
            if video not in video_ids:
                continue

            # each video is split into 5-second temporal chunks
            # average the points from each annotator
            start_ = np.mean([t_[0] for t_ in times]) * 5
            end_ = (np.mean([t_[1] for t_ in times]) + 1) * 5
            if video in caption_dict:
                caption_dict[video]["start"].append(start_)
                caption_dict[video]["end"].append(end_)
                caption_dict[video]["text"].append(description)
            else:
                caption_dict[video] = {}
                caption_dict[video]["start"] = [start_]
                caption_dict[video]["end"] = [end_]
                caption_dict[video]["text"] = [description]

        for k_ in caption_dict.keys():
            caption_dict[k_]["start"] = [0]
            # trick to save time on obtaining each video length
            # [https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md]:
            # Some videos are longer than 30 seconds. These videos were truncated to 30 seconds during annotation.
            caption_dict[k_]["end"] = [31]
            caption_dict[k_]["text"] = [" ".join(caption_dict[k_]["text"])]

        video_dict = {}
        for root, dub_dir, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = video_file
                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_

        self.caption_dict = caption_dict
        self.video_dict = video_dict
        
        video_ids = list(set(video_ids) & set(self.caption_dict.keys()) & set(self.video_dict.keys()))

        # Get all captions
        self.iter2video_pairs_dict = {}
        for video_id in self.caption_dict.keys():
            if video_id not in video_ids:
                continue
            caption = self.caption_dict[video_id]
            n_caption = len(caption['start'])
            for sub_id in range(n_caption):
                self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = (video_id, sub_id)

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        
        print("len(video_ids): ", len(video_ids))

    def __len__(self):
        return len(self.iter2video_pairs_dict)

    def _get_text(self, video_id, sub_id):
        caption = self.caption_dict[video_id]
        
        ind = sub_id
        start, end = caption['start'][ind], caption['end'][ind]
        sentence = caption['text'][ind]
        
        text_output = self.tokenizer([sentence], padding='max_length', truncation=True, max_length=self.max_words, return_tensors="pt")

        return text_output.input_ids[0], start, end, sentence
    
    def get_video_dec(self, video_path, start_time, end_time):
        vreader = VideoReader(video_path, ctx=cpu(0))
        fps = vreader.get_avg_fps()
        f_start = 0 if start_time is None else int(start_time * fps)
        f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
        
        sample_fps = int(self.feature_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > self.max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=self.max_frames, dtype=int)]
        else:
            sample_pos = all_pos

        images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

        return images

    def _get_rawvideo(self, idx, start, end):
        
        video_path = self.video_dict[idx]
        
        start_time = start if start >= 0. else 0.
        end_time = end if end >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = end_time + 1

        #PIL.Image list, sampled exactly with 8 frames
        raw_video_data = self.get_video_dec(video_path, start_time, end_time)
        video_base64 = images_to_base64_strings(raw_video_data)
        #print("len(raw_video_data):",len(raw_video_data))

        inputs = self.image_processor(images=raw_video_data, return_tensors="pt")

        pixel_values = inputs['pixel_values']
        pixel_masks = inputs['pixel_attention_mask']
        spatial_shapes = inputs['spatial_shapes']

        now_len = pixel_values.shape[0]
        total_tokens = self.max_frames * self.tokens_per_frame
    
        # number of valid tokens
        valid_tokens = now_len * self.tokens_per_frame

        # create mask
        frame_mask = torch.zeros((total_tokens), dtype=pixel_masks.dtype)
        frame_mask[:valid_tokens] = 1

        if now_len < self.max_frames:
            pixel_values = torch.cat([pixel_values, torch.zeros_like(pixel_values[0].unsqueeze(0)).repeat(self.max_frames - now_len, *[1]*(pixel_values.dim()-1))], dim=0)
            pixel_masks = torch.cat([pixel_masks, torch.zeros_like(pixel_masks[0].unsqueeze(0)).repeat(self.max_frames - now_len, *[1]*(pixel_masks.dim()-1))], dim=0)
            spatial_shapes = torch.cat([spatial_shapes, torch.ones_like(spatial_shapes[0].unsqueeze(0)).repeat(self.max_frames - now_len, *[1]*(pixel_masks.dim()-1)) * 16], dim=0)


        return pixel_values,pixel_masks,spatial_shapes, video_base64, frame_mask

    def __getitem__(self, feature_idx):
        video_id, sub_id = self.iter2video_pairs_dict[feature_idx]

        text_input_ids, starts, ends,caption = self._get_text(video_id, sub_id)
        pixel_values, pixel_masks, spatial_shapes,video_frames, frame_mask = self._get_rawvideo(video_id, starts, ends)
        return text_input_ids, pixel_values, pixel_masks, spatial_shapes, frame_mask, caption, video_frames
