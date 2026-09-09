import os
import torch
import sys
from copy import deepcopy

import argparse
import datetime
import pickle
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, SequentialSampler, DistributedSampler
from collections import OrderedDict
from einops import rearrange

from concurrent.futures import ProcessPoolExecutor
from concurrent import futures

import io
import base64
from dataset.hdfs_io import hlist_files, hopen
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import json
import jsonlines
from tqdm import tqdm
from trainer_misc import init_distributed_mode
from config import cfg
from IPython import embed
from video_clip import VideoSigLIP2ForMusic, VideoSigLIP2ForMusicOnlyMatching, VideoSigLIP2ForMusicNegativeFiltering, VideoSigLIP2ForMusicNegativeFilteringAddLang, VideoSigLIP2ForMusicNegativeFilteringBMAddLang
from transformers import AutoTokenizer, AutoProcessor
from dataset.siglip2_preprocessor import Siglip2ImageProcessorFast
from dataset.video_music_dataset import VideoMusicHDFSDataset
from collections import defaultdict

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif v.lower() in ('none', 'null', ''):
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser('Pytorch Multi-process script', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--model_dtype', default='bf16', type=str, help="The Model Dtype: bf16 or df16")
    parser.add_argument('--model_path', default='', type=str, help='The pre-trained weight path')
    parser.add_argument('--output_dir', type=str, default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory or file')
    parser.add_argument('--add_lyrics_ue', default=False, type=str2bool)
    parser.add_argument('--add_title_ue', default=False, type=str2bool)    
    parser.add_argument('--add_title_text', default=False, type=str2bool)
    parser.add_argument('--add_user_lang_and_country_method', default="gate", type=str)
    return parser.parse_args()


def build_model(model_path, pretrained_ckpt_path, args):
    cfg_path = "/mnt/bn/search-ad-creative-zhijie/wangxiuqi.0601/mm/MM-Embedding/experiments/config_finetune_mmembed_v3.yaml"

    if cfg_path:
        cfg.update_cfg(cfg_path)

    # model = VideoSigLIP2ForMusic(
    #     model_path,
    #     gpuwise_nce=True,
    #     interpolate=6,
    #     use_frame_mask=True,
    #     # add_lyrics_ue=args.add_lyrics_ue,
    #     # add_title_ue=args.add_title_ue,
    #     # add_user_lang_and_country_code=args.add_user_lang_and_country_code
    # )

    # model = VideoSigLIP2ForMusicNegativeFiltering(
    #     model_path,
    #     gpuwise_nce=True,
    #     interpolate=6,
    #     use_frame_mask=True,
    #     # add_lyrics_ue=args.add_lyrics_ue,
    #     # add_title_ue=args.add_title_ue,
    #     # add_user_lang_and_country_code=args.add_user_lang_and_country_code
    # )

    # model = VideoSigLIP2ForMusicNegativeFilteringAddLang(
    #     model_path,
    #     gpuwise_nce=True,
    #     interpolate=6,
    #     use_frame_mask=True,
    #     # add_lyrics_ue=args.add_lyrics_ue,
    #     # add_title_ue=args.add_title_ue,
    #     user_info_fusion_method='gate'
    # )

    model = VideoSigLIP2ForMusicNegativeFilteringBMAddLang(
        model_path,
        gpuwise_nce=True,
        interpolate=6,
        use_frame_mask=True,
        # add_lyrics_ue=args.add_lyrics_ue,
        # add_title_ue=args.add_title_ue,
        user_info_fusion_method=args.add_user_lang_and_country_method,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path) 
    processor = Siglip2ImageProcessorFast()

    if pretrained_ckpt_path:
        print(f"Loading the pre-trained checkpoint from {pretrained_ckpt_path}")
        pretrained_checkpoint = torch.load(pretrained_ckpt_path, map_location='cpu')
        load_res = model.load_state_dict(pretrained_checkpoint, strict=False)
        print(f"Loading result: {load_res}")

    return model, processor, tokenizer


def process_text(tokenizer, music_caption_dict, args):
    if music_caption_dict is None:
        music_attribute = ''
        music_capton = ''
    else:
        try:
            music_capton = music_caption_dict['content']
        except:
            music_capton = ''
        try:
            music_genre = ','.join(music_caption_dict['genre'])
        except:
            music_genre = ''

        try:
            music_language = ','.join(music_caption_dict['language'])
        except:
            music_language = ''

        try:
            music_theme = ','.join(music_caption_dict['theme'])
        except:
            music_theme = ''

        try:
            music_mood  = ','.join(music_caption_dict['mood'])
        except:
            music_mood = ''
        
        try:
            music_title = music_caption_dict['title']
        except:
            music_title = ''
        
        if args.add_title_text:
            music_attribute = f"title: {music_title}; genre: {music_genre}; language: {music_language}; theme: {music_theme}; mood: {music_mood}"
        else:
            music_attribute = f"genre: {music_genre}; language: {music_language}; theme: {music_theme}; mood: {music_mood}"
    
    music_attribute_input_ids = tokenizer(music_attribute, padding='max_length', truncation=True, max_length=64, return_attention_mask=False, return_tensors="pt").input_ids[0]
    music_caption_input_ids = tokenizer(music_capton, padding='max_length', truncation=True, max_length=284, return_attention_mask=False, return_tensors="pt").input_ids[0]
    
    return music_attribute_input_ids, music_caption_input_ids


def process_video(processor, image_path):
    max_frame_len = 5
    tokens_per_frame = 32   # CANNOT MODIFY
    image = Image.open(image_path).convert("RGB")
    # video = [image for _ in range(max_frame_len)]
    video = [image]
    inputs = processor(images=video, return_tensors="pt")

    pixel_values = inputs['pixel_values']
    pixel_masks = inputs['pixel_attention_mask']
    spatial_shapes = inputs['spatial_shapes']

    now_len = pixel_values.shape[0]
    total_tokens = max_frame_len * tokens_per_frame
    # number of valid tokens
    valid_tokens = now_len * tokens_per_frame

    # create mask
    frame_mask = torch.zeros((total_tokens), dtype=pixel_masks.dtype)
    frame_mask[:valid_tokens] = 1

    if now_len < max_frame_len:
        pixel_values = torch.cat([pixel_values, torch.zeros_like(pixel_values[0].unsqueeze(0)).repeat(max_frame_len - now_len, *[1]*(pixel_values.dim()-1))], dim=0)
        pixel_masks = torch.cat([pixel_masks, torch.zeros_like(pixel_masks[0].unsqueeze(0)).repeat(max_frame_len - now_len, *[1]*(pixel_masks.dim()-1))], dim=0)
        spatial_shapes = torch.cat([spatial_shapes, torch.ones_like(spatial_shapes[0].unsqueeze(0)).repeat(max_frame_len - now_len, *[1]*(pixel_masks.dim()-1)) * 16], dim=0)
    
    return pixel_values, pixel_masks, spatial_shapes, frame_mask

def process_evalset_video(processor, frames):
    max_frame_len = 5
    tokens_per_frame = 32   # CANNOT MODIFY
    video = [Image.open(io.BytesIO(base64.b64decode(frame))).convert("RGB") for frame in frames]

    inputs = processor(images=video, return_tensors="pt")

    pixel_values = inputs['pixel_values']
    pixel_masks = inputs['pixel_attention_mask']
    spatial_shapes = inputs['spatial_shapes']

    now_len = pixel_values.shape[0]
    total_tokens = max_frame_len * tokens_per_frame
    # number of valid tokens
    valid_tokens = now_len * tokens_per_frame

    # create mask
    frame_mask = torch.zeros((total_tokens), dtype=pixel_masks.dtype)
    frame_mask[:valid_tokens] = 1

    if now_len < max_frame_len:
        pixel_values = torch.cat([pixel_values, torch.zeros_like(pixel_values[0].unsqueeze(0)).repeat(max_frame_len - now_len, *[1]*(pixel_values.dim()-1))], dim=0)
        pixel_masks = torch.cat([pixel_masks, torch.zeros_like(pixel_masks[0].unsqueeze(0)).repeat(max_frame_len - now_len, *[1]*(pixel_masks.dim()-1))], dim=0)
        spatial_shapes = torch.cat([spatial_shapes, torch.ones_like(spatial_shapes[0].unsqueeze(0)).repeat(max_frame_len - now_len, *[1]*(pixel_masks.dim()-1)) * 16], dim=0)
    
    return pixel_values, pixel_masks, spatial_shapes, frame_mask

def calculate_music_vectors(music_tensor_file, model, tokenizer, output_dir, device, music_caption_file_dict, args):
    music_tensor_list = torch.load(music_tensor_file)
    filename = os.path.split(music_tensor_file)[-1]
    cache_dir = '/mnt/bn/search-ad-creative-zhijie/wangxiuqi.0601/mm/music_emb/music_library_0911_music_candidates_filter_train/feature_cache'
    feature_cache_file = os.path.join(cache_dir, filename)

    if os.path.exists(feature_cache_file):
        feature_cache = torch.load(feature_cache_file)
        all_music_ids = feature_cache['all_music_ids']
        all_music_content_vector = feature_cache['all_music_content_vector']
        all_music_title_vector = feature_cache['all_music_title_vector']
        all_music_cover_vector = feature_cache['all_music_cover_vector']
        all_music_attribute_input_ids = feature_cache['all_music_attribute_input_ids']
        all_music_caption_input_ids = feature_cache['all_music_caption_input_ids']

    else:
        all_music_ids = []
        all_music_content_vector, all_music_title_vector, all_music_cover_vector = [], [], []
        all_music_attribute_input_ids, all_music_caption_input_ids = [], []
        
        print(f"Processing the input features")
        for item in tqdm(music_tensor_list):
            music_id = item['music_id']
            music_caption_file = music_caption_file_dict[music_id]

            if os.path.exists(music_caption_file):
                with open(music_caption_file, 'r') as fr:
                    music_caption_dict = json.load(fr)
            else:
                music_caption_dict = None

            music_attribute_input_ids, music_caption_input_ids = process_text(tokenizer, music_caption_dict, args)

            all_music_ids.append(music_id)
            all_music_content_vector.append(item['music_content_ue_vector'])
            all_music_title_vector.append(item['music_title_ue_vector'])
            all_music_cover_vector.append(item['music_cover_ue_vector'])
            all_music_attribute_input_ids.append(music_attribute_input_ids)
            all_music_caption_input_ids.append(music_caption_input_ids)
        
        all_music_content_vector = torch.stack(all_music_content_vector, dim=0)
        all_music_title_vector = torch.stack(all_music_title_vector, dim=0)
        all_music_cover_vector = torch.stack(all_music_cover_vector, dim=0)
        all_music_attribute_input_ids = torch.stack(all_music_attribute_input_ids, dim=0)
        all_music_caption_input_ids = torch.stack(all_music_caption_input_ids, dim=0)

        feature_cache = {
            'all_music_ids': all_music_ids,
            'all_music_content_vector': all_music_content_vector,
            'all_music_title_vector': all_music_title_vector,
            'all_music_cover_vector': all_music_cover_vector,
            'all_music_attribute_input_ids': all_music_attribute_input_ids,
            'all_music_caption_input_ids': all_music_caption_input_ids,
        }
        torch.save(feature_cache, feature_cache_file)

    batch_size = 2048
    all_music_ue_vector = []
    print(f"Starting to inference")
    for i in tqdm(range(0, len(all_music_content_vector), batch_size)):
        batch_music_content_vector = all_music_content_vector[i:i+batch_size].to(device)
        batch_music_cover_vector = all_music_cover_vector[i:i+batch_size].to(device)
        batch_music_attribute_input_ids = all_music_attribute_input_ids[i:i+batch_size].to(device)
        batch_music_caption_input_ids = all_music_caption_input_ids[i:i+batch_size].to(device)

        with torch.no_grad():
            batch_music_ue_vector = model.extract_music_embeds(batch_music_content_vector, batch_music_cover_vector, batch_music_attribute_input_ids, batch_music_caption_input_ids)
            batch_music_ue_vector = batch_music_ue_vector.cpu()
        
        all_music_ue_vector.append(batch_music_ue_vector)

    all_music_ue_vector = torch.cat(all_music_ue_vector, dim=0)
    
    output_data = []
    for index in range(len(all_music_ids)):
        output_data.append({
            'music_id': all_music_ids[index],
            'music_ue_vector': all_music_ue_vector[index],
        })
    
    output_file = os.path.join(output_dir, filename)
    torch.save(output_data, output_file)
    return len(output_data)

def to_tensor(x):
    # 如果是字符串格式的向量，如 "[-0.1, 0.2, 0.3]"，先解析
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except json.JSONDecodeError:
            x = ast.literal_eval(x)
    # 转为 torch tensor，强制 float32 类型
    return torch.tensor(x, dtype=torch.float32)

def calculate_evalset_music_vectors(music_dict_file, model, tokenizer, output_dir, device, args):
    all_music_ids = []
    all_music_content_vector, all_music_title_vector, all_music_cover_vector, all_music_lyrics_vector = [], [], [], []
    all_music_attribute_input_ids, all_music_caption_input_ids = [], []
    with open(music_dict_file, 'r') as f:
        for line in f:
            music_caption_dict = json.loads(line.strip())
            music_id = music_caption_dict['clip_id']
            music_content_vector = to_tensor(music_caption_dict['content_vector'])
            music_title_vector = to_tensor(music_caption_dict['title_vector'])
            music_cover_vector = to_tensor(music_caption_dict['cover_vector'])
            if 'lyric_vector' in music_caption_dict:
                music_lyrics_vector = to_tensor(music_caption_dict['lyric_vector'])
            else:
                music_lyrics_vector = torch.zeros_like(music_title_vector)
            music_attribute_input_ids, music_caption_input_ids = process_text(tokenizer, music_caption_dict, args)
            all_music_ids.append(music_id)
            all_music_content_vector.append(music_content_vector)
            all_music_title_vector.append(music_title_vector)
            all_music_cover_vector.append(music_cover_vector)
            all_music_lyrics_vector.append(music_lyrics_vector)
            all_music_attribute_input_ids.append(music_attribute_input_ids)
            all_music_caption_input_ids.append(music_caption_input_ids)
    all_music_content_vector = torch.stack(all_music_content_vector, dim=0)
    all_music_title_vector = torch.stack(all_music_title_vector, dim=0)
    all_music_cover_vector = torch.stack(all_music_cover_vector, dim=0)
    all_music_lyrics_vector = torch.stack(all_music_lyrics_vector, dim=0)
    all_music_attribute_input_ids = torch.stack(all_music_attribute_input_ids, dim=0)
    all_music_caption_input_ids = torch.stack(all_music_caption_input_ids, dim=0)

    feature_cache = {
        'all_music_ids': all_music_ids,
        'all_music_content_vector': all_music_content_vector,
        'all_music_title_vector': all_music_title_vector,
        'all_music_cover_vector': all_music_cover_vector,
        'all_music_lyrics_vector': all_music_lyrics_vector,
        'all_music_attribute_input_ids': all_music_attribute_input_ids,
        'all_music_caption_input_ids': all_music_caption_input_ids,
    }
    feature_cache_file = os.path.join(output_dir, "/feature_cache")

    torch.save(feature_cache, feature_cache_file)

    batch_size = 2048
    all_music_ue_vector = []
    print(f"Starting to inference")
    for i in tqdm(range(0, len(all_music_content_vector), batch_size)):
        batch_music_content_vector = all_music_content_vector[i:i+batch_size].to(device)
        batch_music_cover_vector = all_music_cover_vector[i:i+batch_size].to(device)
        batch_music_title_vector = all_music_title_vector[i:i+batch_size].to(device)
        batch_music_lyrics_vector = all_music_lyrics_vector[i:i+batch_size].to(device)
        batch_music_attribute_input_ids = all_music_attribute_input_ids[i:i+batch_size].to(device)
        batch_music_caption_input_ids = all_music_caption_input_ids[i:i+batch_size].to(device)

        with torch.no_grad():
            batch_music_ue_vector = model.extract_music_embeds(batch_music_content_vector, batch_music_cover_vector, batch_music_attribute_input_ids, batch_music_caption_input_ids, batch_music_lyrics_vector, batch_music_title_vector)
            batch_music_ue_vector = batch_music_ue_vector.cpu()
        
        all_music_ue_vector.append(batch_music_ue_vector)

    all_music_ue_vector = torch.cat(all_music_ue_vector, dim=0)
    
    output_data = []
    for index in range(len(all_music_ids)):
        output_data.append({
            'music_id': all_music_ids[index],
            'music_ue_vector': all_music_ue_vector[index],
        })
    
    output_file = os.path.join(output_dir, "music_ue_vector.pt")
    torch.save(output_data, output_file)
    return len(output_data)

def extract_evalset_music_embeds_main_func(model_path, ckpt_path, output_dir, args):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model, processor, tokenizer = build_model(model_path, ckpt_path, args)
    model = model.eval()
    model = model.to(device)
    music_dict_file = "/mlx_devbox/users/wangxiuqi.0601/playground/data/evalset_1w_music_data.json"
    finished_num = calculate_evalset_music_vectors(music_dict_file, model, tokenizer, output_dir, device, args)
    
    print(f"Deal with {finished_num} music items finised")

def extract_music_embeds_main_func(model_path, ckpt_path, output_dir, args):
    init_distributed_mode(args)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda')
    rank = args.rank
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model, processor, tokenizer = build_model(model_path, ckpt_path, args)
    model = model.eval()
    model = model.to(device)
    music_library_dir = '/mnt/bn/search-ad-creative-zhijie/wangxiuqi.0601/mm/music_emb/music_library_0911_music_candidates_filter_train'
    music_vec_files = [os.path.join(music_library_dir, file) for file in os.listdir(music_library_dir) if file.endswith('.pt')]

    with open('/mnt/bn/search-ad-creative-zhijie/wangxiuqi.0601/mm/music_captions/music_clip_ids_1009_enriched_650w.pkl', 'rb') as fr:
        music_caption_file_dict = pickle.load(fr)
        print(f"Totally {len(music_caption_file_dict)} video music ids in meta info dict")
    print(f"Totally {len(music_vec_files)} music vec files")
    total_num = 0
    for index in range(len(music_vec_files)):
        if index % args.world_size == rank:
            print(f"Processing {index} file")
            finished_num = calculate_music_vectors(music_vec_files[index], model, tokenizer, output_dir, device, music_caption_file_dict, args)
            total_num += finished_num
    
    print(f"Deal with {total_num} music items finised")
    torch.distributed.barrier()


def extract_video_embeds(model_path, ckpt_path, output_dir, args):
    # The model config path
    demo_image_dir = '/mnt/bn/search-ad-creative-zhijie/wangxiuqi.0601/mm/MMpretrain'

    model, processor, tokenizer = build_model(model_path, ckpt_path, args)
    model = model.eval()
    device = torch.device('cuda')
    model = model.to(device)
    torch_dtype = torch.bfloat16 

    image_path_list = [
        'demo_cases/test_flight.jpeg',
        'demo_cases/test_flight_3.png',
        'demo_cases/test_food_3.jpeg',
        'demo_cases/test_food_4.png',
        'demo_cases/cat2.jpg',
        'demo_cases/scene.jpg', 
        'demo_cases/makeup.png',
        'demo_cases/climb.png',
        'demo_cases/food3.png',
        'demo_cases/animal_4.jpg',
        'demo_cases/bay_scene.jpg',
        'demo_cases/children.png',
        'demo_cases/plane2.webp',
        'demo_cases/moon_tea.webp',
        'demo_cases/hotel.jpg',
        'demo_cases/wedding.jpeg',
        'demo_cases/kids.jpg',
        'demo_cases/my_cat.jpg',
    ]

    for image_path in image_path_list:
        image_name = os.path.split(image_path)[-1].split('.')[0]
        image_path = os.path.join(demo_image_dir, image_path)
        pixel_values, pixel_masks, spatial_shapes, frame_mask = process_video(processor, image_path)
        pixel_values = pixel_values.unsqueeze(0)
        pixel_masks = pixel_masks.unsqueeze(0)
        spatial_shapes = spatial_shapes.unsqueeze(0)
        frame_mask = frame_mask.unsqueeze(0)

        user_languages = [17] # en
        user_countries = [233] # United States
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
            video_embed = model.extract_video_embeds(pixel_values.to(device), pixel_masks.to(device), spatial_shapes.to(device), frame_mask.to(device), user_languages, user_countries)
        if not os.path.exists(os.path.join(output_dir, 'demo_cases/')):
            os.makedirs(os.path.join(output_dir, 'demo_cases/'), exist_ok=True)
        with open(os.path.join(output_dir, f'demo_cases/{image_name}_en-US_vector.pkl'), 'wb') as fw:
            pickle.dump(video_embed.cpu(), fw)

    for image_path in image_path_list:
        image_name = os.path.split(image_path)[-1].split('.')[0]
        image_path = os.path.join(demo_image_dir, image_path)
        pixel_values, pixel_masks, spatial_shapes, frame_mask = process_video(processor, image_path)
        pixel_values = pixel_values.unsqueeze(0)
        pixel_masks = pixel_masks.unsqueeze(0)
        spatial_shapes = spatial_shapes.unsqueeze(0)
        frame_mask = frame_mask.unsqueeze(0)

        user_languages = [80] # th
        user_countries = [217] # Thailand
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
            video_embed = model.extract_video_embeds(pixel_values.to(device), pixel_masks.to(device), spatial_shapes.to(device), frame_mask.to(device), user_languages, user_countries)
        if not os.path.exists(os.path.join(output_dir, 'demo_cases/')):
            os.makedirs(os.path.join(output_dir, 'demo_cases/'), exist_ok=True)
        with open(os.path.join(output_dir, f'demo_cases/{image_name}_th-TH_vector.pkl'), 'wb') as fw:
            pickle.dump(video_embed.cpu(), fw)

    for image_path in image_path_list:
        image_name = os.path.split(image_path)[-1].split('.')[0]
        image_path = os.path.join(demo_image_dir, image_path)
        pixel_values, pixel_masks, spatial_shapes, frame_mask = process_video(processor, image_path)
        pixel_values = pixel_values.unsqueeze(0)
        pixel_masks = pixel_masks.unsqueeze(0)
        spatial_shapes = spatial_shapes.unsqueeze(0)
        frame_mask = frame_mask.unsqueeze(0)
        user_languages = [4] # ar
        user_countries = [231] # United Arab Emirates
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
            video_embed = model.extract_video_embeds(pixel_values.to(device), pixel_masks.to(device), spatial_shapes.to(device), frame_mask.to(device), user_languages, user_countries)
        if not os.path.exists(os.path.join(output_dir, 'demo_cases/')):
            os.makedirs(os.path.join(output_dir, 'demo_cases/'), exist_ok=True)
        with open(os.path.join(output_dir, f'demo_cases/{image_name}_ar-AR_vector.pkl'), 'wb') as fw:
            pickle.dump(video_embed.cpu(), fw)

    for image_path in image_path_list:
        image_name = os.path.split(image_path)[-1].split('.')[0]
        image_path = os.path.join(demo_image_dir, image_path)
        pixel_values, pixel_masks, spatial_shapes, frame_mask = process_video(processor, image_path)
        pixel_values = pixel_values.unsqueeze(0)
        pixel_masks = pixel_masks.unsqueeze(0)
        spatial_shapes = spatial_shapes.unsqueeze(0)
        frame_mask = frame_mask.unsqueeze(0)
        user_languages = [40] # ko
        user_countries = [172] # Republic of Korea
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
            video_embed = model.extract_video_embeds(pixel_values.to(device), pixel_masks.to(device), spatial_shapes.to(device), frame_mask.to(device), user_languages, user_countries)
        if not os.path.exists(os.path.join(output_dir, 'demo_cases/')):
            os.makedirs(os.path.join(output_dir, 'demo_cases/'), exist_ok=True)
        with open(os.path.join(output_dir, f'demo_cases/{image_name}_ko-KR_vector.pkl'), 'wb') as fw:
            pickle.dump(video_embed.cpu(), fw)

def extract_evalset_video_embeds(model_path, ckpt_path, output_dir, args):
    # The model config path
    video_evalset_path = "/mnt/bn/tt-search-ads-nas/wangxiuqi.0601/data/evalset_1k_video_data.json"

    model, processor, tokenizer = build_model(model_path, ckpt_path, args)
    model = model.eval()
    device = torch.device('cuda')
    model = model.to(device)
    torch_dtype = torch.bfloat16 
    evalset_item_id_to_emb = {}

    evalset_country_and_lang_path = "/mnt/bn/tt-search-ads-nas/wangxiuqi.0601/data/evalset_all_item_and_country_language.tsv"
    lang2idx = {'ID': 0, 'None': 1, 'US': 2, 'Zh-Hans': 3, 'ae': 4, 'am': 5, 'ar': 6, 'arq': 7, 'au': 8, 'az': 9, 'be': 10, 'bg': 11, 'bn': 12, 'bn-IN': 13, 'ca': 14, 'ceb': 15, 'cs': 16, 'cs-CZ': 17, 'da': 18, 'de': 19, 'de-DE': 20, 'el': 21, 'el-GR': 22, 'en': 23, 'en-GB': 24, 'en-US': 25, 'es': 26, 'es-419': 27, 'et': 28, 'fi': 29, 'fi-FI': 30, 'fil': 31, 'fr': 32, 'fr-CA': 33, 'ga': 34, 'he': 35, 'he-IL': 36, 'hi': 37, 'hr': 38, 'hu': 39, 'hu-HU': 40, 'id': 41, 'id-ID': 42, 'in': 43, 'is': 44, 'it': 45, 'it-IT': 46, 'ja': 47, 'ja-JP': 48, 'jv': 49, 'jv-ID': 50, 'ka': 51, 'kk': 52, 'km': 53, 'km-KH': 54, 'ko': 55, 'ko-KR': 56, 'kr': 57, 'lt': 58, 'lv': 59, 'ml': 60, 'mn': 61, 'ms': 62, 'ms-MY': 63, 'my': 64, 'my-MM': 65, 'nb': 66, 'ne': 67, 'nl': 68, 'nl-NL': 69, 'pl': 70, 'pl-PL': 71, 'pt': 72, 'pt-BR': 73, 'ro': 74, 'ro-RO': 75, 'ru': 76, 'ru-RU': 77, 'si': 78, 'sk': 79, 'sl': 80, 'sq': 81, 'sv': 82, 'sv-SE': 83, 'sw': 84, 'th': 85, 'th-TH': 86, 'tr': 87, 'tr-TR': 88, 'ua': 89, 'uk': 90, 'uk-UA': 91, 'ur': 92, 'uz': 93, 'vi': 94, 'vi-VN': 95, 'zh': 96, 'zh-Hans': 97, 'zh-Hant': 98, 'zh-Hant-TW': 99}
    country2idx = {'AD': 0, 'AE': 1, 'AF': 2, 'AG': 3, 'AI': 4, 'AL': 5, 'AM': 6, 'AO': 7, 'AQ': 8, 'AR': 9, 'AS': 10, 'AT': 11, 'AU': 12, 'AW': 13, 'AX': 14, 'AZ': 15, 'BA': 16, 'BB': 17, 'BD': 18, 'BE': 19, 'BF': 20, 'BG': 21, 'BH': 22, 'BI': 23, 'BJ': 24, 'BL': 25, 'BM': 26, 'BN': 27, 'BO': 28, 'BQ': 29, 'BR': 30, 'BS': 31, 'BT': 32, 'BW': 33, 'BY': 34, 'BZ': 35, 'CA': 36, 'CD': 37, 'CF': 38, 'CG': 39, 'CH': 40, 'CI': 41, 'CK': 42, 'CL': 43, 'CM': 44, 'CN': 45, 'CO': 46, 'CR': 47, 'CU': 48, 'CV': 49, 'CW': 50, 'CY': 51, 'CZ': 52, 'DE': 53, 'DJ': 54, 'DK': 55, 'DM': 56, 'DO': 57, 'DZ': 58, 'EC': 59, 'EE': 60, 'EG': 61, 'EH': 62, 'ER': 63, 'ES': 64, 'ET': 65, 'FI': 66, 'FJ': 67, 'FK': 68, 'FM': 69, 'FO': 70, 'FR': 71, 'GA': 72, 'GB': 73, 'GD': 74, 'GE': 75, 'GF': 76, 'GG': 77, 'GH': 78, 'GI': 79, 'GL': 80, 'GM': 81, 'GN': 82, 'GP': 83, 'GQ': 84, 'GR': 85, 'GT': 86, 'GU': 87, 'GW': 88, 'GY': 89, 'HK': 90, 'HN': 91, 'HR': 92, 'HT': 93, 'HU': 94, 'ID': 95, 'IE': 96, 'IL': 97, 'IM': 98, 'IN': 99, 'IO': 100, 'IQ': 101, 'IR': 102, 'IS': 103, 'IT': 104, 'JE': 105, 'JM': 106, 'JO': 107, 'JP': 108, 'KE': 109, 'KG': 110, 'KH': 111, 'KI': 112, 'KM': 113, 'KN': 114, 'KR': 115, 'KW': 116, 'KY': 117, 'KZ': 118, 'LA': 119, 'LB': 120, 'LC': 121, 'LI': 122, 'LK': 123, 'LR': 124, 'LS': 125, 'LT': 126, 'LU': 127, 'LV': 128, 'LY': 129, 'MA': 130, 'MC': 131, 'MD': 132, 'ME': 133, 'MF': 134, 'MG': 135, 'MH': 136, 'MK': 137, 'ML': 138, 'MM': 139, 'MN': 140, 'MO': 141, 'MP': 142, 'MQ': 143, 'MR': 144, 'MS': 145, 'MT': 146, 'MU': 147, 'MV': 148, 'MW': 149, 'MX': 150, 'MY': 151, 'MZ': 152, 'NA': 153, 'NC': 154, 'NE': 155, 'NF': 156, 'NG': 157, 'NI': 158, 'NL': 159, 'NO': 160, 'NP': 161, 'NR': 162, 'NU': 163, 'NZ': 164, 'None': 165, 'OM': 166, 'PA': 167, 'PE': 168, 'PF': 169, 'PG': 170, 'PH': 171, 'PK': 172, 'PL': 173, 'PM': 174, 'PR': 175, 'PS': 176, 'PT': 177, 'PW': 178, 'PY': 179, 'QA': 180, 'RE': 181, 'RO': 182, 'RS': 183, 'RU': 184, 'RW': 185, 'SA': 186, 'SB': 187, 'SC': 188, 'SD': 189, 'SE': 190, 'SG': 191, 'SH': 192, 'SI': 193, 'SJ': 194, 'SK': 195, 'SL': 196, 'SM': 197, 'SN': 198, 'SO': 199, 'SR': 200, 'SS': 201, 'ST': 202, 'SV': 203, 'SX': 204, 'SY': 205, 'SZ': 206, 'TC': 207, 'TD': 208, 'TG': 209, 'TH': 210, 'TJ': 211, 'TK': 212, 'TL': 213, 'TM': 214, 'TN': 215, 'TO': 216, 'TR': 217, 'TT': 218, 'TV': 219, 'TW': 220, 'TZ': 221, 'UA': 222, 'UG': 223, 'UM': 224, 'US': 225, 'UY': 226, 'UZ': 227, 'VC': 228, 'VE': 229, 'VG': 230, 'VI': 231, 'VN': 232, 'VU': 233, 'WF': 234, 'WS': 235, 'XK': 236, 'YE': 237, 'YT': 238, 'ZA': 239, 'ZM': 240, 'ZW': 241}
    item_id2lang_code = defaultdict(lambda: lang2idx['None'])
    item_id2country_code = defaultdict(lambda: country2idx['None'])
    with open(evalset_country_and_lang_path, 'r') as f:
        for line in f:
            lines = line.strip().split("\t")
            item_id = int(lines[0])
            country_code = lines[1]
            lang_code = lines[2]
            item_id2lang_code[item_id] = lang2idx[lang_code] if lang_code in lang2idx else lang2idx['None']
            item_id2country_code[item_id] = country2idx[country_code] if country_code in country2idx else country2idx['None']
    print(f"Totally {len(item_id2lang_code)} item_id2lang_code")
    print(f"Totally {len(item_id2country_code)} item_id2country_code")

    with open(video_evalset_path, 'r') as f:
        for line in f:
            line = line.strip()
            data = json.loads(line)
            item_id = data['item_id']
            frames = data['frames']
            pixel_values, pixel_masks, spatial_shapes, frame_mask = process_evalset_video(processor, frames)
            pixel_values = pixel_values.unsqueeze(0)
            pixel_masks = pixel_masks.unsqueeze(0)
            spatial_shapes = spatial_shapes.unsqueeze(0)
            frame_mask = frame_mask.unsqueeze(0)
            user_languages = [item_id2lang_code[item_id]]
            user_countries = [item_id2country_code[country_code]]
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
                video_embed = model.extract_video_embeds(pixel_values.to(device), pixel_masks.to(device), spatial_shapes.to(device), frame_mask.to(device), user_languages, user_countries)
            evalset_item_id_to_emb[item_id] = video_embed.cpu()
    with open(os.path.join(output_dir, 'evalset_1000_image_embs.pkl'), 'wb') as fw:
        pickle.dump(evalset_item_id_to_emb, fw)
    print("len evalset_item_id_to_emb={}".format(len(evalset_item_id_to_emb)))


if __name__ == '__main__':
    args = get_args()
    model_path = "/mnt/bn/tt-search-ads-nas/wangxiuqi.0601/huggingface/siglip2-base"
    ckpt_path =   os.path.join(args.data_path, "pytorch_model.bin")
    output_dir =  os.path.join(args.data_path, "evalset_whole_data/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # extract_evalset_video_embeds(model_path, ckpt_path, output_dir, args)
    # extract_evalset_music_embeds_main_func(model_path, ckpt_path, output_dir, args)

    # extract_video_embeds(model_path, ckpt_path, output_dir, args)
    extract_music_embeds_main_func(model_path, ckpt_path, output_dir, args)
