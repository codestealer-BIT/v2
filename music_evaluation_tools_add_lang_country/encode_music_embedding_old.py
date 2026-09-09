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
import ast
import jsonlines
from tqdm import tqdm
from trainer_misc import init_distributed_mode
from config import cfg
from IPython import embed
from video_clip import VideoSigLIP2ForMusic, VideoSigLIP2ForMusicOnlyMatching, VideoSigLIP2ForMusicOnlyMatchingAddUserInfo, VideoSigLIP2ForMusicNegativeFilteringAddLang, VideoSigLIP2ForMusicNegativeFilteringBMAddLang, VideoSigLIP2ForMusicOnlyMatchingV2
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
    parser.add_argument('--add_user_lang_and_country_method', default="residual", type=str)
    parser.add_argument('--user_language_code', default=1, type=int, help="The user language code for checking cases, e.g. 1:en")
    parser.add_argument('--user_country_code', default=1, type=int, help="The user country code for checking cases, e.g. 1:US")
    parser.add_argument('--embed_lang_caption', default=False, type=str2bool)
    return parser.parse_args()


def build_model(model_path, pretrained_ckpt_path, args):
    cfg_path = '/mnt/bn/jiny-ttls-i18n-fr1q/MM-Embedding/experiments/config_finetune_mmembed_v3.yaml'

    if cfg_path:
        cfg.update_cfg(cfg_path)
    
    # model = VideoSigLIP2ForMusicNegativeFilteringAddLang(
    #     model_path,
    #     gpuwise_nce=True,
    #     interpolate=6,
    #     use_frame_mask=True,
    #     # add_lyrics_ue=args.add_lyrics_ue,
    #     # add_title_ue=args.add_title_ue,
    #     user_info_fusion_method=args.add_user_lang_and_country_method,
    #     language_add_constant=0.0,
    # )

    model = VideoSigLIP2ForMusicOnlyMatching(
            model_path,
            gpuwise_nce=True,
            interpolate=6,
            use_frame_mask = True,
            add_user_lang_and_country_code= False,
    )

    # model = VideoSigLIP2ForMusicOnlyMatchingV2(
    #         model_path,
    #         gpuwise_nce=True,
    #         interpolate=6,
    #         use_frame_mask = True,
    #         add_user_lang_and_country_code= True,
    # )

    # model = VideoSigLIP2ForMusicNegativeFilteringBMAddLang(
    #     model_path,
    #     gpuwise_nce=True,
    #     interpolate=6,
    #     use_frame_mask=True,
    #     # add_lyrics_ue=args.add_lyrics_ue,
    #     # add_title_ue=args.add_title_ue,
    #     user_info_fusion_method=args.add_user_lang_and_country_method,
    # )

    

    tokenizer = AutoTokenizer.from_pretrained(model_path) 
    processor = Siglip2ImageProcessorFast()

    if pretrained_ckpt_path:
        print(f"Loading the pre-trained checkpoint from {pretrained_ckpt_path}")
        pretrained_checkpoint = torch.load(pretrained_ckpt_path, map_location='cpu')
        load_res = model.load_state_dict(pretrained_checkpoint, strict=True)
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
        
        is_pgc_flag = int(music_caption_dict.get('is_pgc', 0))
        has_meta_song = music_caption_dict.get('meta_song_id', None) is not None
        if is_pgc_flag == 1:
            music_dist_code = 2  # PGC
        else:
            if has_meta_song:
                music_dist_code = 1  # PUGC
            else:
                music_dist_code = 0  # UGC
        pgc_map = {0: 'UGC', 1: 'PUGC', 2: 'PGC'}
        
        # music_attribute = f"genre: {music_genre}; language: {music_language}; theme: {music_theme}; mood: {music_mood}"
        music_attribute = f"language: {music_language}; genre: {music_genre}; theme: {music_theme}; mood: {music_mood}; music source: {pgc_map[music_dist_code]}"
        if args.add_title_text:
            if has_meta_song: # only pgc/pugc
                music_attribute = f"language: {music_language}; title: {music_title}; genre: {music_genre}; theme: {music_theme}; mood: {music_mood}; music source: {pgc_map[music_dist_code]}"
        
        if "publish_180d" in music_caption_dict:
            music_attribute += f"; music popularity: {music_caption_dict['publish_180d']}"

        if args.embed_lang_caption:
            music_capton = f"这是一首{music_language}语言的歌曲。 {music_capton}"

    music_attribute_input_ids = tokenizer(music_attribute, padding='max_length', truncation=True, max_length=64, return_attention_mask=False, return_tensors="pt").input_ids[0]
    music_caption_input_ids = tokenizer(music_capton, padding='max_length', truncation=True, max_length=284, return_attention_mask=False, return_tensors="pt").input_ids[0]
    
    return music_attribute_input_ids, music_caption_input_ids


def process_video(processor, image_path=None, frames=None, rescale_img=True):
    max_frame_len = 5
    tokens_per_frame = 32   # CANNOT MODIFY
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
        video = [image]
    elif frames is not None:
        video = [Image.open(io.BytesIO(base64.b64decode(frame))).convert("RGB") for frame in frames]
    else:
        raise ValueError("Either image_path or frames must be provided")
    
    if rescale_img: # rescale img to max 256
        rescaled_video = []
        for frame in video:
            w, h = frame.size
            if w > h:
                new_w = 256
                new_h = int(h * (256 / w))
            else:
                new_h = 256
                new_w = int(w * (256 / h))
            rescaled_frame = frame.resize((new_w, new_h), Image.LANCZOS)
            rescaled_video.append(rescaled_frame)
        video = rescaled_video

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
    # cache_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/music_library_0911_music_candidates_filter_train/feature_cache'
    cache_dir  = '/mnt/bn/jiny-ttls-i18n-fr1q/lutong/music_input_cache/lang_in_caption_pgc_populairity'
    # cache_dir  = '/mnt/bn/jiny-ttls-i18n-fr1q/lutong/music_input_cache/lang_in_caption_pgc_populairity_filtered'
    feature_cache_file = os.path.join(cache_dir, filename)

    if os.path.exists(feature_cache_file):
        feature_cache = torch.load(feature_cache_file)
        all_music_ids = feature_cache['all_music_ids']
        all_music_content_vector = feature_cache['all_music_content_vector']
        all_music_title_vector = feature_cache['all_music_title_vector']
        all_music_cover_vector = feature_cache['all_music_cover_vector']
        all_music_attribute_input_ids = feature_cache['all_music_attribute_input_ids']
        all_music_caption_input_ids = feature_cache['all_music_caption_input_ids']
        all_music_dist_codes = feature_cache['all_music_dist_codes']

    else:
        all_music_ids = []
        all_music_content_vector, all_music_title_vector, all_music_cover_vector = [], [], []
        all_music_attribute_input_ids, all_music_caption_input_ids = [], []
        all_music_dist_codes = []
        
        print(f"Processing the input features")
        for item in tqdm(music_tensor_list):
            music_id = item['music_id']
            music_caption_file = music_caption_file_dict[music_id]

            if os.path.exists(music_caption_file):
                with open(music_caption_file, 'r') as fr:
                    music_caption_dict = json.load(fr)
            else:
                music_caption_dict = None
            if not music_caption_dict or music_caption_dict.get('content_vector', None) is None or music_caption_dict.get('content', None) is None:
                continue

            # lang_in_caption = any([lang in music_caption_dict.get('content', '') for lang in ["念白", "对话", "口白", "口播", "独白","诵读", "音频", "旁白", "广播"]])
            # if lang_in_caption:
            #     continue      
            
            is_pgc_flag = int(music_caption_dict.get('is_pgc', 0))
            has_meta_song = music_caption_dict.get('meta_song_id', None) is not None
            music_dist_code = 2 if is_pgc_flag == 1 else (1 if has_meta_song else 0)

            music_attribute_input_ids, music_caption_input_ids = process_text(tokenizer, music_caption_dict, args)
            all_music_dist_codes.append(music_dist_code)

            all_music_ids.append(music_id)
            all_music_content_vector.append(to_tensor(music_caption_dict.get('content_vector')))
            all_music_title_vector.append(to_tensor(music_caption_dict.get('title_vector')))
            all_music_cover_vector.append(to_tensor(music_caption_dict.get('cover_vector')))
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
            'all_music_dist_codes': all_music_dist_codes,
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
        batch_music_dist_codes = all_music_dist_codes[i:i+batch_size]

        with torch.no_grad():
            batch_music_ue_vector = model.extract_music_embeds(batch_music_content_vector, batch_music_cover_vector, batch_music_attribute_input_ids, batch_music_caption_input_ids, batch_music_dist_codes)
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

def get_ue_vector(vector_name, music_caption_dict, dim=128):
    if vector_name in music_caption_dict:
        vector = to_tensor(music_caption_dict[vector_name])
    else:
        vector = torch.zeros(dim)
    return vector

def calculate_evalset_music_vectors(music_dict_file, model, tokenizer, output_dir, device, args):
    all_music_ids = []
    all_music_content_vector, all_music_title_vector, all_music_cover_vector, all_music_lyrics_vector = [], [], [], []
    all_music_attribute_input_ids, all_music_caption_input_ids = [], []
    all_music_dist_codes = []
    with open(music_dict_file, 'r') as f:
        for line in f:
            music_caption_dict = json.loads(line.strip())
            music_id = music_caption_dict['clip_id']
            if 'content_vector' not in music_caption_dict:
                continue
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
            all_music_dist_codes.append(2 if int(music_caption_dict.get('is_pgc', 0)) == 1 else (1 if music_caption_dict.get('meta_song_id', None) is not None else 0))
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
        'all_music_dist_codes': all_music_dist_codes,
    }
    feature_cache_file = os.path.join(output_dir, "feature_cache")

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
        batch_music_dist_codes = all_music_dist_codes[i:i+batch_size]

        with torch.no_grad():
            batch_music_ue_vector = model.extract_music_embeds(batch_music_content_vector, batch_music_cover_vector, batch_music_attribute_input_ids, batch_music_caption_input_ids, batch_music_dist_codes)
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
    "generate 1w music embeddings for evalset"
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model, processor, tokenizer = build_model(model_path, ckpt_path, args)
    model = model.eval()
    model = model.to(device)
    # music_dict_file = "/mlx_devbox/users/wangxiuqi.0601/playground/data/evalset_1w_music_data.json"
    music_dict_file = "/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/evalset_1w_music_data.jsonl"
    # music_dict_file = "/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/evalset_1w_natural_popularity_music_data.jsonl"
    print(f"music_dict_file: {music_dict_file}")
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
    # music_library_dir = '/mnt/bn/search-ad-creative-zhijie/wangxiuqi.0601/mm/music_emb/music_library_0911_music_candidates_filter_train'
    music_library_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/music_library_0911_music_candidates_filter_train'
    music_vec_files = [os.path.join(music_library_dir, file) for file in os.listdir(music_library_dir) if file.endswith('.pt')]

    # with open('/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/music_captions/music_clip_ids_1009_enriched_650w.pkl', 'rb') as fr:
    with open('/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/music_clip_ids_1009_enriched_650w_final_dedup.pkl', 'rb') as fr:
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


def extract_video_embeds_cases(ckpt_path, user_language_code, user_country_code):
    print(f"Start to extract video embeds for demo cases, user_language={user_language_code}, user_country={user_country_code}")
    # The model config path
    model_path = "/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"
    # demo_image_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/MMPretrain'
    demo_image_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/'

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
        'demo_cases/argument.png',
        'demo_cases/kpop.jpg',
        'demo_cases/Thanksgiving.png',
        'demo_cases/sad.jpg',
        'demo_cases/Fish_and_chips.jpg',
        'demo_cases/car.jpeg',
        'demo_cases/celeberity.jpg',
        'demo_cases/workout_man.jpg',
        'demo_cases/zumba-dance.jpg',
        'demo_cases/diy_cup.png',
        'demo_cases/cars_vintage.jpg',
        'demo_cases/Xiaomi_SU7.jpg',
        'demo_cases/football.jpg',
        'demo_cases/selfie_mid_woman.png',
        'demo_cases/selfie_caucasion.png',
        'demo_cases/aunties_fashion.png',
        'demo_cases/outfit_mid_man.jpg',
        'demo_cases/tiananmen.jpg',
        'demo_cases/winter.jpg',
        'demo_cases/fishing.jpg',
        'demo_cases/sushi.jpg',
        'demo_cases/christmas.jpg',
        # rescaled
        'demo_cases/fishing_rescaled.jpeg',
        'demo_cases/winter_rescaled.jpeg',
        'demo_cases/cat2_rescaled.jpeg',
    ]
    # image_path_list = [
    #     "demo_cases/junqi_food.webp"
    # ]

    for image_path in image_path_list:
        image_name = os.path.split(image_path)[-1].split('.')[0]
        image_path = os.path.join(demo_image_dir, image_path)
        pixel_values, pixel_masks, spatial_shapes, frame_mask = process_video(processor, image_path=image_path)
        pixel_values = pixel_values.unsqueeze(0)
        pixel_masks = pixel_masks.unsqueeze(0)
        spatial_shapes = spatial_shapes.unsqueeze(0)
        frame_mask = frame_mask.unsqueeze(0)
        user_languages = [user_language_code] # manually assign the case a languange
        user_countries = [user_country_code]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
            video_embed = model.extract_video_embeds(pixel_values.to(device), pixel_masks.to(device), spatial_shapes.to(device), frame_mask.to(device), user_languages, user_countries)
        with open(os.path.join(demo_image_dir, f'demo_cases/{image_name}_vector.pkl'), 'wb') as fw:
            pickle.dump(video_embed.cpu(), fw)

def extract_evalset_video_embeds(model_path, ckpt_path, output_dir, args):
    "Generate video embeddings for 1k evalset videos"
    # The model config path
    # video_evalset_path = "/mnt/bn/tt-search-ads-nas/wangxiuqi.0601/data/evalset_1k_video_data.json"
    video_evalset_path = "/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/evalset_1k_video_data.jsonl"

    model, processor, tokenizer = build_model(model_path, ckpt_path, args)
    model = model.eval()
    device = torch.device('cuda')
    model = model.to(device)
    torch_dtype = torch.bfloat16 
    evalset_item_id_to_emb = {}
    
    # method 1: use all langs and countries
    lang2idx = defaultdict(lambda: 0)        
    country2idx = defaultdict(lambda: 0)

    with open('/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/train_country2idx.jsonl', 'r') as fr:
        for line in fr:
            value = json.loads(line.strip())
            country2idx[value['key']] = value['idx']
    with open('/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/train_lang2idx.jsonl', 'r') as fr:
        for line in fr:
            value = json.loads(line.strip())
            lang2idx[value['key']] = value['idx']
    
    # # method 2: only use top20 lang and countries
    # lang2idx = {"None":0, "en": 1, "es": 2, "ar": 3, "pt": 4, "fr": 5, "ru": 6, "id": 7, "tr": 8, "th": 9, "vi": 10, "de": 11, "ja": 12, "it": 13, "uk": 14, "ro": 15, "pl": 16, "az": 17, "ko": 18, "ms": 19, "zh-Hant": 20}
    # lang2idx = defaultdict(lambda: lang2idx['None'], lang2idx)
    
    # country2idx = {'None': 0, 'US': 1, 'PK': 2, 'BR': 3, 'ID': 4, 'MX': 5, 'TR': 6, 'TH': 7, 'GB': 8, 'VN': 9, 'BD': 10, 'IQ': 11, 'DE': 12, 'FR': 13, 'NP': 14, 'MY': 15, 'SA': 16, 'IT': 17, 'JP': 18, 'UA': 19, 'ES': 20}
    # country2idx = defaultdict(lambda: country2idx['None'], country2idx)

    print(f"Totally {len(country2idx)} country2idx")
    print(f"Totally {len(lang2idx)} lang2idx")

    evalset_country_and_lang_path = "/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/evalset_all_item_and_country_language.tsv"
    
    item_id2lang_code = {}
    item_id2country_code = {}
    with open(evalset_country_and_lang_path, 'r') as f:
        for line in f:
            lines = line.strip().split("\t")
            item_id = int(lines[0])
            country_code = lines[2]
            lang_code = lines[3]
            item_id2lang_code[item_id] = lang2idx[lang_code]
            item_id2country_code[item_id] = country2idx[country_code]

    with open(video_evalset_path, 'r') as f:
        for line in f:
            line = line.strip()
            data = json.loads(line)
            item_id = data['item_id']
            frames = data['frames']
            pixel_values, pixel_masks, spatial_shapes, frame_mask = process_video(processor, frames=frames)
            pixel_values = pixel_values.unsqueeze(0)
            pixel_masks = pixel_masks.unsqueeze(0)
            spatial_shapes = spatial_shapes.unsqueeze(0)
            frame_mask = frame_mask.unsqueeze(0)
            user_languages = [item_id2lang_code[item_id]]
            user_countries = [item_id2country_code[item_id]]
            # print("item_id", item_id, "user_languages", user_languages, "user_countries", user_countries)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
                video_embed = model.extract_video_embeds(pixel_values.to(device), pixel_masks.to(device), spatial_shapes.to(device), frame_mask.to(device), user_languages, user_countries)
            evalset_item_id_to_emb[item_id] = video_embed.cpu()
    with open(os.path.join(output_dir, 'evalset_1000_image_embs.pkl'), 'wb') as fw:
        pickle.dump(evalset_item_id_to_emb, fw)
    print("len evalset_item_id_to_emb={}".format(len(evalset_item_id_to_emb)))


if __name__ == '__main__':
    args = get_args()
    print(f"args:{args}")
    model_path = "/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"
    ckpt_path =   os.path.join(args.data_path, "pytorch_model.bin")
    print(f"ckpt_path={ckpt_path}")

    # # 1. evalset video & music
    # output_dir =  os.path.join(args.data_path, "evalset_1w_data/")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)

    # print("[IMPORTANT] generating video embeddings...")
    # extract_evalset_video_embeds(model_path, ckpt_path, output_dir, args)
    # print("[IMPORTANT] generating music embeddings...")
    # extract_evalset_music_embeds_main_func(model_path, ckpt_path, output_dir, args)

    # # 2. 650w music
    # output_dir =  os.path.join(args.data_path, "650w_music_filtered")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    # extract_music_embeds_main_func(model_path, ckpt_path, output_dir, args)
    
    # 3. check cases
    extract_video_embeds_cases(ckpt_path, args.user_language_code, args.user_country_code)