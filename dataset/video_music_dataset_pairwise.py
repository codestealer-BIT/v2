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
from collections import defaultdict

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


language_mapping = {'aa': 'Afar', 'ab': 'Abkhazian', 'af': 'Afrikaans', 'ak': 'Akan', 'am': 'Amharic', 'ar': 'Arabic', 'an': 'Aragonese', 'as': 'Assamese', 'av': 'Avaric', 'ay': 'Aymara', 'az': 'Azerbaijani', 'ba': 'Bashkir', 'be': 'Belarusian', 'bg': 'Bulgarian', 'bh': 'Bihari', 'bi': 'Bislama', 'bm': 'Bambara', 'bn': 'Bengali', 'bo': 'Tibetan', 'br': 'Breton', 'bs': 'Bosnian', 'ca': 'Catalan', 'ce': 'Chechen', 'ch': 'Chamorro', 'co': 'Corsican', 'cr': 'Cree', 'cs': 'Czech', 'cu': 'Old Church Slavonic', 'cv': 'Chuvash', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'dv': 'Divehi', 'dz': 'Dzongkha', 'ee': 'Ewe', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'ha': 'Hausa', 'he': 'Hebrew', 'hi': 'Hindi', 'ho': 'Hiri Motu', 'hr': 'Croatian', 'ht': 'Haitian', 'hu': 'Hungarian', 'hy': 'Armenian', 'hz': 'Herero', 'ia': 'Interlingua', 'lt': 'Lithuanian', 'lu': 'Luba-Katanga', 'lv': 'Latvian', 'mg': 'Malagasy', 'mh': 'Marshallese', 'mi': 'Maori', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mr': 'Marathi', 'ms': 'Malay', 'mt': 'Maltese', 'my': 'Burmese', 'nv': 'Navajo', 'ny': 'Chichewa', 'oc': 'Occitan', 'oj': 'Ojibwa', 'om': 'Oromo', 'or': 'Oriya', 'os': 'Ossetian', 'pa': 'Punjabi', 'pi': 'Pali', 'pl': 'Polish', 'ps': 'Pushto', 'pt': 'Portuguese', 'qu': 'Quechua', 'rm': 'Romansh', 'sc': 'Sardinian', 'sd': 'Sindhi', 'se': 'Northern Sami', 'sg': 'Sango', 'tl': 'Tagalog', 'tn': 'Tswana', 'to': 'Tonga', 'tr': 'Turkish', 'ts': 'Tsonga', 'tt': 'Tatar', 'tw': 'Twi', 'ty': 'Tahitian', 'ug': 'Uighur', 'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek', 've': 'Venda', 'yi': 'Yiddish', 'yo': 'Yoruba', 'za': 'Zhuang', 'zh': 'Chinese', 'zu': 'Zulu', 'fa': 'Persian', 'ff': 'Fulah', 'fi': 'Finnish', 'fj': 'Fijian', 'st': 'Southern Sotho', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'id': 'Indonesian', 'ie': 'Interlingue', 'ig': 'Igbo', 'ii': 'Sichuan Yi', 'ik': 'Inupiaq', 'io': 'Ido', 'is': 'Icelandic', 'it': 'Italian', 'iu': 'Inuktitut', 'fo': 'Faroese', 'fr': 'French', 'fy': 'Western Frisian', 'ga': 'Irish', 'gd': 'Scottish Gaelic', 'gl': 'Galician', 'gn': 'Guarani', 'gu': 'Gujarati', 'gv': 'Manx', 'na': 'Nauru', 'nb': 'Norwegian Bokmål', 'nd': 'North Ndebele', 'ne': 'Nepali', 'ng': 'Ndonga', 'ja': 'Japanese', 'jv': 'Javanese', 'ka': 'Georgian', 'kg': 'Kongo', 'nl': 'Dutch', 'nn': 'Norwegian Nynorsk', 'no': 'Norwegian', 'nr': 'South Ndebele', 'ki': 'Kikuyu', 'kj': 'Kuanyama', 'kk': 'Kazakh', 'kl': 'Kalaallisut', 'km': 'Central Khmer', 'rn': 'Rundi', 'ro': 'Romanian', 'ru': 'Russian', 'rw': 'Kinyarwanda', 'sa': 'Sanskrit', 'kn': 'Kannada', 'ko': 'Korean', 'kr': 'Kanuri', 'ks': 'Kashmiri', 'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian', 'sm': 'Samoan', 'sn': 'Shona', 'ku': 'Kurdish', 'kv': 'Komi', 'kw': 'Cornish', 'ky': 'Kirghiz', 'la': 'Latin', 'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian', 'ss': 'Swati', 'lb': 'Luxembourgish', 'lg': 'Ganda', 'li': 'Limburgan', 'ln': 'Lingala', 'lo': 'Lao', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'ti': 'Tigrinya', 'tk': 'Turkmen', 'vi': 'Vietnamese', 'vo': 'Volapük', 'wa': 'Walloon', 'wo': 'Wolof', 'xh': 'Xhosa'}


class VideoMusicPairwiseHDFSDataset(IterableDataset):
    def __init__(
            self, 
            data_path, 
            rank: int = 0,
            world_size: int = 1,
            shuffle: bool = False,
            repeat: bool = False,
            verbose: bool = True,
            buffer_size: int = -1,
            max_frame_len: int = 8,
            max_text_len: int = 284,
            max_short_len: int = 64,
            tokens_per_frame: int = 32,
            random_drop_frame: bool = False,
            random_rotate_video: bool = False,
            filter_video_id_file: str = None,
            tokenizer_dir=None,
            add_title_text: bool = False,
            embed_lang_caption: bool = False,
            add_user_lang_and_country_code: bool = False,
            allowed_music_selected_from: str = "",
            force_rescale_prob: float = 0.0,
            filter_music_id_file: str = None,
            pairwise_score_thresh: float = 2.0,
            num_of_pair_per_ins: int = 800,
        ):
        super(VideoMusicPairwiseHDFSDataset).__init__()
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size

        with open(data_path, 'r') as fr:
            self.files = json.load(fr)

        self.files = [f for f in self.files if f.find('_SUCCESS') < 0 and f.find("_temporary") < 0 and f.find(".caption") < 0]
        self.files.sort()
        print(f"Totally {len(self.files)} video annotation files")
        if len(self.files) % self.world_size != 0:
            print('[DATA]--Whole dataset file num %s cannot split to worldsize %s ' %
                     (len(self.files), self.world_size))
        self.verbose = verbose
        self.repeat = repeat
        self.buffer = []
        self.buffer_size = buffer_size

        print(f"Using the buffer size {self.buffer_size}")

        self.max_frame_len = max_frame_len
        self.max_text_len = max_text_len #long text length
        self.max_short_len = max_short_len #short text length
        self.tokens_per_frame = tokens_per_frame

        self.random_drop_frame = random_drop_frame
        self.random_rotate_video = random_rotate_video
        self.force_rescale_prob = force_rescale_prob
        print(f"force_rescale_prob: {self.force_rescale_prob}")

        if filter_video_id_file is not None:
            print(f'[DATA]--loading the item_id filter set from {filter_video_id_file}')
            with open(filter_video_id_file, 'rb') as fr:
                self.filter_video_id_set = pickle.load(fr)
        else:
            self.filter_video_id_set = None

        if filter_music_id_file is not None:
            print(f'[DATA]--loading the music filter set from {filter_music_id_file}')
            with open(filter_music_id_file, 'r') as fr:
                self.filter_music_id_set = set()
                for line in fr:
                    self.filter_music_id_set.add(str(line.strip()))
            print(f'[DATA]--loaded the music filter set of size {len(self.filter_music_id_set)}')

        else:
            self.filter_music_id_set = None

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir) 
        self.processor = Siglip2ImageProcessorFast()
        self.add_title_text = add_title_text
        self.embed_lang_caption = embed_lang_caption
        print(f"Add title text: {self.add_title_text}")
        print(f"Embed lang caption: {self.embed_lang_caption}")
        
        # with open('/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/music_clip_ids_1009_enriched_650w_final_dedup.pkl', 'rb') as fr:
        with open('/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/music/music_train_1216_1000w_plus_200w.pkl', 'rb') as fr:
            self.all_music_caption_dict = pickle.load(fr)
            print(f"Totally {len(self.all_music_caption_dict)} video music ids in meta info dict")

        self.allowed_music_selected_from = allowed_music_selected_from
        self.allowed_music_selected_from_set = set()
        self.item_id_2_music_selected_from = {}
        if allowed_music_selected_from != "":
            self.allowed_music_selected_from_set = set(allowed_music_selected_from.split(","))
            print(f"Allowed_music_selected_from_set {self.allowed_music_selected_from_set}")
            music_selected_from_data_path = "/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/data/all_60m_item_ids_with_music_selected_from.jsonl"
            print(f"Loading music_selected_from data from {music_selected_from_data_path}")
            with open(music_selected_from_data_path, "r") as fr:
                for line in fr:
                    data = json.loads(line.strip())
                    if "item_id" not in data or "music_selected_from" not in data:
                        continue
                    self.item_id_2_music_selected_from[data["item_id"]] = data["music_selected_from"]
            print(f"{len(self.item_id_2_music_selected_from)} item_id_2_music_selected_from data loaded!")
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

        music_lang2idx = {'None': 0, 'English': 1, 'Spanish': 2, 'Arabic': 3, 'non_vocal': 4, 'Portuguese': 5, 'Turkish': 6, 'Hindi': 7, 'Indonesian': 8, 'Panjabi': 9, 'Bengali': 10, 'Russian': 11, 'Thai': 12, 'Vietnamese': 13, 'French': 14, 'Tamil': 15, 'Pashto': 16, 'Sindhi': 17, 'Japanese': 18, 'Chinese': 19, 'Urdu': 20, 'Bhojpuri': 21, 'Nepali': 22, 'Korean': 23, 'Italian': 24, 'Kurdish': 25, 'Burmese': 26, 'Afrikaans': 27, 'Polish': 28, 'Malay': 29, 'Persian': 30, 'Tagalog': 31, 'Zulu': 32, 'German': 33, 'Serbian': 34, 'Swahili': 35, 'Romanian': 36, 'Javanese': 37, 'Marathi': 38, 'Albanian': 39, 'Ukrainian': 40, 'Gujarati': 41, 'Greek, Modern (1453-)': 42, 'Telugu': 43, 'Kazakh': 44, 'Wolof': 45, 'Malayalam': 46, 'Azerbaijani': 47, 'Hebrew': 48, 'Sinhalese': 49, 'Minangkabau': 50}
        self.music_lang2idx = defaultdict(lambda: music_lang2idx['None'], music_lang2idx)
        
        print(f"Totally {len(self.country2idx)} country2idx")
        print(f"Totally {len(self.lang2idx)} lang2idx")
        print(f"Totally {len(self.music_lang2idx)} music_lang2idx")

        self.pairwise_score_thresh = pairwise_score_thresh
        self.num_of_pair_per_ins = num_of_pair_per_ins
        print(f"pairwise_score_thresh: {pairwise_score_thresh}")
        print(f"num_of_pair_per_ins: {num_of_pair_per_ins}")


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
            print(f"cur_worker_files, {cur_worker_files}")
            for filepath in cur_worker_files:
                try:
                    with hopen(filepath, 'rb') as reader:
                        for _, line in enumerate(reader):
                            try:
                                data_item = json.loads(line.decode())
                                if self.filter_video_id_set is not None:
                                    video_id = data_item['item_id']
                                    if video_id not in self.filter_video_id_set:
                                        # print(f"video_id {video_id} filter_video_id_set")
                                        continue
                                if self.filter_music_id_set is not None:
                                    music_id = data_item.get('music_id_unpublish') or data_item['music_id']
                                    music_id = str(music_id)
                                    if music_id in self.filter_music_id_set:
                                        # print(f"music_id {music_id} filter_music_id_set")
                                        continue
                            except Exception as e:
                                continue
                            music_id = data_item.get('music_id_unpublish') or data_item['music_id']
                            # 缺少相关的meta信息
                            if music_id not in self.all_music_caption_dict:
                                print(f"music_id {music_id} has no caption")
                                continue
                            
                            music_caption_file = self.all_music_caption_dict[music_id]
                            with open(music_caption_file, 'r') as fr:
                                music_caption_dict = json.load(fr)
                            # music_caption_dict = self.all_music_caption_dict[music_id]
                            data_item.update(music_caption_dict)
                            if self.buffer_size <= 0:
                                yield data_item
                            elif len(self.buffer) < self.buffer_size:
                                self.buffer.append(data_item)
                            else:
                                yield self.enq_deq_buffer(data_item)
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
        
        return pixel_values, pixel_masks, spatial_shapes, frame_mask

    def read_video_frames(self, video):
        if len(video) > self.max_frame_len:
            sample_idx = torch.linspace(0, len(video) - 1, self.max_frame_len).round().long().tolist()
            video = [video[i] for i in sample_idx]

        video = [base64.b64decode(video[i]) for i in range(len(video))]
        video = [Image.open(io.BytesIO(image_str)).convert('RGB') for image_str in video]

        if self.force_rescale_prob and self.force_rescale_prob > 0.0 and random.random() < self.force_rescale_prob:
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

        if self.random_rotate_video:
            angles = [Image.ROTATE_90,Image.ROTATE_180,Image.ROTATE_270]
            if random.random() < 0.3:
                angle = random.choice(angles)
                video = [frame.transpose(angle) for frame in video]

        if self.random_drop_frame:
            if random.random() < 0.5:
                selected_frame_num = random.randint(1, len(video))
                selected_frame_idx = list(range(len(video)))
                selected_frame_idx = sorted(random.sample(selected_frame_idx, selected_frame_num))
                video = [video[idx] for idx in selected_frame_idx]

        video_tensors,frame_masks,spatial_shapes,frame_mask = self.process_image(video)

        assert video_tensors.shape[0] == self.max_frame_len, f"The input video frame number does not equal to {self.max_frame_len}"
        
        return video_tensors, frame_masks, spatial_shapes, frame_mask

    def _load_music_meta(self, music_id):
        """
        辅助函数：根据 music_id 读取 meta 信息文件
        """
        music_id = int(music_id)
        if music_id not in self.all_music_caption_dict:
            return None
        
        music_caption_file = self.all_music_caption_dict[music_id]
        try:
            with open(music_caption_file, 'r') as fr:
                return json.load(fr)
        except Exception as e:
            print(f"Error loading music meta for {music_id}: {e}")
            return None

    def _get_music_features(self, music_item, score, suffix=""):
        """
        核心函数：将音乐的 raw dict 处理为模型需要的 tensor 和 input_ids
        suffix: 特征后缀，'' 为第一路，'_2' 为第二路
        """
        # 1. 基础信息解析
        music_capton = music_item.get('content', '')
        is_pgc_flag = int(music_item.get('is_pgc', 0))
        has_meta_song = music_item.get('meta_song_id', None) is not None
        
        if is_pgc_flag == 1:
            music_dist_code = 2  # PGC
        else:
            if has_meta_song:
                music_dist_code = 1  # PUGC
            else:
                music_dist_code = 0  # UGC
        pgc_map = {0: 'UGC', 1: 'PUGC', 2: 'PGC'}

        # 2. 属性拼接
        def get_joined_str(key):
            val = music_item.get(key, [])
            if isinstance(val, list):
                return ','.join(val)
            return str(val) if val else ''

        music_genre = get_joined_str('genre')
        music_language = get_joined_str('language')
        music_theme = get_joined_str('theme')
        music_mood = get_joined_str('mood')
        music_title = music_item.get('title', '')

        music_attribute = f"language: {music_language}; genre: {music_genre}; theme: {music_theme}; mood: {music_mood}; music source: {pgc_map[music_dist_code]}"
        
        if self.add_title_text and has_meta_song:
            music_attribute = f"language: {music_language}; title: {music_title}; genre: {music_genre}; theme: {music_theme}; mood: {music_mood}; music source: {pgc_map[music_dist_code]}"
            
        if "publish_180d" in music_item:
            music_attribute += f"; music popularity: {music_item['publish_180d']}"

        if self.embed_lang_caption:
            music_capton = f"这是一首{music_language}语言的歌曲。 {music_capton}"

        # 3. Tokenization
        music_attribute_input_ids = self.tokenizer(
            music_attribute, padding='max_length', truncation=True, 
            max_length=self.max_short_len, return_attention_mask=False, return_tensors="pt"
        ).input_ids[0]
        
        music_caption_input_ids = self.tokenizer(
            music_capton, padding='max_length', truncation=True, 
            max_length=self.max_text_len, return_attention_mask=False, return_tensors="pt"
        ).input_ids[0]

        # 4. Vector 转换 (处理 eval 安全性建议在生产环境优化，这里保持原逻辑)
        def parse_vector(key):
            val = music_item.get(key)
            if val is None:
                raise ValueError(f"Missing {key}")
            vec = torch.tensor(eval(val))
            return vec

        try:
            music_content_vector =  torch.tensor(eval(music_item.get('content_vector')))
            music_title_vector = torch.tensor(eval(music_item.get('title_vector')))
            music_cover_vector =  torch.tensor(eval(music_item.get('cover_vector')))
        except Exception as e:
            print(f"Error parsing vector for music_id {music_item.get('music_id', 'unknown')}: {music_item}")
            return None
        # 5. 语言编码
        lang_key = 'None'
        if isinstance(music_item.get('language'), list) and len(music_item['language']) > 0:
            lang_key = music_item['language'][0]
        elif isinstance(music_item.get('language'), str):
            lang_key = music_item['language'].strip()
        music_lang_code = self.music_lang2idx.get(lang_key, 0) # 假设 0 是 default

        # 6. 组装结果
        res = {
            f'music_ids{suffix}': str(music_item.get('music_id', '')),
            f'music_content_ue_vector{suffix}': music_content_vector,
            f'music_title_ue_vector{suffix}': music_title_vector,
            f'music_cover_ue_vector{suffix}': music_cover_vector,
            f'music_attribute_input_ids{suffix}': music_attribute_input_ids,
            f'music_caption_input_ids{suffix}': music_caption_input_ids,
            f'music_attribute{suffix}': music_attribute,
            f'music_capton{suffix}': music_capton,
            f'video_music_match_score{suffix}': score,
            f'music_dist_code{suffix}': music_dist_code,
            f'music_lang_code{suffix}': music_lang_code
        }
        return res

    def __iter__(self):
        for data_item in self.generate():
            try:
                item_id = data_item['item_id']
                # print(f"process item_id: {item_id}")
                if data_item.get('content_vector', None) is None:
                    print(f"item_id {item_id} has no content_vector")
                    continue

                if self.allowed_music_selected_from != "":
                    if int(item_id) not in self.item_id_2_music_selected_from:
                        continue
                    if self.item_id_2_music_selected_from[int(item_id)] not in self.allowed_music_selected_from_set:
                        continue

                # 视频特征提取 (这是共用的)
                video_frames = data_item['video']
                pixel_values, attention_mask, spatial_shapes, frame_mask = self.read_video_frames(video_frames)

                if frame_mask.sum() == 0:
                    print('encounter all padding frames in video %s' % (item_id))
                    continue
                
                # 准备 User 特征
                user_feat = {
                    'user_lang': data_item.get('user_lang'),
                    'user_country': data_item.get('user_country'),
                    'user_lang_code': self.lang2idx.get(data_item.get('user_lang'), 0),
                    'user_country_code': self.country2idx.get(data_item.get('user_country'), 0)
                }

                # 准备视频特征字典
                video_feat = {
                    'item_ids': str(item_id), 
                    'pixel_values': pixel_values, 
                    'pixel_attention_mask': attention_mask, 
                    'spatial_shapes': spatial_shapes,
                    'frame_mask': frame_mask,
                }
                video_feat.update(user_feat)

                # --- 1. 生成原始 Pointwise 样本 ---
                
                # 确定 Label
                if 'llm_score_v5' in data_item:
                    pw_score = int(data_item['llm_score_v5'])
                    if pw_score <= 0 or pw_score >= 6:
                        continue
                    else:
                        # 只有满足采样条件才生成 Pointwise
                        try:
                            music_feat_1 = self._get_music_features(data_item, pw_score, suffix="")
                            final_res = {"is_pairwise": 0}
                            final_res.update(video_feat)
                            final_res.update(music_feat_1)
                            # print("yield sample", final_res.keys())
                            music_feat_2 = self._get_music_features(data_item, pw_score, suffix="_2")
                            final_res.update(music_feat_2)
                            yield final_res
                        except ValueError:
                            print(f"item_id {item_id} has no music_feat_1")
                            continue
                else:
                    continue
                
                # --- 2. 生成 Pairwise 样本 ---
       
                selected_pair = data_item.get('pair', {})
                
                if selected_pair is not {} and "id1" in selected_pair:
                    mid1 = selected_pair['id1']
                    mid2 = selected_pair['id2']
                    score1 = selected_pair['score1']
                    score2 = selected_pair['score2']
                    # 获取 Music 1 数据
                    
                    meta1 = self._load_music_meta(mid1)
                    
                    # 获取 Music 2 数据
                    
                    meta2 = self._load_music_meta(mid2)

                    if meta1 is None or meta2 is None:
                        continue # 如果缺少 meta 文件则跳过

                    # 提取特征
                    feat1, feat2 = None, None
                    try: 
                        feat1 = self._get_music_features(meta1, score1, suffix="")
                        feat2 = self._get_music_features(meta2, score2, suffix="_2")
                    except ValueError as e:
                        print("extract pair feat err")
                        print(e.what())
                        continue


                    # 组装 Pairwise 样本
                    pair_res = {}
                    pair_res.update(video_feat) # 视频特征复用
                    pair_res.update(feat1)
                    pair_res.update(feat2)
                    pair_res['is_pairwise'] = 1
                    assert("music_content_ue_vector" in pair_res)
                    assert("music_content_ue_vector_2" in pair_res)
                    # print("yield pairwise")
                    yield pair_res

            except Exception as e:
                print(f'encounter broken data {data_item.get("item_id", "unknown")}: {e}')
                raise

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


def multi_thread_get_files(root_dir):
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


if __name__ == "__main__":
    import time
   
    tokenizer_dir = '/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base'
    data_path = '/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/music_captions/llm_judge_pretrain_data_pgc_1023_28m.json'
    hdfs_dataset = VideoMusicPairwiseHDFSDataset(data_path, shuffle=True, repeat=True, buffer_size=256, max_frame_len=5, random_drop_frame=True, tokenizer_dir=tokenizer_dir)
    torch.cuda.set_device(0)
    loader = create_mm_embed_dataloader(
        hdfs_dataset,
        batch_size=2,
        num_workers=16,
        cuda_prefetch=False,
    )

    # # text_list = ['hello world, my dear country']
    # text_list = ['The video features a person in a dark room with musical equipment, including a keyboard and a guitar. The individual is wearing a black jacket with white stripes and a cap with text on it. They are engaged in activities such as sitting at a desk, handling papers, and wearing headphones. The room has a modern aesthetic with a whiteboard and various items on the desk. There is no discernible text for OCR.']
    # # text_list = ["""The video showcases a serene winter landscape featuring a snow-covered mountain range in the background. The primary subjects are the natural elements, including the snow-capped mountains, a partially frozen lake, and dense evergreen trees. The attributes of the scene include the white and gray hues of the snow and ice, the rugged texture of the mountain peaks, and the tall, dark green trees. There are no visible human or animal activities, except for a distant train moving along the tracks near the base of the mountains. The actions in the video are minimal, primarily focusing on the stillness of the winter environment and the slow movement of the train. The scenes transition from wide shots of the lake and mountains to closer views of the trees and the lake's edge. There is no text overlay or visible OCR content in thevideo."""]
    # # text_list = [""]
    # text_input_ids, text_segment_ids = hdfs_dataset.process_text(text_list)
    # text_input_masks = np.array(text_input_ids > 0, dtype=np.int32)
    # print(text_input_ids, text_segment_ids, text_input_masks)

    for data in loader:
        # print(data['music_ids'], data['item_ids'])
        # print(data['frames'].shape, data['frames_mask'])
        # print(data['pixel_attention_mask'].shape, data['frame_mask'].shape,  data['frame_mask'], data['spatial_shapes'])
        print(data['video_music_match_score'])
        time.sleep(1)

    # files = multi_thread_get_files('hdfs://harunasg/home/byte_data_tt_m/jinyang.leo/video_pretrain_data_music/llm_judge_pretrain_data_pgc_1023/part_0_27m/data_merge_fea')
    # with open('/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/music_captions/llm_judge_pretrain_data_pgc_1023_28m.json', 'w') as fw:
    #     json.dump(files, fw)
