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
from collections import OrderedDict, defaultdict
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
from video_clip import VideoMusicCLIP, VideoMusicTextCLIP, VideoSigLIP2ForMusicNegativeFilteringBMAddLang
from config import cfg
from video_clip import VideoSigLIP2ForMusic
from transformers import AutoTokenizer
from dataset.siglip2_preprocessor import Siglip2ImageProcessorFast
from torch.utils.data.distributed import DistributedSampler


data_pipeline = TiktokDataFusedPipeline()


class TikTokVideoDataset(Dataset):

    def __init__(self, anno_path: str, max_frames=5, processor=None, tokens_per_frame=32):
        super().__init__()
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

        self.max_frame_len = max_frames
        self.tokens_per_frame = tokens_per_frame
        self.processor = processor
        self.annotation = annotation
        self.generate_video_language_country_mapping()
    
    def generate_video_language_country_mapping(self):
        lang2idx = {"None":0, "aa": 1, "af": 2, "am": 3, "ar": 4, "az": 5, "bg": 6, "bn": 7, "bo": 8, "br": 9, "ca": 10, "ce": 11, "cs": 12, "cy": 13, "da": 14, "de": 15, "el": 16, "en": 17, "eo": 18, "es": 19, "et": 20, "eu": 21, "fa": 22, "fi": 23, "fr": 24, "fy": 25, "gl": 26, "gu": 27, "he": 28, "hi": 29, "hr": 30, "hu": 31, "id": 32, "is": 33, "it": 34, "ja": 35, "ka": 36, "kk": 37, "km": 38, "kn": 39, "ko": 40, "ku": 41, "ky": 42, "la": 43, "lb": 44, "lt": 45, "lv": 46, "mg": 47, "mk": 48, "ml": 49, "mr": 50, "ms": 51, "my": 52, "\\N": 53, "nb": 54, "ne": 55, "nl": 56, "nn": 57, "no": 58, "not_lang": 59, "ny": 60, "oc": 61, "pa": 62, "pl": 63, "ps": 64, "pt": 65, "ro": 66, "ru": 67, "sa": 68, "sh": 69, "si": 70, "sk": 71, "sl": 72, "sq": 73, "sr": 74, "sv": 75, "sw": 76, "ta": 77, "te": 78, "tg": 79, "th": 80, "tk": 81, "tl": 82, "tr": 83, "tt": 84, "ug": 85, "ur": 86, "uz": 87, "vi": 88, "vo": 89, "zh": 90}
        lang2idx = defaultdict(lambda: lang2idx['None'], lang2idx)
        country2idx = {'None': 0, 'Afghanistan': 1, 'Albania': 2, 'Algeria': 3, 'American Samoa': 4, 'Andorra': 5, 'Angola': 6, 'Anguilla': 7, 'Antarctica': 8, 'Antigua and Barbuda': 9, 'Argentina': 10, 'Armenia': 11, 'Aruba': 12, 'Australia': 13, 'Austria': 14, 'Azerbaijan': 15, 'Bahamas': 16, 'Bahrain': 17, 'Bangladesh': 18, 'Barbados': 19, 'Belarus': 20, 'Belgium': 21, 'Belize': 22, 'Benin': 23, 'Bermuda': 24, 'Bhutan': 25, 'Bolivia': 26, 'Bosnia and Herzegovina': 27, 'Botswana': 28, 'Brazil': 29, 'British Indian Ocean Territory': 30, 'British Virgin Islands': 31, 'Brunei': 32, 'Bulgaria': 33, 'Burkina Faso': 34, 'Burundi': 35, 'Cambodia': 36, 'Cameroon': 37, 'Canada': 38, 'Cape Verde': 39, 'Caribbean Netherlands': 40, 'Cayman Islands': 41, 'Central African Republic': 42, 'Chad': 43, 'Chile': 44, 'China': 45, 'Colombia': 46, 'Comoros': 47, 'Congo': 48, 'Cook Islands': 49, 'Costa Rica': 50, 'Croatia': 51, 'Cuba': 52, 'Curaçao': 53, 'Cyprus': 54, 'Czechia': 55, 'Denmark': 56, 'Djibouti': 57, 'Dominica': 58, 'Dominican Republic': 59, 'East Timor': 60, 'Ecuador': 61, 'Egypt': 62, 'El Salvador': 63, 'Equatorial Guinea': 64, 'Eritrea': 65, 'Estonia': 66, 'Ethiopia': 67, 'Falkland Islands': 68, 'Faroe Islands': 69, 'Federated States of Micronesia': 70, 'Fiji': 71, 'Finland': 72, 'France': 73, 'French Guiana': 74, 'French Polynesia': 75, 'French Southern Territories': 76, 'Gabon': 77, 'Gambia': 78, 'Georgia': 79, 'Germany': 80, 'Ghana': 81, 'Gibraltar': 82, 'Greece': 83, 'Greenland': 84, 'Grenada': 85, 'Guadeloupe': 86, 'Guam': 87, 'Guatemala': 88, 'Guernsey': 89, 'Guinea': 90, 'Guinea-Bissau': 91, 'Guyana': 92, 'Haiti': 93, 'Hashemite Kingdom of Jordan': 94, 'Honduras': 95, 'Hong Kong': 96, 'Hungary': 97, 'Iceland': 98, 'India': 99, 'Indonesia': 100, 'Iran': 101, 'Iraq': 102, 'Ireland': 103, 'Isle of Man': 104, 'Israel': 105, 'Italy': 106, 'Ivory Coast': 107, 'Jamaica': 108, 'Japan': 109, 'Jersey': 110, 'Kazakhstan': 111, 'Kenya': 112, 'Kiribati': 113, 'Kosovo': 114, 'Kuwait': 115, 'Kyrgyzstan': 116, 'Laos': 117, 'Latvia': 118, 'Lebanon': 119, 'Lesotho': 120, 'Liberia': 121, 'Libya': 122, 'Liechtenstein': 123, 'Luxembourg': 124, 'Macao': 125, 'Macedonia': 126, 'Madagascar': 127, 'Malawi': 128, 'Malaysia': 129, 'Maldives': 130, 'Mali': 131, 'Malta': 132, 'Marshall Islands': 133, 'Martinique': 134, 'Mauritania': 135, 'Mauritius': 136, 'Mayotte': 137, 'Mexico': 138, 'Monaco': 139, 'Mongolia': 140, 'Montenegro': 141, 'Montserrat': 142, 'Morocco': 143, 'Mozambique': 144, 'Myanmar [Burma]': 145, 'Namibia': 146, 'Nauru': 147, 'Nepal': 148, 'Netherlands': 149, 'New Caledonia': 150, 'New Zealand': 151, 'Nicaragua': 152, 'Niger': 153, 'Nigeria': 154, 'Niue': 155, 'Northern Mariana Islands': 156, 'Norway': 157, 'Oman': 158, 'Pakistan': 159, 'Palau': 160, 'Palestine': 161, 'Panama': 162, 'Papua New Guinea': 163, 'Paraguay': 164, 'Peru': 165, 'Philippines': 166, 'Pitcairn Islands': 167, 'Poland': 168, 'Portugal': 169, 'Puerto Rico': 170, 'Qatar': 171, 'Republic of Korea': 172, 'Republic of Lithuania': 173, 'Republic of Moldova': 174, 'Republic of the Congo': 175, 'Romania': 176, 'Russia': 177, 'Rwanda': 178, 'Réunion': 179, 'Saint Helena': 180, 'Saint Kitts and Nevis': 181, 'Saint Lucia': 182, 'Saint Martin': 183, 'Saint Pierre and Miquelon': 184, 'Saint Vincent and the Grenadines': 185, 'Saint-Barthélemy': 186, 'Samoa': 187, 'San Marino': 188, 'Saudi': 189, 'Saudi Arabia': 190, 'Senegal': 191, 'Serbia': 192, 'Seychelles': 193, 'Sierra Leone': 194, 'Singapore': 195, 'Sint Maarten': 196, 'Slovak Republic': 197, 'Slovenia': 198, 'Solomon Islands': 199, 'Somalia': 200, 'South Africa': 201, 'South Georgia and the South Sandwich Islands': 202, 'South Sudan': 203, 'Spain': 204, 'Sri Lanka': 205, 'Sudan': 206, 'Suriname': 207, 'Svalbard and Jan Mayen': 208, 'Swaziland': 209, 'Sweden': 210, 'Switzerland': 211, 'Syria': 212, 'São Tomé and Príncipe': 213, 'Taiwan': 214, 'Tajikistan': 215, 'Tanzania': 216, 'Thailand': 217, 'Togo': 218, 'Tokelau': 219, 'Tonga': 220, 'Trinidad and Tobago': 221, 'Tunisia': 222, 'Turkey': 223, 'Turkmenistan': 224, 'Turks and Caicos Islands': 225, 'Tuvalu': 226, 'U.S. Minor Outlying Islands': 227, 'U.S. Virgin Islands': 228, 'Uganda': 229, 'Ukraine': 230, 'United Arab Emirates': 231, 'United Kingdom': 232, 'United States': 233, 'Uruguay': 234, 'Uzbekistan': 235, 'Vanuatu': 236, 'Vatican City': 237, 'Venezuela': 238, 'Vietnam': 239, 'Wallis and Futuna': 240, 'Western Sahara': 241, 'Yemen': 242, 'Zambia': 243, 'Zimbabwe': 244, '\\N': 245, 'Åland': 246}
        country2idx = defaultdict(lambda: country2idx['None'], country2idx)
        
        evalset_country_and_lang_path = "/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/evalset_all_item_and_country_language.tsv"
        item_id2lang_code = {}
        item_id2country_code = {}
        with open(evalset_country_and_lang_path, 'r') as f:
            for line in f:
                lines = line.strip().split("\t")
                item_id = int(lines[0])
                country_name = lines[2]
                lang_code = lines[3]
                item_id2lang_code[item_id] = lang2idx[lang_code]
                item_id2country_code[item_id] = country2idx[country_name]
        self.item_id2lang_code = item_id2lang_code
        self.item_id2country_code = item_id2country_code

    def read_video_frames(self, video):
        video = [base64.b64decode(video[i]) for i in range(len(video))]
        video = [Image.open(io.BytesIO(image_str)).convert('RGB') for image_str in video]
        # Removed legacy photo check and manual resizing; the SigLIP2 processor handles resizing internally

        inputs = self.processor(images=video, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        pixel_masks = inputs['pixel_attention_mask']
        spatial_shapes = inputs['spatial_shapes']

        now_len = pixel_values.shape[0]
        total_tokens = self.max_frame_len * self.tokens_per_frame
        valid_tokens = now_len * self.tokens_per_frame

        # Create frame-level mask over tokens
        frame_mask = torch.zeros((total_tokens), dtype=pixel_masks.dtype)
        frame_mask[:valid_tokens] = 1

        # Pad to max_frame_len (5) if necessary
        if now_len < self.max_frame_len:
            pad_frames = self.max_frame_len - now_len
            pixel_values = torch.cat([
                pixel_values,
                torch.zeros_like(pixel_values[0].unsqueeze(0)).repeat(pad_frames, *[1]*(pixel_values.dim()-1))], dim=0)
            pixel_masks = torch.cat([
                pixel_masks,
                torch.zeros_like(pixel_masks[0].unsqueeze(0)).repeat(pad_frames, *[1]*(pixel_masks.dim()-1))], dim=0)
            spatial_shapes = torch.cat([
                spatial_shapes,
                torch.ones_like(spatial_shapes[0].unsqueeze(0)).repeat(pad_frames, *[1]*(spatial_shapes.dim()-1)) * 16], dim=0)
        
        return pixel_values, pixel_masks, spatial_shapes, frame_mask

    def __getitem__(self, index):
        try:
            anno = self.annotation[index]
            item_id = anno['item_id']
            video_meta = data_pipeline.request_all_info(item_id, frame_number=self.max_frame_len)
            pixel_values, pixel_masks, spatial_shapes, frame_mask = self.read_video_frames(video_meta['video'])
            # print("video_meta", video_meta)
            user_languages = self.item_id2lang_code.get(item_id, 0)
            user_countries = self.item_id2country_code.get(item_id, 0)
            res = {
                'pixel_values': pixel_values,
                'pixel_masks': pixel_masks,
                'spatial_shapes': spatial_shapes,
                'frame_mask': frame_mask,
                'item_ids': item_id,
                'user_languages': user_languages,
                'user_countries': user_countries,
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
    parser.add_argument('--dataset_type', default='batch_eval', choices=['batch_eval', 'custom_dataset'], type=str, help="The dataset type")
    parser.add_argument('--anno_dir', type=str, default='', help="The video annotation file")
    parser.add_argument('--model_dtype', default='bf16', type=str, help="The Model Dtype: bf16 or df16")
    parser.add_argument('--max_frames', default=8, type=int, help='number of max video frames')
    parser.add_argument('--cfg_path', default='', type=str, help='The config file path')
    parser.add_argument('--model_path', default='', type=str, help='The pre-trained weight path')
    parser.add_argument('--ckpt_path', default='', type=str, help='Fine-tuned checkpoint path')
    parser.add_argument('--output_dir', type=str, default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--add_user_lang_and_country_method', default="residual", type=str)
    return parser.parse_args()


def build_data_loader(args, processor):
    def custom_collate(batch):
        sample_dict = {
            'pixel_values': [],
            'pixel_masks': [],
            'spatial_shapes': [],
            'frame_mask': [],
            'item_ids': [],
            'user_languages': [],
            'user_countries': [],
        }
        for data_dict in batch:
            if data_dict is None:
                continue
            sample_dict['pixel_values'].append(data_dict['pixel_values'])
            sample_dict['pixel_masks'].append(data_dict['pixel_masks'])
            sample_dict['spatial_shapes'].append(data_dict['spatial_shapes'])
            sample_dict['frame_mask'].append(data_dict['frame_mask'])
            sample_dict['item_ids'].append(data_dict['item_ids'])
            sample_dict['user_languages'].append(data_dict['user_languages'])
            sample_dict['user_countries'].append(data_dict['user_countries'])

        sample_dict['pixel_values'] = torch.stack(sample_dict['pixel_values'], dim=0)
        sample_dict['pixel_masks'] = torch.stack(sample_dict['pixel_masks'], dim=0)
        sample_dict['spatial_shapes'] = torch.stack(sample_dict['spatial_shapes'], dim=0)
        sample_dict['frame_mask'] = torch.stack(sample_dict['frame_mask'], dim=0)
        return sample_dict

    assert args.dataset_type == 'custom_dataset'
    dataset = TikTokVideoDataset(
        anno_path=args.anno_dir,
        max_frames=args.max_frames,
        processor=processor,
        tokens_per_frame=32,
    )

    # Use DistributedSampler for multi-GPU runs to split work across ranks
    # sampler = SequentialSampler(dataset)
    sampler = DistributedSampler(dataset, shuffle=False)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        sampler=sampler,
        shuffle=False,
        collate_fn=custom_collate,
        drop_last=False,
        prefetch_factor=2,
    )

    return loader


def build_model(model_path, pretrained_ckpt_path, args):
    cfg_path = "/mnt/bn/jiny-ttls-i18n-fr1q/MM-Embedding/experiments/config_finetune_mmembed_v3.yaml"

    if cfg_path:
        cfg.update_cfg(cfg_path)

    model = VideoSigLIP2ForMusicNegativeFilteringBMAddLang(
        model_path,
        gpuwise_nce=True,
        interpolate=6,
        use_frame_mask=True,
        # add_lyrics_ue=args.add_lyrics_ue,
        # add_title_ue=args.add_title_ue,
        user_info_fusion_method=args.add_user_lang_and_country_method
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path) 
    processor = Siglip2ImageProcessorFast()

    if pretrained_ckpt_path:
        print(f"Loading the pre-trained checkpoint from {pretrained_ckpt_path}")
        pretrained_checkpoint = torch.load(pretrained_ckpt_path, map_location='cpu')
        load_res = model.load_state_dict(pretrained_checkpoint, strict=False)
        print(f"Loading result: {load_res}")

    return model, processor, tokenizer


def main():
    args = get_args()
    print(args)

    init_distributed_mode(args)

    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda')
    rank = args.rank

    model, processor, tokenizer = build_model(args.model_path, args.ckpt_path, args)
    model = model.eval()
    device = torch.device('cuda')
    model = model.to(device)
    torch_dtype = torch.bfloat16 

    data_loader = build_data_loader(args, processor)
    torch.distributed.barrier()
    
    args.output_dir = os.path.join(args.output_dir, f"rank_{args.rank}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    cnt = 0
    for samples in tqdm(data_loader):
        video_ids = samples['item_ids']

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
            video_embed = model.extract_video_embeds(
                samples['pixel_values'].to(device),
                samples['pixel_masks'].to(device),
                samples['spatial_shapes'].to(device),
                samples['frame_mask'].to(device),
                samples['user_languages'],
                samples['user_countries'],
            )
        
        video_embed = video_embed.cpu().clone()
        output_path = os.path.join(args.output_dir, f"video_embed_{cnt}.pt")
        
        return_item = {
                'video_ids': video_ids,
                'video_embed': video_embed,
            }

        torch.save(return_item, output_path)
        cnt += 1

    torch.distributed.barrier()


if __name__ == '__main__':
    main()