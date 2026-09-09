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
from collections import OrderedDict
from einops import rearrange

import io
import base64
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import json
import jsonlines
from utils import is_main_process
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from trainer_misc import init_distributed_mode
from dataset import TTLSHDFSDataset, RandomResizeCrop
from video_clip import VideLocalServiceCLIP
from config import cfg


def get_args():
    parser = argparse.ArgumentParser('Pytorch Multi-process script', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--model_name', default='video_local_service', choices=['video_local_service'], type=str)
    parser.add_argument('--anno_dir', type=str, default='', help="The video annotation file")
    parser.add_argument('--model_dtype', default='bf16', type=str, help="The Model Dtype: bf16 or df16")
    parser.add_argument('--max_frames', default=8, type=int, help='number of max video frames')
    parser.add_argument('--cfg_path', default='', type=str, help='The config file path')
    parser.add_argument('--model_path', default='', type=str, help='The pre-trained weight path')
    parser.add_argument('--output_dir', type=str, default='',
                        help='path where to save, empty for no saving')
    return parser.parse_args()


def build_data_loader(args):
    tokenizer_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/models/byted_nlp_model/m_albertv2_base_v2_t858845_879315'

    dataset = TTLSHDFSDataset(
        args.anno_dir,
        args.rank,
        args.world_size,
        max_frame_len=args.max_frames,
        shuffle=False,
        repeat=False,
        is_training=False,
        tokenizer_dir=tokenizer_dir,
        use_text_feature=True
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
        pin_memory=True, drop_last=False, collate_fn=default_collate,
        prefetch_factor=4,
    )

    return loader


def build_model(args):
    model_dtype = args.model_dtype
    model_path = args.model_path
    cfg_path = args.cfg_path

    if cfg_path:
        cfg.update_cfg(cfg_path)

    model = VideLocalServiceCLIP(
        config=cfg,
        gpuwise_nce=False,
    )

    if model_path:
        print(f"Loading the pre-trained checkpoint from {model_path}")
        pretrained_checkpoint = torch.load(model_path, map_location='cpu')
        load_res = model.load_state_dict(pretrained_checkpoint, strict=False)
        print(f"Loading result: {load_res}")

    return model


def main(args):

    init_distributed_mode(args, block_master=False)

    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda')
    rank = args.rank

    model = build_model(args)
    model.to(device)

    if args.model_dtype == "bf16":
        torch_dtype = torch.bfloat16 
    elif args.model_dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    data_loader = build_data_loader(args)
    torch.distributed.barrier()

    cnt = 0
    all_predict_scores = []
    all_item_ids = []
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    for samples in tqdm(data_loader):
        video_ids = samples['item_ids']
        for key in samples:
            if isinstance(samples[key], torch.Tensor):
                samples[key] = samples[key].to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
            predict_scores = model.predict_label(samples)

        all_item_ids.extend(video_ids)
        all_predict_scores.append(predict_scores)

    all_predict_scores = torch.cat(all_predict_scores, dim=0)
    print(all_predict_scores.shape)
    torch.save({'item_ids': all_item_ids, 'all_predict_scores': all_predict_scores}, os.path.join(args.output_dir, f"rank_{args.rank}_predict.pt"))
    torch.distributed.barrier()


def calculate_score(args):
    all_item_ids = []
    all_predict_scores = []

    if is_main_process():
        for rank in range(8):
            output_path = os.path.join(args.output_dir, f"rank_{rank}_predict.pt")
            data = torch.load(output_path, map_location='cpu')
            item_ids = data['item_ids']
            predict_scores = data['all_predict_scores']
            all_item_ids.extend(item_ids)
            all_predict_scores.append(predict_scores)
        
        all_predict_scores = torch.cat(all_predict_scores, dim=0)
        print("All evaluation data shape: ", all_predict_scores.shape)
        print(f"********** Reporting the TTLS Eval results of {args.model_path}  ***********")

        country_anno_path = '/mnt/bn/jiny-ttls-i18n-fr1q/ttls_content/csv_data/ttls_reg_15_country_video_data_eval_all_added.meta'
        item_id_to_country = {}
        item_id_to_gt = {}
        with jsonlines.open(country_anno_path, 'r') as reader:
            for item in reader:
                item_id_to_country[item['item_id']] = item['country']
                item_id_to_gt[item['item_id']] = item['gne_local_service']

        region_pred_dict = {}
        boundary_score = 0.5
        for index in range(len(all_item_ids)):
            item_id = int(all_item_ids[index])
            pred_score = all_predict_scores[index, 0].item()
            country = item_id_to_country[item_id]

            if item_id_to_gt[item_id] == 'yes':
                local_service = 1
            else:
                local_service = 0

            region_pred_dict.setdefault(country, {
                'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0
            })

            if pred_score > boundary_score:
                pred_label = 1
            else:
                pred_label = 0

            if local_service == 1 and pred_label == 1:
                region_pred_dict[country]['tp'] += 1
            elif local_service == 1 and pred_label == 0:
                region_pred_dict[country]['fn'] += 1
            elif local_service == 0 and pred_label == 1:
                region_pred_dict[country]['fp'] += 1
            elif local_service == 0 and pred_label == 0:
                region_pred_dict[country]['tn'] += 1
        
        total_precision = []
        total_recall = []
        total_f1 = []

        for region in sorted(list(region_pred_dict.keys())):
            tp = region_pred_dict[region]['tp']
            fp = region_pred_dict[region]['fp']
            tn = region_pred_dict[region]['tn']
            fn = region_pred_dict[region]['fn']
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + fp + tn + fn)
            f1 = 2 * precision * recall / (precision + recall)
            print(f"########### Country {region} #############")
            print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
            total_precision.append(precision)
            total_recall.append(recall)
            total_f1.append(f1)

        print(f"########### Average for score boundary {boundary_score} #############")
        print(f"average precision: {sum(total_precision) / len(total_precision)}, average recall: {sum(total_recall) / len(total_recall)}, average f1: {sum(total_f1) / len(total_f1)}")

if __name__ == '__main__':
    args = get_args()
    main(args)
    if int(os.environ["RANK"]) == 0:
        calculate_score(args)