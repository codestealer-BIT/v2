import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
import datetime
import numpy as np
import time
import torch
import pickle

import sys
sys.path.append('/opt/tiger/MMPretrain')

torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtypes.Float64DType, np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType])

import logging
import json
import math
import random
import transformers
from pathlib import Path
from packaging import version
from copy import deepcopy
import faiss
from tqdm import tqdm

from dataset import TikTokRetrievalDataset

from video_clip import VideoSigLIP2ShortLongBLIP, VideoSigLIP2ShortLong
from config import cfg

from trainer_misc import (
    init_distributed_mode, 
    setup_for_distributed, 
    create_optimizer_for_encoder,
    train_one_epoch_for_video_siglip,
    constant_scheduler,
    cosine_scheduler,
)

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
from torch.utils.data.dataloader import default_collate



from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig, 
    FullStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
    StateDictType,
)

from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy
from torch.utils.data import DataLoader
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.t5.modeling_t5 import T5Block
from transformers import Blip2Model

import accelerate
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import FullyShardedDataParallelPlugin
from accelerate.utils import DistributedDataParallelKwargs
from diffusers.utils import is_wandb_available
from accelerate.logging import get_logger
from IPython import embed

logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser('Build Faiss Database with siglip2', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int, help="The per device batch size")
    

    # Model parameters
    parser.add_argument('--model_name', default='video_siglip2', type=str)
    parser.add_argument('--model_dtype', default='no', type=str, help="The Model Dtype: bf16 or fp16", choices=['no', 'bf16', 'fp16'])

    # The training manner config
    parser.add_argument('--model_path',default="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base", type=str, help='the config path of the siglip2 models to be used')
    parser.add_argument('--pretrained_path',default="/mnt/bn/yexiaoyu-test/checkpoints/debug_pooling_one_layer/checkpoint-30000", type=str, help='the pretrained model of stage one model')
    parser.add_argument('--blip_path',default="", type=str, help='the blip model path')
 
    # Model input config
    parser.add_argument('--max_frames', default=8, type=int, help='number of max video frames')
    parser.add_argument('--interpolate',default=4, type=int, help='number of text interpolation times')
    parser.add_argument('--max_short_len',default=64, type=int, help='number of text tokens')
    parser.add_argument('--hdfs_file_path', default='', type=str, help="The annotation HDFS remote path")

    # Dataset parameters
    parser.add_argument('--max_steps', type=int, default=1e5,
                        help='hack to use tqdm when we do not know the actual dataset size')
    parser.add_argument('--output_dir', type=str, default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--chunk_size', default=1e5, type=int, help='the chunk size of parquet files')
    parser.add_argument('--logging_dir', type=str, default='log', help='path where to tensorboard log')
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    # Distributed Training parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--global_step', default=0, type=int, metavar='N', help='The global optimization step')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training', type=str)

    return parser.parse_args()


def build_model_runner(args):
    model_dtype = args.model_dtype
    model_path = args.model_path
    model_name = args.model_name
    
    assert args.model_name == 'video_siglip2'
    runner = VideoSigLIP2ShortLongBLIP(
        model_path,
        blip_path=args.blip_path,
        gpuwise_nce=False,
        queue_size=0,
        interpolate=args.interpolate,
    )
    
    print("load pretrained stage one model from: {}".format(args.pretrained_path))
    runner.load_state_dict(torch.load(args.pretrained_path + '/pytorch_model.bin'))

    # freeze the model
    for param in runner.parameters():
        param.requires_grad = False

    runner.eval()

    return runner





def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # Initialize the Environment variables throught MPI run
    init_distributed_mode(args, init_pytorch_ddp=False)   # set `init_pytorch_ddp` to False, since the accelerate will do later

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=args.model_dtype,
        log_with=None,
        project_config=accelerator_project_config,
        fsdp_plugin=None,
        kwargs_handlers=[ddp_kwargs],
    )

    if args.report_to == "wandb":
        try:
            import wandb
        except:
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    device = accelerator.device

    # building model
    runner = build_model_runner(args)

    # building dataloader
    global_rank = accelerator.process_index

    test_dataset = TikTokRetrievalDataset(video_root=args.hdfs_file_path, 
                max_text_len=args.max_text_len,
                max_short_len=args.max_short_len,
                max_frame_len=args.max_frames,
                tokenizer_dir=args.model_path,
                fuse_caption=True,
                )

    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=default_collate,
    )  

    accelerator.wait_for_everyone()
    logger.info("Building dataset finished")

    # To block the print on non main process
    setup_for_distributed(accelerator.is_main_process)

    

    # report model details
    n_learnable_parameters = sum(p.numel() for p in runner.parameters() if p.requires_grad)
    n_fix_parameters = sum(p.numel() for p in runner.parameters() if not p.requires_grad)
    logger.info(f'total number of learnable params: {n_learnable_parameters / 1e6} M')
    logger.info(f'total number of fixed params in : {n_fix_parameters / 1e6} M')
    


    # Wrap the model, optmizer, and scheduler with accelerate
    logger.info(f'before accelerator.prepare')

    # Only wrapping the trained dit and huge text encoder
    runner,dataloader = accelerator.prepare(runner,dataloader)

    logger.info(f'after accelerator.prepare')
    logger.info(f'{runner}')


    # Report the inference info
    total_batch_size = args.batch_size * accelerator.num_processes
    logger.info("***** Running inferencing *****")
    logger.info("Batch size = %d" % total_batch_size)


    # Start inferencing!
    start_time = time.time()
    accelerator.wait_for_everyone()

    
    # Store embeddings and item IDs per rank
    local_embeddings = []
    local_item_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            
            pixel_values = batch['pixel_values']
            pixel_attention_mask = batch['pixel_attention_mask']
            spatial_shapes = batch['spatial_shapes']

            title_input_ids = batch['title_input_ids']

            item_ids = [int(s) for s in batch["item_ids"]]
            item_ids = torch.tensor(item_ids, dtype=torch.int64).to(accelerator.device)
            
            batch_size, concat, seq_len = title_input_ids.shape
            #print("title_input_ids.shape: ",title_input_ids.shape)
            segment_ids = torch.zeros_like(title_input_ids)# 0: title
            segment_ids = segment_ids.view(batch_size, concat* seq_len).contiguous()
            segment_ids[:, 64:128] = 1    # 1: sticker
            segment_ids[:, 128:192] = 2   # 2: ocr
            segment_ids[:, 192:256] = 3   # 3: asr

            title_input_ids = title_input_ids.view(batch_size * concat, seq_len).contiguous()

            
            # title
            title_embeds, title_pooled = runner.encode_text(input_ids = title_input_ids)

            title_embeds = title_embeds.view(batch_size, concat, seq_len, -1).contiguous()

            title_embeds = title_embeds.view(batch_size, concat * seq_len, -1).contiguous()
            #video
            video_embeds, video_pooled = runner.encode_video(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes)
            #fusion
            fused_embeds, fused_pooled = runner.fuse_video_title(vision_embed = video_embeds,title_embed = title_embeds,  
                                                                vision_attn_mask=None, title_attn_mask = None, segment_ids = segment_ids)
            

            local_embeddings.append(fused_pooled)
            local_item_ids.append(item_ids)
    # Gather embeddings and IDs from all ranks
    all_embeddings = accelerator.gather(torch.cat(local_embeddings))
    all_item_ids = accelerator.gather(torch.cat(local_item_ids))

    if accelerator.is_main_process:

        all_embeddings = all_embeddings.detach().cpu().numpy().astype("float32")
        all_item_ids = all_item_ids.detach().cpu().numpy().astype("int64")
        all_item_ids, unique_indices = np.unique(all_item_ids, return_index=True)
        all_embeddings = all_embeddings[unique_indices]
        print("item_ids:", all_item_ids[:5])

        faiss.normalize_L2(all_embeddings)
        dim = all_embeddings.shape[1]
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        index.add_with_ids(all_embeddings, all_item_ids)

        # Save index + ID map
        os.makedirs(args.output_dir, exist_ok=True)
        faiss.write_index(index, os.path.join(args.output_dir, "sticker_caption_vectors.index"))


        print(f"✅ Saved FAISS index in '{args.output_dir}'")
        print(f"Total vectors: {index.ntotal}")

        # saving parquet files
        id_count = index.ntotal
        chunk_size = int(args.chunk_size)

        # saving parquet files
        total_chunks = int((id_count + chunk_size - 1) // chunk_size)

        # 分块导出为Parquet文件
        for i in tqdm(range(total_chunks), desc="Saving parquet files"):
            # 计算当前块的索引范围
            start = i * chunk_size
            end = min((i + 1) * chunk_size, id_count)
            print(f"saving {start} to {end}")
            
            # extract
            item_ids_chunk = all_item_ids[start:end]
            embeddings_chunk = all_embeddings[start:end]  # 每个元素是shape=(768,)的数组
            
            # 创建DataFrame（key对应int64，value对应数组）
            df_chunk = pd.DataFrame({
                "item_id": item_ids_chunk,
                "embedding": list(embeddings_chunk)  # 转为列表，使每个元素作为独立数组存入DataFrame
            })
            
            # 导出为Parquet（自动压缩，支持数组类型）
            df_chunk.to_parquet(os.path.join(args.output_dir,f"embeddings/sticker_caption_fusion_embedding_chunk_{i}.parquet"), index=False)
            print(f"Saving {i+1}/{total_chunks}th parquet")
        
    
    accelerator.wait_for_everyone()
    #accelerator.end_training()


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["FSDP_USE_ORIG_PARAMS"] = "true"
    opts = get_args()
    opts.max_text_len = opts.interpolate * 64 - (opts.interpolate -1) * 20
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(opts.output_dir, "embeddings")).mkdir(parents=True, exist_ok=True)
    main(opts)
