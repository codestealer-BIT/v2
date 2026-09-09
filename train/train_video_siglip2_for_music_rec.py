import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
import datetime
import numpy as np
import time
import torch

torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtypes.Float64DType, np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType])

import logging
import json
import math
import random
import transformers
from pathlib import Path
from packaging import version
from copy import deepcopy

from dataset import (
    VideoMusicHDFSDataset,
    create_mm_embed_dataloader,
)

from video_clip import VideoSigLIP2ForMusic, VideoSigLIP2ForMusicOnlyMatching, VideoSigLIP2ForMusicOnlyMatchingContrastive, VideoSigLIP2ForMusicNegativeFiltering, VideoSigLIP2ForMusicOnlyMatchingAddUserInfo, VideoSigLIP2ForMusicNegativeFilteringAddLang, VideoSigLIP2ForMusicNegativeFilteringBMAddLang, VideoSigLIP2ForMusicNegativeFilteringBM, VideoSigLIP2ForMusicOnlyMatchingV2, VideoSigLIP2ForMusicOnlyMatchingV3
from config import cfg

from trainer_misc import (
    init_distributed_mode, 
    setup_for_distributed, 
    create_optimizer_for_encoder,
    train_one_epoch_for_video_siglip_stage_one,
    train_one_epoch_for_video_siglip_for_music_rec,
    constant_scheduler,
    cosine_scheduler,
)

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.t5.modeling_t5 import T5Block
from transformers import Blip2Model

import accelerate
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import ProjectConfiguration, set_seed, InitProcessGroupKwargs
from accelerate import FullyShardedDataParallelPlugin
from accelerate.utils import DistributedDataParallelKwargs
from diffusers.utils import is_wandb_available
from accelerate.logging import get_logger
from IPython import embed

logger = get_logger(__name__)

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
    parser = argparse.ArgumentParser('Pyramid-Flow Multi-process Training script', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int, help="The per device batch size")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--iters_per_epoch', default=2000, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # Model parameters
    parser.add_argument('--ema_update', action='store_true')
    parser.add_argument('--ema_decay', default=0.9999, type=float, metavar='MODEL', help='ema decay rate')
    parser.add_argument('--ema_update_step', default=1, type=int, help="The update frequency of the ema model")
    parser.add_argument('--queue_scale', default=0, type=int, help="The contrastive momentum queue scale")
    parser.add_argument('--buffer_size', default=256,  type=int, help="The buffer size to further shuffle the sample")
    parser.add_argument('--model_name', default='video_siglip2_for_music', type=str)
    parser.add_argument('--model_dtype', default='bf16', type=str, help="The Model Dtype: bf16 or fp16", choices=['bf16', 'fp16'])
    parser.add_argument('--load_model_ema_to_cpu', action='store_true')
    parser.add_argument('--mean_loss', action='store_true')
    parser.add_argument('--use_frame_mask', action='store_true', help="using frame mask in video encoder")

    # FSDP condig
    parser.add_argument('--use_fsdp', action='store_true')
    parser.add_argument('--fsdp_shard_strategy', default='zero2', type=str, choices=['zero2', 'zero3'])
    

    # The training manner config
    parser.add_argument('--model_path',default="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base", type=str, help='the config path of the siglip2 models to be used')
    parser.add_argument('--pretrained_path', default=None, type=str, help='the pretrained model checkpoint path')
    parser.add_argument('--freeze_vision',action='store_true', help='whether to freeze video encoder model weight')
    parser.add_argument('--freeze_text',action='store_true', help='whether to freeze text encoder model weight')
    parser.add_argument('--use_dwa', action='store_true', help='using dynamic weight for training loss')
 
    # Historical model config
    parser.add_argument('--historical_model_path', default=None, type=str, help="The historical model path to be used for negative filtering")
    parser.add_argument('--historical_score_threshold', default=0.0, type=float, help="The historical score threshold to be used for negative filtering")
    parser.add_argument(
        "--false_positive_filter",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        help="对正样本中低于low_quality_score的处理方式, None 不做处理, True 过滤, False 设为负样本（True/False/None，默认None）"
    )
    # Model input config
    parser.add_argument('--max_frames', default=8, type=int, help='number of max video frames')
    parser.add_argument('--anno_file_path', default='', type=str, help="The annotation HDFS remote path")
    parser.add_argument('--interpolate',default=6, type=int, help='number of text interpolation times')
    parser.add_argument('--max_short_len',default=64, type=int, help='number of text tokens')

    parser.add_argument('--random_drop_frame', action='store_true', help="Whether to random drop frame for training")
    parser.add_argument('--random_rotate_video', action='store_true', help="Whether to random roate frame for training")
    parser.add_argument('--force_rescale_prob', default=0.0, type=float, help="The probability to force rescale video frame to long side 256")
    parser.add_argument('--score_3_sample_rate', default=1.0, type=float, help="The probability to sample score 3")
    parser.add_argument('--score_4_sample_rate', default=1.0, type=float, help="The probability to sample score 4")
    parser.add_argument('--filter_video_id_file', default='', type=str, help="The input item id set to filter data")
    parser.add_argument('--low_quality_score', default=4, type=int, help="The low quality boundary of video music matching score")
    parser.add_argument('--min_score', default=None, type=int, help="Skip training samples whose matching score is lower than this value")

    # Training set config
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help="Number of updates steps to accumulate before performing a backward/update pass.")

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_beta1', default=0.9, type=float, metavar='BETA1',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--opt_beta2', default=0.999, type=float, metavar='BETA2',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--text_encoder_lr', type=float, default=1e-5, metavar='LR',
                        help='text encoder learning rate (default: 1e-5)')
    parser.add_argument('--vision_encoder_lr', type=float, default=1e-4, metavar='LR',
                        help='vision encoder learning rate (default: 1e-4)')
    parser.add_argument('--vision_head_lr', type=float, default=1e-4, metavar='LR',
                        help='vision head learning rate (default: 1e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Dataset parameters
    parser.add_argument('--output_dir', type=str, default='',
                        help='path where to save, empty for no saving')
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

    parser.add_argument('--pos_ins_thresh', default=4, type=int)
    parser.add_argument('--neg_ins_thresh', default=2, type=int)

    parser.add_argument('--add_lyrics_ue', default=False, type=str2bool)
    parser.add_argument('--add_title_ue', default=False, type=str2bool)
    parser.add_argument('--add_title_text', default=False, type=str2bool)
    parser.add_argument('--add_user_lang_and_country_code', default=False, type=str2bool)
    parser.add_argument('--user_info_fusion_method', default='gate', type=str)
    parser.add_argument('--embed_lang_caption', default=False, type=str2bool)
    parser.add_argument('--language_add_constant', default=0.0, type=float)
    parser.add_argument('--allowed_music_selected_from', default="", type=str)
    parser.add_argument('--filter_same_meta_prob', default=0.0, type=float)
    parser.add_argument('--enable_llm_v5', default=True, type=str2bool)
    parser.add_argument('--huber_loss_delta', default=1.0, type=float)
    parser.add_argument('--regression_loss_weight', default=0.0, type=float)
    parser.add_argument('--mbias_delta', default=0.01, type=float)
    parser.add_argument('--filter_music_id_file', default=None, type=str)
    return parser.parse_args()


def build_model_runner(args, queue_size):
    model_dtype = args.model_dtype
    model_path = args.model_path
    model_name = args.model_name
    
    if args.model_name == 'video_siglip2_for_music':
        runner = VideoSigLIP2ForMusic(
            model_path,
            gpuwise_nce=True,
            queue_size=queue_size,
            interpolate=args.interpolate,
            mean_loss = args.mean_loss,
            use_frame_mask = args.use_frame_mask,
            low_quality_score= args.low_quality_score,
        )

    elif args.model_name == 'video_siglip2_for_music_only_match':
        runner = VideoSigLIP2ForMusicOnlyMatching(
            model_path,
            gpuwise_nce=True,
            queue_size=queue_size,
            interpolate=args.interpolate,
            mean_loss = args.mean_loss,
            use_frame_mask = args.use_frame_mask,
            low_quality_score= args.low_quality_score,
            pos_ins_thresh= args.pos_ins_thresh,
            neg_ins_thresh= args.neg_ins_thresh,
            add_lyrics_ue= args.add_lyrics_ue,
            add_title_ue= args.add_title_ue,
            add_user_lang_and_country_code= args.add_user_lang_and_country_code,
            huber_loss_delta = args.huber_loss_delta,
            regression_loss_weight = args.regression_loss_weight,
            mbias_delta = args.mbias_delta,
        )
    elif args.model_name == 'video_siglip2_for_music_only_match_contrastive':
        runner = VideoSigLIP2ForMusicOnlyMatchingContrastive(
            model_path,
            gpuwise_nce=True,
            queue_size=queue_size,
            interpolate=args.interpolate,
            mean_loss = args.mean_loss,
            use_frame_mask = args.use_frame_mask,
            low_quality_score= args.low_quality_score,
            pos_ins_thresh= args.pos_ins_thresh,
            neg_ins_thresh= args.neg_ins_thresh,
            add_lyrics_ue= args.add_lyrics_ue,
            add_title_ue= args.add_title_ue,
            add_user_lang_and_country_code= args.add_user_lang_and_country_code,
            huber_loss_delta = args.huber_loss_delta,
            regression_loss_weight = args.regression_loss_weight,
            mbias_delta = args.mbias_delta,
        )
    elif args.model_name == 'video_siglip2_for_music_only_match_v2':
        runner = VideoSigLIP2ForMusicOnlyMatchingV2(
            model_path,
            gpuwise_nce=True,
            queue_size=queue_size,
            interpolate=args.interpolate,
            mean_loss = args.mean_loss,
            use_frame_mask = args.use_frame_mask,
            low_quality_score= args.low_quality_score,
            pos_ins_thresh= args.pos_ins_thresh,
            neg_ins_thresh= args.neg_ins_thresh,
            add_lyrics_ue= args.add_lyrics_ue,
            add_title_ue= args.add_title_ue,
            add_user_lang_and_country_code= args.add_user_lang_and_country_code,
            huber_loss_delta = args.huber_loss_delta,
            mbias_delta = args.mbias_delta,
        )
    elif args.model_name == 'video_siglip2_for_music_only_match_v3':
        runner = VideoSigLIP2ForMusicOnlyMatchingV3(
            model_path,
            gpuwise_nce=True,
            queue_size=queue_size,
            interpolate=args.interpolate,
            mean_loss = args.mean_loss,
            use_frame_mask = args.use_frame_mask,
            low_quality_score= args.low_quality_score,
            pos_ins_thresh= args.pos_ins_thresh,
            neg_ins_thresh= args.neg_ins_thresh,
            add_lyrics_ue= args.add_lyrics_ue,
            add_title_ue= args.add_title_ue,
            add_user_lang_and_country_code= args.add_user_lang_and_country_code,
            huber_loss_delta = args.huber_loss_delta,
            mbias_delta = args.mbias_delta,
        )
    elif args.model_name == 'video_siglip2_for_music_negative_filtering':
        runner = VideoSigLIP2ForMusicNegativeFiltering(
            model_path,
            gpuwise_nce=True,
            queue_size=queue_size,
            interpolate=args.interpolate,
            mean_loss = args.mean_loss,
            use_frame_mask = args.use_frame_mask,
            low_quality_score= args.low_quality_score,
            historical_model_path= args.historical_model_path,
            historical_score_threshold= args.historical_score_threshold,
            false_positive_filter= args.false_positive_filter,
        )
    elif args.model_name == 'video_siglip2_for_music_negative_filtering_add_lang':
        runner = VideoSigLIP2ForMusicNegativeFilteringAddLang(
            model_path,
            gpuwise_nce=True,
            queue_size=queue_size,
            interpolate=args.interpolate,
            mean_loss = args.mean_loss,
            use_frame_mask = args.use_frame_mask,
            low_quality_score= args.low_quality_score,
            historical_model_path= args.historical_model_path,
            historical_score_threshold= args.historical_score_threshold,
            false_positive_filter= args.false_positive_filter,
            language_add_constant= args.language_add_constant,
            filter_same_meta_prob= args.filter_same_meta_prob,
        )
    elif args.model_name == 'video_siglip2_for_music_only_match_add_user_info':
        runner = VideoSigLIP2ForMusicOnlyMatchingAddUserInfo(
            model_path,
            gpuwise_nce=True,
            queue_size=queue_size,
            interpolate=args.interpolate,
            mean_loss = args.mean_loss,
            use_frame_mask = args.use_frame_mask,
            low_quality_score= args.low_quality_score,
            pos_ins_thresh= args.pos_ins_thresh,
            neg_ins_thresh= args.neg_ins_thresh,
            add_lyrics_ue= args.add_lyrics_ue,
            add_title_ue= args.add_title_ue,
            add_user_lang_and_country_code= args.add_user_lang_and_country_code,
            user_info_fusion_method= args.user_info_fusion_method,
        )
    elif args.model_name == 'video_siglip2_for_music_negative_filtering_by_bm_add_user_info':
        runner = VideoSigLIP2ForMusicNegativeFilteringBMAddLang(
            model_path,
            gpuwise_nce=True,
            queue_size=queue_size,
            interpolate=args.interpolate,
            mean_loss = args.mean_loss,
            use_frame_mask = args.use_frame_mask,
            low_quality_score= args.low_quality_score,
            historical_model_path= args.historical_model_path,
            historical_score_threshold= args.historical_score_threshold,
            false_positive_filter= args.false_positive_filter,
            user_info_fusion_method= args.user_info_fusion_method,
        )
    elif args.model_name == 'video_siglip2_for_music_negative_filtering_by_bm':
        runner = VideoSigLIP2ForMusicNegativeFilteringBM(
            model_path,
            gpuwise_nce=True,
            queue_size=queue_size,
            interpolate=args.interpolate,
            mean_loss = args.mean_loss,
            use_frame_mask = args.use_frame_mask,
            low_quality_score= args.low_quality_score,
            historical_model_path= args.historical_model_path,
            historical_score_threshold= args.historical_score_threshold,
            false_positive_filter= args.false_positive_filter,
        )
    else:
        raise ValueError(f"Not implemented for model name {args.model_name}")
    
    assert args.pretrained_path is not None, "For video music finetuning, we must need a pretrained checkpoint"

    if args.pretrained_path:
        pretrained_ckpt_path = os.path.join(args.pretrained_path, 'pytorch_model.bin')
        print(f"Loading pretrained checkpoint from {pretrained_ckpt_path}")
        pretrained_checkpoint = torch.load(pretrained_ckpt_path, map_location='cpu')
        load_res = runner.load_state_dict(pretrained_checkpoint, strict=False)
        print(f"Loading result: {load_res}")

    runner.freeze_modules(freeze_vision=args.freeze_vision, freeze_text=args.freeze_text)

    return runner


def auto_resume(args, accelerator, runner=None, model_ema=None):
    if len(args.resume) > 0:
        path = args.resume
    else:
        # Get the most recent checkpoint
        print(f"args.output_dir: {args.output_dir}")
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        print(f"Latest dirs: {dirs[-5:]}")

    if path is None:
        accelerator.print(
            f"Checkpoint does not exist. Starting a new training run."
        )
        initial_global_step = 0
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        ckpt_dir = os.path.join(args.output_dir, path)
        try:
            accelerator.load_state(ckpt_dir, strict=False)
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
        except ValueError as e:
            # Fallback: optimizer param groups changed (e.g., unfreezing text). Load only model weights.
            accelerator.print(f"Optimizer state incompatible: {e}. Falling back to loading model weights only.")
            model_bin = os.path.join(ckpt_dir, 'pytorch_model.bin')
            try:
                if runner is not None and os.path.exists(model_bin):
                    base_runner = accelerator.unwrap_model(runner)
                    pretrained_checkpoint = torch.load(model_bin, map_location='cpu')
                    load_res = base_runner.load_state_dict(pretrained_checkpoint, strict=False)
                    accelerator.print(f"Loaded model weights from {model_bin}: {load_res}")
                # Try to load EMA weights if available
                if model_ema is not None:
                    ema_bin = os.path.join(ckpt_dir, 'pytorch_model_ema.bin')
                    if os.path.exists(ema_bin):
                        ema_state = torch.load(ema_bin, map_location='cpu')
                        ema_res = model_ema.load_state_dict(ema_state)
                        accelerator.print(f"Loaded EMA weights from {ema_bin}: {ema_res}")
            except Exception as e2:
                accelerator.print(f"Fallback load failed: {e2}. Proceeding without loading weights.")
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
    
    return initial_global_step



def build_fsdp_plugin(args):
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP if args.fsdp_shard_strategy == 'zero2' else ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        auto_wrap_policy=ModuleWrapPolicy([T5Block, CLIPEncoderLayer]),
        cpu_offload=CPUOffload(offload_params=False),
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    )
    return fsdp_plugin


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # Initialize the Environment variables throught MPI run
    init_distributed_mode(args, init_pytorch_ddp=False)   # set `init_pytorch_ddp` to False, since the accelerate will do later

    if args.use_fsdp:
        fsdp_plugin = build_fsdp_plugin(args)
    else:
        fsdp_plugin = None

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=40))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.model_dtype,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[ddp_kwargs, process_group_kwargs],
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
    queue_size = args.batch_size * accelerator.num_processes * args.queue_scale
    print("start build_model_runner")
    runner = build_model_runner(args, queue_size)
    print("end build_model_runner", runner)

    # building dataloader
    global_rank = accelerator.process_index
    anno_file_path = args.anno_file_path

    train_dataset = VideoMusicHDFSDataset(
        anno_file_path,
        rank=global_rank,
        world_size=accelerator.num_processes,
        shuffle=True,
        repeat=True,
        buffer_size=args.buffer_size,
        max_frame_len=args.max_frames,
        max_text_len=args.max_text_len,
        max_short_len = args.max_short_len,
        tokenizer_dir=args.model_path,
        random_drop_frame=args.random_drop_frame,
        random_rotate_video=args.random_rotate_video,
        filter_video_id_file=args.filter_video_id_file if args.filter_video_id_file else None,
        add_title_text=args.add_title_text,
        embed_lang_caption=args.embed_lang_caption,
        add_user_lang_and_country_code=args.add_user_lang_and_country_code,
        allowed_music_selected_from=args.allowed_music_selected_from,
        force_rescale_prob=args.force_rescale_prob,
        score_3_sample_rate=args.score_3_sample_rate,
        score_4_sample_rate=args.score_4_sample_rate,
        min_score=args.min_score,
        enable_llm_v5=args.enable_llm_v5,
        filter_music_id_file=args.filter_music_id_file,
    )
    
    train_dataloader = create_mm_embed_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cuda_prefetch=False,
    )

    accelerator.wait_for_everyone()
    logger.info("Building dataset finished")

    # To block the print on non main process
    setup_for_distributed(accelerator.is_main_process)

    # building ema model
    model_ema = deepcopy(runner) if args.ema_update else None
    if model_ema:
        model_ema.eval()

    # set the ema model not update by gradient
    if model_ema:
        for param in model_ema.parameters():
            param.requires_grad = False

    # report model details
    n_learnable_parameters = sum(p.numel() for p in runner.parameters() if p.requires_grad)
    n_fix_parameters = sum(p.numel() for p in runner.parameters() if not p.requires_grad)
    logger.info(f'total number of learnable params: {n_learnable_parameters / 1e6} M')
    logger.info(f'total number of fixed params in : {n_fix_parameters / 1e6} M')

    # `accelerate` 0.16.0 will have better support for customized saving
    # Register Hook to load and save model_ema
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if model_ema:
                    model_ema_state = model_ema.state_dict()
                    torch.save(model_ema_state, os.path.join(output_dir, 'pytorch_model_ema.bin'))

        def load_model_hook(models, input_dir):
            if model_ema:
                model_ema_path = os.path.join(input_dir, 'pytorch_model_ema.bin')
                if os.path.exists(model_ema_path):
                    model_ema_state = torch.load(model_ema_path, map_location='cpu')
                    load_res = model_ema.load_state_dict(model_ema_state)
                    print(f"Loading ema weights {load_res}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Create the Optimizer
    optimizer = create_optimizer_for_encoder(args, runner)
    logger.info(f"optimizer: {optimizer}")

    # Create the LR scheduler
    num_training_steps_per_epoch = args.iters_per_epoch
    args.max_train_steps = args.epochs * num_training_steps_per_epoch
    warmup_iters = args.warmup_epochs * num_training_steps_per_epoch

    if args.warmup_steps > 0:
        warmup_iters = args.warmup_steps

    logger.info(f"LRScheduler: {args.lr_scheduler}, Warmup steps: {warmup_iters * args.gradient_accumulation_steps}")

    if args.lr_scheduler == 'cosine':
        lr_schedule_values = cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, start_warmup_value=args.warmup_lr, warmup_steps=args.warmup_steps,
        ) 
    elif args.lr_scheduler == 'constant_with_warmup':
        lr_schedule_values = constant_scheduler(
            args.lr, args.epochs, num_training_steps_per_epoch, 
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
    else:
        raise NotImplementedError(f"Not Implemented for scheduler {args.lr_scheduler}")

    # Wrap the model, optmizer, and scheduler with accelerate
    logger.info(f'before accelerator.prepare')

    if fsdp_plugin is not None:
        logger.info(f'show fsdp configs:')
        print('accelerator.state.fsdp_plugin.use_orig_params', accelerator.state.fsdp_plugin.use_orig_params)
        print('accelerator.state.fsdp_plugin.sync_module_states', accelerator.state.fsdp_plugin.sync_module_states)
        print('accelerator.state.fsdp_plugin.forward_prefetch', accelerator.state.fsdp_plugin.forward_prefetch)
        print('accelerator.state.fsdp_plugin.mixed_precision_policy', accelerator.state.fsdp_plugin.mixed_precision_policy)
        print('accelerator.state.fsdp_plugin.backward_prefetch', accelerator.state.fsdp_plugin.backward_prefetch)

    # Only wrapping the trained dit and huge text encoder
    runner, optimizer = accelerator.prepare(runner, optimizer)

    logger.info(f'after accelerator.prepare')
    logger.info(f'{runner}')

    if model_ema and (not args.load_model_ema_to_cpu):
        model_ema.to(device)

    if accelerator.is_main_process:
        accelerator.init_trackers(os.path.basename(args.output_dir), config=vars(args))

    # Report the training info
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    if args.use_dwa:
        logger.info("Use DWA")
        print("Use DWA")
    else:
        logger.info("Not Use DWA")
        print("Not Use DWA")
    logger.info("LR = %.8f" % args.lr)
    logger.info("Min LR = %.8f" % args.min_lr)
    logger.info("Weigth Decay = %.8f" % args.weight_decay)
    logger.info("Batch size = %d" % total_batch_size)
    logger.info("Number of training steps = %d" % (num_training_steps_per_epoch * args.epochs))
    logger.info("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    # Auto resume the checkpoint
    initial_global_step = auto_resume(args, accelerator)
    first_epoch = initial_global_step // num_training_steps_per_epoch

    # Start Train!
    start_time = time.time()
    accelerator.wait_for_everyone()

    for epoch in range(first_epoch, args.epochs):
        train_stats = train_one_epoch_for_video_siglip_for_music_rec(
            runner, 
            model_ema,
            accelerator,
            args.model_dtype,
            train_dataloader,
            optimizer,
            lr_schedule_values,
            epoch, 
            args.clip_grad,
            start_steps=epoch * num_training_steps_per_epoch,
            args=args,
            print_freq=args.print_freq,
            iters_per_epoch=num_training_steps_per_epoch,
            ema_update_step=args.ema_update_step,
            ema_decay=args.ema_decay,
            use_dwa = args.use_dwa,
        )

        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                if accelerator.sync_gradients:
                    global_step = num_training_steps_per_epoch * (epoch + 1)
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path, safe_serialization=False)
                    logger.info(f"Saved state to {save_path}")

            accelerator.wait_for_everyone()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    'epoch': epoch, 'n_parameters': n_learnable_parameters}

        if args.output_dir and accelerator.is_main_process:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["FSDP_USE_ORIG_PARAMS"] = "true"
    opts = get_args()
    opts.max_text_len = opts.interpolate * 64 - (opts.interpolate -1) * 20
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
