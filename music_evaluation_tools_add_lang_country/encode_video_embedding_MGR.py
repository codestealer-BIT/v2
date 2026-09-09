# encode_video_embedding_MGR.py
# Infer content-understanding video embeddings from HDFS JSONL/TXT files containing image_b64.
# image_b64 format: base64(zip(frames)).
# Model: VideoSigLIP2ForMusicOnlyMatching(add_user_lang_and_country_code=False)

import os
import sys
import io
import re
import json
import base64
import zipfile
import random
import argparse
from urllib.parse import urlparse

import torch
import numpy as np
import pyarrow.fs as pafs
from PIL import Image, ImageFile
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader

sys.path.append("/mnt/bn/jiny-ttls-i18n-fr1q/ttls_content")

from trainer_misc import init_distributed_mode
from config import cfg
from video_clip import VideoSigLIP2ForMusicOnlyMatching
from transformers import AutoTokenizer
from dataset.siglip2_preprocessor import Siglip2ImageProcessorFast
from dataset.hdfs_io import hmkdir

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_args():
    parser = argparse.ArgumentParser("Infer video emb from HDFS image_b64 zip frames")

    parser.add_argument(
        "--input_hdfs_path",
        type=str,
        default="hdfs://harunava/home/byte_tiktok_music/proj/content_understanding/music/MGR/ft_data_filtered_capsule",
        help="HDFS input root. Will recursively read part*.txt / part-* files.",
    )
    parser.add_argument(
        "--output_hdfs_path",
        type=str,
        required=True,
        help="HDFS output directory for per-rank JSONL.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Checkpoint directory containing pytorch_model.bin.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base",
        help="SigLIP2 base model path.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=5)
    parser.add_argument("--tokens_per_frame", type=int, default=32)
    parser.add_argument("--max_records_cache", type=int, default=20000)
    parser.add_argument("--create_time", type=str, default="20260518")
    parser.add_argument("--shuffle_files", action="store_true", default=False)
    parser.add_argument("--print_every", type=int, default=5000)

    return parser.parse_args()


def build_model(model_path, pretrained_ckpt_path):
    cfg_path = "/mnt/bn/jiny-ttls-i18n-fr1q/MM-Embedding/experiments/config_finetune_mmembed_v3.yaml"
    if cfg_path:
        cfg.update_cfg(cfg_path)

    model = VideoSigLIP2ForMusicOnlyMatching(
        model_path,
        gpuwise_nce=True,
        interpolate=6,
        use_frame_mask=True,
        add_user_lang_and_country_code=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = Siglip2ImageProcessorFast()

    if pretrained_ckpt_path:
        print(f"Loading checkpoint from {pretrained_ckpt_path}")
        ckpt = torch.load(pretrained_ckpt_path, map_location="cpu")
        load_res = model.load_state_dict(ckpt, strict=True)
        print(f"Loading result: {load_res}")

    return model, processor, tokenizer


def make_hdfs_fs():
    """
    Create a fresh HDFS filesystem handle.

    Important:
    Do not create one fs in Dataset.__init__ and reuse it in DataLoader workers.
    PyArrow HDFS client is not always fork-safe and may cause protobuf errors / segfault.
    """
    fs, _ = pafs.FileSystem.from_uri("hdfs://harunava")
    return fs


def hdfs_path_to_arrow_path(hdfs_path: str) -> str:
    """
    hdfs://harunava/a/b/c -> /a/b/c
    /a/b/c -> /a/b/c
    """
    if hdfs_path.startswith("hdfs://"):
        return urlparse(hdfs_path).path
    return hdfs_path


def list_hdfs_part_files(hdfs_input_path: str, shuffle_files: bool = False):
    """
    Recursively list HDFS part files under input path.
    Do not return _SUCCESS.
    """
    fs = make_hdfs_fs()
    path_without_scheme = hdfs_path_to_arrow_path(hdfs_input_path)

    infos = fs.get_file_info(
        pafs.FileSelector(path_without_scheme, recursive=True)
    )

    files = []
    for info in infos:
        if info.type != pafs.FileType.File:
            continue

        name = os.path.basename(info.path)
        if name == "_SUCCESS":
            continue

        # compatible with:
        # part-00000-xxx-c000.txt
        # part-00000
        # part-r-00000
        if name.startswith("part"):
            files.append(info.path)

    files = sorted(files)
    if shuffle_files:
        random.shuffle(files)

    return files


def decode_zip_frames(image_b64: str, max_frames: int = 5):
    """
    image_b64 is base64(zip(frames)).
    Return List[PIL.Image] sampled uniformly from zip images.
    """
    images = []
    if not image_b64 or max_frames <= 0:
        return images

    try:
        zip_binary = base64.b64decode(image_b64)
        with zipfile.ZipFile(io.BytesIO(zip_binary)) as z:
            image_files = [
                f for f in z.namelist()
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
            ]

            def extract_num(name):
                m = re.search(r"(\d+)", name)
                return int(m.group(1)) if m else 0

            image_files = sorted(image_files, key=extract_num)
            num_frames = len(image_files)
            if num_frames == 0:
                return images

            if num_frames <= max_frames:
                indices = list(range(num_frames))
            else:
                step = num_frames / max_frames
                indices = [int(i * step) for i in range(max_frames)]

            for idx in indices:
                try:
                    with z.open(image_files[idx]) as fh:
                        img = Image.open(io.BytesIO(fh.read())).convert("RGB")
                        images.append(img)
                except Exception:
                    continue

    except Exception:
        return []

    return images


class HdfsImageB64VideoDataset(IterableDataset):
    def __init__(
        self,
        hdfs_input_path,
        rank,
        world_size,
        processor,
        max_frames=5,
        tokens_per_frame=32,
        shuffle_files=False,
    ):
        super().__init__()

        self.input_hdfs_path = hdfs_input_path
        self.rank = rank
        self.world_size = world_size
        self.processor = processor
        self.max_frame_len = max_frames
        self.tokens_per_frame = tokens_per_frame

        # 根治点：
        # 这里只保存 files 列表，不保存 self.fs。
        # DataLoader worker 进程里会在 __iter__ 内重新创建 fs。
        self.files = list_hdfs_part_files(
            hdfs_input_path=hdfs_input_path,
            shuffle_files=shuffle_files,
        )

        print(f"[Rank {rank}] Total input files found: {len(self.files)}")
        if len(self.files) > 0:
            print(f"[Rank {rank}] First file: {self.files[0]}")

    def process_images(self, images):
        """
        PIL images -> SigLIP2 tensors:
        pixel_values, pixel_attention_mask, spatial_shapes, frame_mask
        """
        inputs = self.processor(images=images, return_tensors="pt")

        pixel_values = inputs["pixel_values"]
        pixel_masks = inputs["pixel_attention_mask"]
        spatial_shapes = inputs["spatial_shapes"]

        now_len = pixel_values.shape[0]
        total_tokens = self.max_frame_len * self.tokens_per_frame
        valid_tokens = now_len * self.tokens_per_frame

        frame_mask = torch.zeros((total_tokens), dtype=pixel_masks.dtype)
        frame_mask[:valid_tokens] = 1

        if now_len < self.max_frame_len:
            pad_frames = self.max_frame_len - now_len

            pixel_values = torch.cat(
                [
                    pixel_values,
                    torch.zeros_like(pixel_values[0].unsqueeze(0)).repeat(
                        pad_frames, *[1] * (pixel_values.dim() - 1)
                    ),
                ],
                dim=0,
            )

            pixel_masks = torch.cat(
                [
                    pixel_masks,
                    torch.zeros_like(pixel_masks[0].unsqueeze(0)).repeat(
                        pad_frames, *[1] * (pixel_masks.dim() - 1)
                    ),
                ],
                dim=0,
            )

            spatial_shapes = torch.cat(
                [
                    spatial_shapes,
                    torch.ones_like(spatial_shapes[0].unsqueeze(0)).repeat(
                        pad_frames, *[1] * (spatial_shapes.dim() - 1)
                    ) * 16,
                ],
                dim=0,
            )

        return pixel_values, pixel_masks, spatial_shapes, frame_mask

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        shard_count = self.world_size * num_workers
        shard_id = self.rank * num_workers + worker_id

        # 根治点：
        # 每个 worker 进程内部重新创建 HDFS fs。
        # 不使用 Dataset.__init__ 里创建的 fs，也不使用父进程继承来的 fs。
        fs = make_hdfs_fs()

        local_read_files = 0
        local_yield = 0

        for file_idx, file_path in enumerate(self.files):
            if (file_idx % shard_count) != shard_id:
                continue

            local_read_files += 1

            try:
                with fs.open_input_stream(file_path) as stream:
                    fr = io.TextIOWrapper(io.BufferedReader(stream), encoding="utf-8")

                    for line in fr:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            j = json.loads(line)
                        except Exception:
                            continue

                        image_b64 = j.get("image_b64")
                        if not image_b64:
                            continue

                        images = decode_zip_frames(image_b64, self.max_frame_len)
                        if len(images) == 0:
                            continue

                        try:
                            pixel_values, pixel_masks, spatial_shapes, frame_mask = self.process_images(images)
                        except Exception:
                            continue

                        music_id = j.get("music_id", None)
                        creation_id = j.get("creation_id", None)

                        if creation_id is None:
                            creation_id = j.get("item_id", None)

                        local_yield += 1

                        yield {
                            "pixel_values": pixel_values,
                            "pixel_masks": pixel_masks,
                            "spatial_shapes": spatial_shapes,
                            "frame_mask": frame_mask,
                            "music_id": music_id,
                            "creation_id": creation_id,
                        }

            except Exception as e:
                print(
                    f"[Rank {self.rank} worker {worker_id}] "
                    f"Failed to read file {file_path}: {type(e).__name__}: {e}",
                    flush=True,
                )
                continue

        print(
            f"[Rank {self.rank} worker {worker_id}] "
            f"finished. read_files={local_read_files}, yielded={local_yield}",
            flush=True,
        )


def build_data_loader(args, processor):
    def custom_collate(batch):
        sample_dict = {
            "pixel_values": [],
            "pixel_masks": [],
            "spatial_shapes": [],
            "frame_mask": [],
            "music_ids": [],
            "creation_ids": [],
        }

        for x in batch:
            if x is None:
                continue
            sample_dict["pixel_values"].append(x["pixel_values"])
            sample_dict["pixel_masks"].append(x["pixel_masks"])
            sample_dict["spatial_shapes"].append(x["spatial_shapes"])
            sample_dict["frame_mask"].append(x["frame_mask"])
            sample_dict["music_ids"].append(x["music_id"])
            sample_dict["creation_ids"].append(x["creation_id"])

        if len(sample_dict["pixel_values"]) == 0:
            return None

        sample_dict["pixel_values"] = torch.stack(sample_dict["pixel_values"], dim=0)
        sample_dict["pixel_masks"] = torch.stack(sample_dict["pixel_masks"], dim=0)
        sample_dict["spatial_shapes"] = torch.stack(sample_dict["spatial_shapes"], dim=0)
        sample_dict["frame_mask"] = torch.stack(sample_dict["frame_mask"], dim=0)

        return sample_dict

    dataset = HdfsImageB64VideoDataset(
        hdfs_input_path=args.input_hdfs_path,
        rank=args.rank,
        world_size=args.world_size,
        processor=processor,
        max_frames=args.max_frames,
        tokens_per_frame=args.tokens_per_frame,
        shuffle_files=args.shuffle_files,
    )

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=False,
        collate_fn=custom_collate,
        drop_last=False,
    )

    # num_workers=0 时不能传 prefetch_factor。
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = False

    loader = DataLoader(**loader_kwargs)
    return loader


def write_results_to_hdfs(results, output_dir, batch_id, rank):
    if not results:
        return

    part_name = f"rank_{rank}_batch_{batch_id:05d}.jsonl"
    part_path = os.path.join(output_dir, part_name)

    # 每次写也新建 fs，避免复用有问题的连接。
    fs = make_hdfs_fs()
    path_without_scheme = hdfs_path_to_arrow_path(part_path)

    with fs.open_output_stream(path_without_scheme) as stream:
        stream.write(("\n".join(results) + "\n").encode("utf-8"))


def main():
    args = get_args()
    print(f"args: {args}", flush=True)

    init_distributed_mode(args)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    rank = args.rank
    device = torch.device("cuda")

    ckpt_path = os.path.join(args.data_path, "pytorch_model.bin")
    print(f"[Rank {rank}] ckpt_path = {ckpt_path}", flush=True)

    model, processor, tokenizer = build_model(args.model_path, ckpt_path)
    model = model.eval().to(device)

    torch_dtype = torch.bfloat16

    hmkdir(args.output_hdfs_path)

    data_loader = build_data_loader(args, processor)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    buffer = []
    batch_id = 0
    total_success = 0

    pbar = tqdm(data_loader, disable=(rank != 0))

    for samples in pbar:
        if samples is None:
            continue

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch_dtype):
            # VideoSigLIP2ForMusicOnlyMatching(add_user_lang_and_country_code=False)
            # 不需要传 user_languages / user_countries。
            video_embed = model.extract_video_embeds(
                samples["pixel_values"].to(device),
                samples["pixel_masks"].to(device),
                samples["spatial_shapes"].to(device),
                samples["frame_mask"].to(device),
            )

        video_embed = video_embed.detach().cpu()

        batch_size = len(samples["music_ids"])
        for i in range(batch_size):
            record = {
                "music_id": samples["music_ids"][i],
                "creation_id": samples["creation_ids"][i],
                "video_ue_vector": video_embed[i].tolist(),
                "create_time": args.create_time,
            }
            buffer.append(json.dumps(record, ensure_ascii=False))
            total_success += 1

        if len(buffer) >= args.max_records_cache:
            write_results_to_hdfs(
                results=buffer,
                output_dir=args.output_hdfs_path,
                batch_id=batch_id,
                rank=rank,
            )
            buffer = []
            batch_id += 1

        if args.print_every > 0 and total_success % args.print_every < batch_size:
            print(f"[Rank {rank}] total_success={total_success}", flush=True)

    if buffer:
        write_results_to_hdfs(
            results=buffer,
            output_dir=args.output_hdfs_path,
            batch_id=batch_id,
            rank=rank,
        )

    print(f"[Rank {rank}] total_success = {total_success}", flush=True)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()