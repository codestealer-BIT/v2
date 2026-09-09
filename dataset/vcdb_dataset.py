import os
import av
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
from transformers import AutoTokenizer,AutoProcessor

def parse_time(timestr: str) -> float:
    """Convert 'HH:MM:SS' to seconds (float)."""
    h, m, s = timestr.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def extract_frames(video_path, start_time, end_time, max_frame_len=8):
    """
    Extract exactly max_frame_len uniformly from [start_time, end_time).
    Decode all frames in the segment and sample evenly.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)

    start_pts = int(start_time * fps)
    end_pts   = int(end_time * fps)

    frames_buffer = []
    container.seek(int(start_time * av.time_base), stream=stream)

    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        t = frame.pts * stream.time_base
        if t < start_time:
            continue
        if t >= end_time:
            break
        frames_buffer.append(frame.to_image())

    container.close()

    if len(frames_buffer) == 0:
        raise RuntimeError(f"No frames extracted from {video_path} between {start_time}-{end_time}s")

    # Uniform sampling (with replacement if fewer frames than requested)
    indices = np.linspace(0, len(frames_buffer) - 1, max_frame_len).astype(int)
    sampled = [frames_buffer[i].convert('RGB') for i in indices]

    return sampled

class VCDBDataset(Dataset):
    def __init__(self, annotations_dir, video_root, max_frame_len=8, tokenizer_dir = None):
        """
        annotations_dir: directory containing VCDB annotation txt files
        video_root: directory containing subfolders with videos
        """
        self.max_frame_len = max_frame_len
        self.processor = AutoProcessor.from_pretrained(tokenizer_dir)
        self.samples = []

        # iterate through annotation files
        annotation_files = glob.glob(os.path.join(annotations_dir, "*.txt"))
        for ann in annotation_files:
            root_name = os.path.splitext(os.path.basename(ann))[0]
            video_dir = os.path.join(video_root, root_name)

            with open(ann, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    v1, v2, s1, e1, s2, e2 = line.split(",")
                    self.samples.append({
                        "video_a": os.path.join(video_dir, v1),
                        "video_b": os.path.join(video_dir, v2),
                        "start_a": parse_time(s1),
                        "end_a": parse_time(e1),
                        "start_b": parse_time(s2),
                        "end_b": parse_time(e2),
                    })
        
        valid_samples = []
        for s in self.samples:
            try:
                _ = extract_frames(s["video_a"], s["start_a"], s["end_a"], 1)  # quick probe
                _ = extract_frames(s["video_b"], s["start_b"], s["end_b"], 1)
                valid_samples.append(s)
            except Exception as e:
                print(f"[Warning] Dropping broken pair {s['video_a']} {s['video_b']} | {e}")
        self.samples = valid_samples
        print("loading {} samples in total".format(len(self.samples)))

    def __len__(self):
        return len(self.samples)
    
    def process_image(self, images):
        
        inputs = self.processor(images=images, return_tensors="pt")

        pixel_values = inputs['pixel_values']
        pixel_masks = inputs['pixel_attention_mask']
        spatial_shapes = inputs['spatial_shapes']

        now_len = pixel_values.shape[0]
        if now_len < self.max_frame_len:
            pixel_values = torch.cat([pixel_values, torch.zeros_like(pixel_values[0].unsqueeze(0)).repeat(self.max_frame_len - now_len, *[1]*(pixel_values.dim()-1))], dim=0)
            pixel_masks = torch.cat([pixel_masks, torch.zeros_like(pixel_masks[0].unsqueeze(0)).repeat(self.max_frame_len - now_len, *[1]*(pixel_masks.dim()-1))], dim=0)
            spatial_shapes = torch.cat([spatial_shapes, torch.ones_like(spatial_shapes[0].unsqueeze(0)).repeat(self.max_frame_len - now_len, *[1]*(pixel_masks.dim()-1)) * 16], dim=0)
        
        
        return pixel_values,pixel_masks,spatial_shapes
    

    def read_video_frames(self, video):
        if len(video) > self.max_frame_len:
            sample_idx = torch.linspace(0, len(video) - 1, self.max_frame_len).round().long().tolist()
            video = [video[i] for i in sample_idx]


        video_tensors,frame_masks,spatial_shapes = self.process_image(video)

        assert video_tensors.shape[0] == self.max_frame_len, f"The input video frame number does not equal to {self.max_frame_len}"
        
        return video_tensors, frame_masks, spatial_shapes

    def __getitem__(self, idx):
        item = self.samples[idx]
        frames_a = extract_frames(item["video_a"], item["start_a"], item["end_a"], self.max_frame_len)
        frames_b = extract_frames(item["video_b"], item["start_b"], item["end_b"], self.max_frame_len)

        frames_a, frame_masks_a, spatial_shapes_a = self.read_video_frames(frames_a)
        frames_b, frame_masks_b, spatial_shapes_b = self.read_video_frames(frames_b)

        res = {
            'frames_a': frames_a,
            'frames_b': frames_b,
            'frame_masks_a': frame_masks_a,
            'frame_masks_b': frame_masks_b,
            'spatial_shapes_a': spatial_shapes_a,
            'spatial_shapes_b': spatial_shapes_b,
        }


        return res

# Example usage:
if __name__ == '__main__':
    dataset = VCDBDataset(annotations_dir="/mnt/bn/yexiaoyu-test/data/VCDB/annotation", video_root="/mnt/bn/yexiaoyu-test/data/VCDB/core_dataset", tokenizer_dir = '/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base')
    res = dataset[0]  # frames_a and frames_b are lists of PIL.Image
    print(res['frames_a'].shape)