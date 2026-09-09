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
import whatimage
import pyheif

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import subprocess
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, IterableDataset, DataLoader, RandomSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as F

try:
    from .hdfs_io import hlist_files, hisdir, hopen
except:
    from hdfs_io import hlist_files, hisdir, hopen

try:
    try:
        from .byte_tokenizer import Tokenizer_Pipeline
    except:
        from byte_tokenizer import Tokenizer_Pipeline
except:
    print("Tokenizer_Pipeline is not installed, please install the matx first")

from IPython import embed


class RandomResizeCrop(object):

    def __init__(self, size, crop_w, crop_h, resize_long_edge=False, is_training=True):
        self.size = size
        self.resize_long_edge = resize_long_edge
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.is_training = is_training

    def __call__(self, image_tensor):
        # resize
        h, w = image_tensor.shape[-2:]
        if self.resize_long_edge:
            ratio = float(self.size / float(max(h, w)))
        else:
            ratio = float(self.size / float(min(h, w)))

        new_w, new_h = round(w* ratio), round(h * ratio)
        new_w = max(self.crop_w, new_w)
        new_h = max(self.crop_h, new_h)

        image_tensor = F.resize(image_tensor, (new_h, new_w), interpolation=InterpolationMode.BICUBIC, antialias=True)

        if self.is_training:
            image_tensor = transforms.RandomCrop((self.crop_h, self.crop_w))(image_tensor)
        else:
            image_tensor = transforms.CenterCrop((self.crop_h, self.crop_w))(image_tensor)
        
        return image_tensor


class TTLSHDFSDataset(IterableDataset):
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
            image_w: int = 192,
            image_h: int = 384,
            image_mean: list = [0.485, 0.456, 0.406],
            image_std: list = [0.229, 0.224, 0.225],
            is_training: bool = True,
            tokenizer_dir=None,
            use_text_feature: bool = False,   # whether to tokenize texts
        ):
        super(TTLSHDFSDataset).__init__()
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size

        self.files = self.multi_thread_get_files(data_path)
        self.files = [f for f in self.files if f.find('_SUCCESS') < 0 and f.find("_temporary") < 0 and f.find(".caption") < 0]
        self.files.sort()
        print(len(self.files))
        if len(self.files) % self.world_size != 0:
            print('[DATA]--Whole dataset file num %s cannot split to worldsize %s ' %
                     (len(self.files), self.world_size))
        self.verbose = verbose
        self.repeat = repeat
        self.buffer = []
        self.buffer_size = buffer_size

        self.max_frame_len = max_frame_len
        self.max_text_len = 128
        
        self.segment_map = {
            'caption': 0,
            'query': 0,
            'title': 0,
            'tier3_label': 0, # new
            'sticker': 2,
            'sticker_texts': 2,#new
            'challenge': 3,
            "challenge_names": 3,#new
            "hash_tags": 3,#new
            'music': 4,
            "music_title": 4,#new
            'user_nickname': 5,
            "nickname": 5,#new
            'ocr': 6,
            'asr': 7
        }

        self.use_text_feature = use_text_feature

        if tokenizer_dir is not None and use_text_feature:
            self.tokenizer = Tokenizer_Pipeline(tokenizer_dir, self.max_text_len)
        else:
            self.tokenizer = None

        self.transform = transforms.Compose([
            RandomResizeCrop(size=image_h, crop_w=image_w, crop_h=image_h, resize_long_edge=True, is_training=is_training),
            transforms.Normalize(mean=image_mean, std=image_std),
        ])
        
    def multi_thread_get_files(self, root_dir):
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

            for filepath in cur_worker_files:
                try:
                    with hopen(filepath, 'rb') as reader:
                        for _, line in enumerate(reader):
                            if self.buffer_size <= 0:
                                yield line.decode()
                            elif len(self.buffer) < self.buffer_size:
                                self.buffer.append(line.decode())
                            else:
                                yield self.enq_deq_buffer(line.decode())
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

    def check_is_photo(self, video):
        frame_resolutions = [(frame.width, frame.height) for frame in video]
        if len(set(frame_resolutions)) > 1:
            return True
        else:
            return False

    def read_video_frames(self, video, item_id):
        video = [base64.b64decode(video[i].encode('utf-8')) for i in range(len(video))]

        video_frames = []
        has_none_image = False
        for image_str in video:
            image_format = whatimage.identify_image(image_str)
            # if image_format is None:
            #     has_none_image = True
            if image_format in ['heic']:
                pyheif_img = pyheif.read(image_str)
                image = Image.frombytes(mode=pyheif_img.mode, size=pyheif_img.size, data=pyheif_img.data)
            elif image_format is not None:
                image = Image.open(io.BytesIO(image_str)).convert('RGB')
            else:
                continue
            video_frames.append(image)

        # if has_none_image:
        #     print("Have none image: ", len(video_frames), item_id)

        video = video_frames

        if len(video) > self.max_frame_len:
            sample_idx = torch.linspace(0, len(video) - 1, self.max_frame_len).round().long().tolist()
            video = [video[i] for i in sample_idx]

        is_photo = self.check_is_photo(video)

        if is_photo:
            resize_height = 512
            resize_width = video[0].width * resize_height // video[0].height
            video = [frame.resize((resize_width, resize_height)) for frame in video]

        video_tensors = [torch.from_numpy(np.array(frame)).permute(2, 0, 1) for frame in video]
        frames_mask = [1 for _ in range(len(video_tensors))]
        video_tensors = torch.stack(video_tensors).float() / 255

        if len(video_tensors) < self.max_frame_len:
            frames_mask.extend([0 for _ in range(self.max_frame_len - len(video_tensors))])
            video_tensors = torch.cat([video_tensors, torch.zeros(self.max_frame_len - len(video_tensors), *video_tensors.shape[1:])])
            
        assert video_tensors.shape[0] == self.max_frame_len, f"The input video frame number does not equal to {self.max_frame_len}"
        
        video_tensors = self.transform(video_tensors)
        frames_mask = torch.LongTensor(frames_mask)

        return video_tensors, frames_mask

    def process_text(self, text_list, ret_is_empty=False):
        text_tokens = []
        segment_ids = [0]  # 加入起始符seg_id
        
        is_empty = True
        for text in text_list:
            cur_tokens = self.tokenizer.text2tokens([text])[0]
            if len(cur_tokens) > 0:
                is_empty = False

            cur_seg_ids = [int(self.segment_map['caption']) for _ in range(len(cur_tokens) + 1)]
            segment_ids.extend(cur_seg_ids)
            text_tokens.extend(cur_tokens)
            text_tokens.append(self.tokenizer.special_tokens['sep'])
        
        text_tokens = text_tokens[:-1]  # 去掉结尾符<sep>

        if len(text_tokens) > self.max_text_len - 2:  # 刨去起始符和结尾符(InputBuilder会自动补充这俩，所以需要提前空出来)
            text_tokens = text_tokens[:self.max_text_len - 2]
            segment_ids = segment_ids[:self.max_text_len - 1]
    
            if text_tokens[-1] != self.tokenizer.special_tokens['sep']:
                segment_ids += [segment_ids[-1]]
            else:
                text_tokens = text_tokens[:-1]  # 去掉结尾符<sep>
                segment_ids += [0]
        else:
            segment_ids += [0 for _ in range(self.max_text_len - len(segment_ids))]

        text_ids = self.tokenizer.InputBuilder([text_tokens]).asnumpy()[0]
        segment_ids = np.int32(segment_ids)
        
        ret = (text_ids, segment_ids)

        if ret_is_empty:
            return ret + (is_empty,)

        return ret

    def __iter__(self):
        for item_str in self.generate():
            try:
                data_item = json.loads(item_str)
                item_id = data_item['item_id']
                title = data_item['title'] + ' ' + data_item['challenge']
                poi_name = data_item['poi_name']
                poi_info = data_item['poi_info']
                ocr_text = data_item['ocr']
                asr_text = data_item['asr']
                user_name = data_item['nickname']
                user_bio = data_item['author_description']
                label = torch.LongTensor([data_item['local_service']])

                # process video data
                video_frames = data_item['frames_uniform']
                video_tensors, frames_mask = self.read_video_frames(video_frames, item_id)
        
                # title information
                title_input_ids, title_segment_ids = self.process_text([title])
                title_input_masks = np.array(title_input_ids > 0, dtype=np.int32)
                title_input_ids = torch.LongTensor(title_input_ids)
                title_segment_ids = torch.LongTensor(title_segment_ids)
                title_input_masks = torch.LongTensor(title_input_masks)
                
                # ocr information
                ocr_input_ids, ocr_segment_ids = self.process_text([ocr_text])
                ocr_input_masks = np.array(ocr_input_ids > 0, dtype=np.int32)
                ocr_input_ids = torch.LongTensor(ocr_input_ids)
                ocr_segment_ids = torch.LongTensor(ocr_segment_ids)
                ocr_input_masks = torch.LongTensor(ocr_input_masks)

                # asr information
                asr_input_ids, asr_segment_ids = self.process_text([asr_text])
                asr_input_masks = np.array(asr_input_ids > 0, dtype=np.int32)
                asr_input_ids = torch.LongTensor(asr_input_ids)
                asr_segment_ids = torch.LongTensor(asr_segment_ids)
                asr_input_masks = torch.LongTensor(asr_input_masks)

                # user information
                user_input_ids, user_segment_ids = self.process_text([user_name + '; ' + user_bio])
                user_input_masks = np.array(user_input_ids > 0, dtype=np.int32)
                user_input_ids = torch.LongTensor(user_input_ids)
                user_segment_ids = torch.LongTensor(user_segment_ids)
                user_input_masks = torch.LongTensor(user_input_masks)

                # poi information
                poi_input_ids, poi_segment_ids = self.process_text([poi_name + '; '+ poi_info])
                poi_input_masks = np.array(poi_input_ids > 0, dtype=np.int32)
                poi_input_ids = torch.LongTensor(poi_input_ids)
                poi_segment_ids = torch.LongTensor(poi_segment_ids)
                poi_input_masks = torch.LongTensor(poi_input_masks)

                res = {
                    'item_ids': str(item_id), 'frames': video_tensors, 
                    'frames_mask': frames_mask, 'label': label,
                    'title_input_ids': title_input_ids, 'title_segment_ids': title_segment_ids, 'title_input_masks': title_input_masks,
                    'ocr_input_ids': ocr_input_ids, 'ocr_segment_ids': ocr_segment_ids, 'ocr_input_masks': ocr_input_masks,
                    'asr_input_ids': asr_input_ids, 'asr_segment_ids': asr_segment_ids, 'asr_input_masks': asr_input_masks,
                    'user_input_ids': user_input_ids, 'user_segment_ids': user_segment_ids, 'user_input_masks': user_input_masks,
                    'poi_input_ids': poi_input_ids, 'poi_segment_ids': poi_segment_ids, 'poi_input_masks': poi_input_masks,
                }

                yield res

            except Exception as e:
                print('encounter broken data %s: %s' % (item_id, e))

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


def create_ttls_dataloader(
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


if __name__ == "__main__":
    import time
    # data_path = 'hdfs://harunasg/home/byte_data_tt_m/jinyang.leo/music_rec_data/data'
    # data_path = 'hdfs://harunasg/home/byte_data_tt_m/jinyang.leo/music_rec_data/finetune_v1_data'
    # data_path = '/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/finetune_v1_data_plus'
    data_path = 'hdfs://harunava/user/huangchenzhe.7/LocalService/ls_model/datasets/train_cmb-LS-train'
    tokenizer_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/models/byted_nlp_model/m_albertv2_base_v2_t858845_879315'
    hdfs_dataset = TTLSHDFSDataset(data_path, shuffle=True, repeat=True, buffer_size=256, max_frame_len=8, tokenizer_dir=tokenizer_dir, use_text_feature=True)
    # torch.cuda.set_device(0)
    loader = create_ttls_dataloader(
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
        print(data)
        time.sleep(1)

    

