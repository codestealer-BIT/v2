from typing import Tuple, List
import logging
import json
import torch
import base64
import torchvision
import whatimage
import io
import pyheif
from PIL import Image
import fermion_ops
from video_clip import VideoMusicEncoderForDeploy
from config import cfg
from fermion_core_public.template import utils  # why can't I find this? do: export PYTHONPATH=$PYTHONPATH:<your_repo_root>/template
from IPython import embed
from dataset.hdfs_io import hlist_files, hopen


# OUTPUT_DIR = "output/video_music_clip_no_text"
OUTPUT_DIR = "output/video_music_clip_with_text"


def prepare_frames(video, max_frames=5):
    video = [base64.b64decode(video[i].encode('utf-8')) for i in range(len(video))]
    video_frames = []
    for image_str in video:
        image_format = whatimage.identify_image(image_str)
        if image_format in ['heic']:
            pyheif_img = pyheif.read(image_str)
            image = Image.frombytes(mode=pyheif_img.mode, size=pyheif_img.size, data=pyheif_img.data)
        elif image_format is not None:
            image = Image.open(io.BytesIO(image_str)).convert('RGB')
        else:
            continue
        video_frames.append(image)

    video = video_frames
    if len(video) > max_frames:
        sample_idx = torch.linspace(0, len(video) - 1, max_frames).round().long().tolist()
        video = [video[i] for i in sample_idx]

    output_frames = []
    for image in video:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        output_frames.append(img_byte_arr.getvalue())

    return output_frames


def load_sample_service_input() -> List[Tuple]:
    eval_data_hdfs_root = 'hdfs://harunava/user/huangchenzhe.7/LocalService/ls_model/datasets/val_cmb-LS-train'
    eval_data_files = hlist_files([eval_data_hdfs_root])
    output_list = []

    with hopen(eval_data_files[0], 'rb') as reader:
        for line in reader:
            item_str = line.decode()
            data_item = json.loads(item_str)
            frame_byte_list = prepare_frames(data_item['frames_uniform'])
            output_list.append((frame_byte_list,))

    return output_list[:20]


def build_model(model_path):
    cfg_path = '/mnt/bn/jiny-ttls-i18n-fr1q/MM-Embedding/experiments/config_finetune_mmembed_v3.yaml'
    if cfg_path:
        cfg.update_cfg(cfg_path)

    model = VideoMusicEncoderForDeploy(
        config=cfg,
        gpuwise_nce=False,
    )

    if model_path:
        print(f"Loading the pre-trained checkpoint from {model_path}")
        pretrained_checkpoint = torch.load(model_path, map_location='cpu')
        load_res = model.load_state_dict(pretrained_checkpoint, strict=False)
        print(f"Loading result: {load_res}")

    return model


class FermionModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.decode = fermion_ops.ImageDecodeOp(use_libjpeg_turbo=False, resize_width=192, resize_height=384)
        self.mean_pixel = torch.FloatTensor([0.485, 0.456, 0.406]).cuda()
        self.scale = torch.FloatTensor([0.229, 0.224, 0.225]).cuda()
        self.transpose = [0, 1, 4, 2, 3]
        self.frame_length = 5

        self.model = build_model(model_path).cuda().eval().half()
        self.model = torch.jit.trace(self.model, (torch.randn(1, 5, 3, 384, 192).cuda().half(), torch.ones(1, 5).to(torch.int64).cuda()))

    @torch.jit.export
    def preprocess_single(self, video: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.decode(video[0], is_video=True)[0]
        x = x.float()
        x_batch = x.shape[0]

        x_input = torch.zeros((x_batch, self.frame_length, 384, 192, 3))
        frames_mask = torch.LongTensor([0 for _ in range(self.frame_length)])

        x_input[:, :x.shape[1]] = x
        frames_mask[:x.shape[1]] = 1
        frames_mask = frames_mask.expand(x_batch, -1)

        return (x_input, frames_mask,)

    @torch.jit.export
    def infer_batched(self, video: torch.Tensor,  mask: torch.Tensor) -> Tuple[torch.Tensor]:
        video = torch.div(video, 255)
        video = torch.sub(video, self.mean_pixel)
        video = torch.div(video, self.scale)
        video = video.permute(self.transpose)
        embedding = self.model(video.half(), mask).to(torch.float32)
        return (embedding,)

    @torch.jit.export
    def postprocess_single(self, tensor: torch.Tensor) -> Tuple[torch.Tensor]:
        return (tensor,)

    def forward(self, video: List[List[str]]) -> Tuple[torch.Tensor]:
        x, frames_mask = self.preprocess_single(video)
        embedding, = self.infer_batched(x.cuda(), frames_mask.cuda())
        embedding, = self.postprocess_single(embedding)
        return (embedding,)  # We can return any intermediate result between the pre/infer/post process


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    service_input: List = load_sample_service_input()

    print("######### Loading input finisehd ###########")
    
    # model_path = '/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_music_clip_dedup_v3_large_queue/checkpoint-200000/pytorch_model_ema.bin'
    model_path = '/mnt/bn/jiny-ttls-i18n-fr1q/checkpoints/video_music_text_clip/checkpoint-200000/pytorch_model_ema.bin'
    full_model = FermionModel(model_path)

    # for i, s_in in enumerate(service_input):
    #     in_ = utils.expand_service_input(s_in)
    #     # out_ = script_model.forward(*in_)
    #     out_ = full_model.preprocess_single(*in_)
    #     embed()
    
    output = utils.forward(full_model, service_input)
    if utils.save(full_model, service_input, OUTPUT_DIR):
        logging.info("Done saving. Successful Run!")
    else:
        logging.error("Fail when saving")

    import shutil
    shutil.copy(__file__, OUTPUT_DIR)


    # 
    # model = build_model(model_path)
    # embed()

