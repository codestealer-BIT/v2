#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from typing import List
from copy import deepcopy
import pickle

try:
    from .embedding_loss import LearnableNTXentLoss
except:
    from embedding_loss import LearnableNTXentLoss

try:
    from .vit import visual_transformer_B32, visual_transformer_B16
except:
    from vit import visual_transformer_B32, visual_transformer_B16

try:
    from .bertbare import BertBare
except:
    from bertbare import BertBare

try:
    from .embeddings import CrossAttentionDecoder
except:
    from embeddings import CrossAttentionDecoder

import sys
sys.path.append('/mnt/bn/jiny-ttls-i18n-fr1q/MM-Embedding')
from utils import is_dist_avail_and_initialized
from IPython import embed


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size, group=None):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor, group)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None,
            None
        )


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if is_dist_avail_and_initialized():
        tensors_gather = [torch.zeros_like(tensor)
            for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor


@torch.no_grad()
def concat_all_gather_common_object(object_list: List, device, concat=False):
    if is_dist_avail_and_initialized():
        serialized = pickle.dumps(object_list)
        storage = torch.ByteStorage.from_buffer(serialized)
        tensor = torch.ByteTensor(storage).to(device)

        # 2. 同步各 rank 的 tensor 长度
        local_size = torch.tensor([tensor.numel()], device=device, dtype=torch.int64)
        all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        dist.all_gather(all_sizes, local_size)  # 收集所有 rank 的 tensor 长度
        max_size = max([size.item() for size in all_sizes])  # 最大长度用于填充

        # 3. 填充 tensor 至统一长度
        padded_tensor = torch.zeros(max_size, dtype=torch.uint8, device=device)
        padded_tensor[: tensor.numel()] = tensor

        # 4. 跨进程收集所有填充后的 tensor
        all_padded = [torch.zeros_like(padded_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(all_padded, padded_tensor)

        # 5. 反序列化并还原原始数据
        all_data = []
        for i in range(dist.get_world_size()):
            size = all_sizes[i].item()
            byte_tensor = all_padded[i][:size]
            deserialized = pickle.loads(byte_tensor.cpu().numpy().tobytes())
            if concat:
                all_data.extend(deserialized)
            else:
                all_data.append(deserialized)
        
        return all_data
    else:
        return object_list


allgather = AllGather.apply


def create_text_tower(text_type, config, n_dcoder_layers=6, n_layers=None, **kwargs):
    if text_type == 'AlbertV2':
        model_config = json.load(open(config.BERT.config_path,'rb'))
        if n_layers is not None:
            model_config['model']['n_layers'] = n_layers

        for key in kwargs:
            model_config[key].update(kwargs[key])

        return BertBare(**model_config['model']), model_config['model']
    else:
        raise NotImplementedError(f"")


def create_vision_tower(visual_type, config):
    if visual_type == 'VitB32':
        return visual_transformer_B32(
            output_dim=config.vision.visual_dim,
            dropout=config.vision.vit_dropout,
            emb_dropout=config.vision.vit_emb_dropout,
            patch_length=config.vision.patch_length,
            remove=config.vision.get('remove', False))
    elif visual_type == 'VitB16':
        return visual_transformer_B16(
            output_dim=config.vision.visual_dim,
            dropout=config.vision.vit_dropout,
            emb_dropout=config.vision.vit_emb_dropout,
            patch_length=config.vision.patch_length,
            remove=config.vision.get('remove', False))
    else:
        raise NotImplementedError(f"Not implemented for visual type {visual_type}")


class VideoMusicTextCLIP(torch.nn.Module):
    def __init__(self,
                 config = None,
                 gpuwise_nce: bool = True,
                 queue_size: int = 0,
                 **kwargs):
        super().__init__()
        # module init
        self.compress_embedding_size = config.network.get('compress_embedding_size', -1)
        self.enable_patch_embedding = config.get('vision.enable_patch_emb', False)
        self.mm_hl_ca_encoder = config.network.get('mm_hl_ca_encoder', False)
        self.text_type = config.network.text_type

        self.init_visual_config(config)
        self.init_text_config(config)
        self.init_transfer_module(config)
        self.gpuwise_nce = gpuwise_nce  # GPU 同步 batch 负例
        self.enable_memory_bank = config.get('network.enable_memory_bank', False)
        self.enable_momentum = config.get('network.enable_momentum', False)
        self.init_loss(config)

        if queue_size > 0:
            self.register_buffer("video_queue", torch.randn(queue_size, self.compress_embedding_size))
            self.register_buffer("text_queue", torch.randn(queue_size, self.compress_embedding_size))
            self.register_buffer("music_queue", torch.randn(queue_size, self.compress_embedding_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.music_item_ids_queue = list(range(queue_size))    # 记录队列的music item id, 初始时为fake music item_ids
        
        self.queue_size = queue_size

    @torch.no_grad()
    def _dequeue_and_enqueue(self, music_feat, music_item_ids, video_feat, text_feat):
        # gather keys before updating queue
        device = music_feat.device
        music_feats = concat_all_gather(music_feat)
        video_feats = concat_all_gather(video_feat)
        text_feats = concat_all_gather(text_feat)
        all_music_item_ids = concat_all_gather_common_object(music_item_ids, device=device, concat=True)

        batch_size = music_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.music_queue[ptr:ptr + batch_size, :] = music_feats
        self.video_queue[ptr:ptr + batch_size, :] = video_feats
        self.text_queue[ptr:ptr + batch_size, :] = text_feats
        self.music_item_ids_queue[ptr:ptr + batch_size] = all_music_item_ids
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr 

    def init_loss(self, config):
        self.main_loss_weight = config.get('network.main_loss_weight', 1.0)
        self.video_text_loss_weight = config.get('network.video_text_loss_weight', 0.333)
        init_tau = config.get('train.init_tau', 0.07)
        tau_clamp = config.get('train.tau_clamp', 4.6051)
        self.calc_nce_loss = LearnableNTXentLoss(init_tau=init_tau, clamp=tau_clamp)

    def init_visual_config(self, config):
        self.visual_type = config.network.visual_type
        self.vision_tower = create_vision_tower(self.visual_type, config)

    def init_text_config(self, config):
        self.text_type = config.network.text_type
        self.text_tower, text_model_config = create_text_tower(self.text_type, config)
        self.text_output_dim = text_model_config['dim']
        self.text_embedding_dim = text_model_config['embedding_dim']
        self.text_model_config = text_model_config

        if self.compress_embedding_size > 0:
            self.compress_text_projector = torch.nn.Sequential(
                torch.nn.Linear(self.text_output_dim, self.text_output_dim//2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.text_output_dim//2, self.compress_embedding_size)
            )
    
    def init_transfer_module(self, config):
        self.transfer_type = config.network.transfer_type
        visual_dim = config.vision.visual_dim
        
        if 'with_text_encode' in self.transfer_type:
            # text
            try:
                n_layers = config.network.transfer_config.get('co_encoder_layers', 12)
            except Exception as e:
                n_layers = 12

            try:
                bert_overwrite_config = config.network.transfer_config.get("bert_overwrite_config",{})
            except Exception as e:
                bert_overwrite_config = {}
                    
            self.coencoder_tower, coencoder_config = create_text_tower(self.text_type, config, n_layers=n_layers, **bert_overwrite_config)
            
            # visual 
            self.visual_output_dim = coencoder_config["dim"]
            self.need_visual_ln = config.vision.need_visual_ln
            transfer_visual_dim = coencoder_config["embedding_dim"]

            self.v_projector = torch.nn.Sequential(
                torch.nn.Linear(visual_dim, config.vision.proj_middle_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(config.vision.proj_middle_dim, transfer_visual_dim))

            if self.need_visual_ln:
                self.visual_ln = torch.nn.LayerNorm(transfer_visual_dim, eps=1e-5)

            self.img_embedder_tokens = torch.nn.Embedding(1, transfer_visual_dim)
            self.v_segment_embeddings = torch.nn.Embedding(1, transfer_visual_dim)

            if self.compress_embedding_size > 0:
                self.compress_vision_projector = torch.nn.Sequential(
                    torch.nn.Linear(self.visual_output_dim, self.visual_output_dim//2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.visual_output_dim//2, self.compress_embedding_size)
                )

    @torch.no_grad()
    def extract_video_embeds(
        self,
        images: torch.Tensor,   # [N, F, C, H, W],
        images_mask: torch.Tensor,  # [N, L]
    ):
        doc_ret = self.encode_video(
            images=images,
            images_mask=images_mask,
        )
        visual_embedding = doc_ret['visual_embs'] # seq embedding

        cls_visual_embedding = visual_embedding[:, 0].contiguous()
        if self.compress_embedding_size > 0:
            cls_visual_embedding = self.compress_vision_projector(cls_visual_embedding)

        return cls_visual_embedding

    def get_deduplicate_music_embedding(self, cur_music_item_ids):
        all_music_embeddings = self.music_queue.clone().detach()
        all_music_item_ids = deepcopy(self.music_item_ids_queue)

        music_id_set = {}
        cur_music_item_id_set = set(cur_music_item_ids)
        for i in range(len(all_music_item_ids)):
            music_id = all_music_item_ids[i]
            if music_id in cur_music_item_id_set:
                continue
            if music_id not in music_id_set:
                music_id_set[music_id] = [i]
            else:
                music_id_set[music_id].append(i)
        
        filtered_music_embeds = []
        for music_id in music_id_set:
            index = music_id_set[music_id][0]
            filtered_music_embeds.append(all_music_embeddings[index:index+1])
        
        filtered_music_embeds = torch.cat(filtered_music_embeds, dim=0)
        return filtered_music_embeds

    def forward(
            self,
            samples,
            model_ema: torch.nn.Module = None,
            rank: int = 0,
            world_size: int = 1,
        ):

        images = samples['frames']
        images_mask = samples['frames_mask']
        music_embedding = samples['music_ue_vector']
        music_item_ids = samples['music_ids']
        text_input_ids = samples['text_input_ids']
        text_segment_ids = samples['text_segment_ids']
        text_input_masks = samples['text_input_masks']
        
        ret_dict = dict()
        # text
        t_embs = self.encode_text(input_ids=text_input_ids, input_segment_ids=text_segment_ids, input_mask=text_input_masks)
        t_emb = t_embs[:, 0].contiguous()

        # doc
        doc_ret = self.encode_video(
            images=images,
            images_mask=images_mask,
        )
        visual_embedding = doc_ret['visual_embs'] # seq embedding

        cls_visual_embedding = visual_embedding[:, 0].contiguous()
        if self.compress_embedding_size > 0:
            cls_visual_embedding = self.compress_vision_projector(cls_visual_embedding)

        # loss
        if self.queue_size > 0:
            with torch.no_grad():
                # Get music embedding from momentum encoder
                filtered_music_embeds = self.get_deduplicate_music_embedding(music_item_ids)
                music_embedding_all = torch.cat([music_embedding, filtered_music_embeds], dim=0)   

                # Get video embedding from momentum encoder
                doc_ret_m = model_ema.encode_video(
                    images=images,
                    images_mask=images_mask,
                )
                visual_embedding_m = doc_ret_m['visual_embs']
                cls_visual_embedding_m = visual_embedding_m[:, 0].contiguous()
                if self.compress_embedding_size > 0:
                    cls_visual_embedding_m = model_ema.compress_vision_projector(cls_visual_embedding_m)
                cls_visual_embedding_all_m = torch.cat([cls_visual_embedding_m, self.video_queue.clone().detach()], dim=0)

                # get text embedding from momentum encoder
                t_embs_m = model_ema.encode_text(input_ids=text_input_ids, input_segment_ids=text_segment_ids, input_mask=text_input_masks)
                t_emb_m = t_embs_m[:, 0].contiguous()
                t_emb_m_all = torch.cat([t_emb_m, self.text_queue.clone().detach()], dim=0)
            
            self._dequeue_and_enqueue(music_embedding, music_item_ids, cls_visual_embedding_m, t_emb_m)
            cls_visual_embedding_all = cls_visual_embedding

            # calculate the t2v and v2t loss
            valid_idx = None
            t2v_loss = self.calc_nce_loss(v_emb=t_emb, t_emb=cls_visual_embedding_all_m, valid_idx=valid_idx, use_double_loss=False)
            v2t_loss = self.calc_nce_loss(v_emb=cls_visual_embedding_all, t_emb=t_emb_m_all, valid_idx=valid_idx, use_double_loss=False)
            video_text_nce_loss = (t2v_loss + v2t_loss) / 2
            ret_dict['video_text_nce_loss'] = video_text_nce_loss * self.video_text_loss_weight

        else:
            if self.gpuwise_nce:
                cls_visual_embedding_all = allgather(cls_visual_embedding, rank, world_size)
                music_embedding_all = allgather(music_embedding, rank, world_size)
                text_embedding_all = allgather(t_emb, rank, world_size)
            else:
                cls_visual_embedding_all = cls_visual_embedding
                music_embedding_all = music_embedding
                text_embedding_all = t_emb

            valid_idx = None
            video_text_nce_loss = self.calc_nce_loss(v_emb=cls_visual_embedding_all, t_emb=text_embedding_all, valid_idx=valid_idx, use_double_loss=True)
            ret_dict['video_text_nce_loss'] = video_text_nce_loss * self.video_text_loss_weight

        # calculate the video-music loss
        valid_idx = None
        video_music_nce_loss = self.calc_nce_loss(v_emb=cls_visual_embedding_all, t_emb=music_embedding_all, valid_idx=valid_idx, use_double_loss=False)
        ret_dict['video_music_nce_loss'] = video_music_nce_loss * self.main_loss_weight

        loss = 0
        for key in ret_dict.keys():
            if 'loss' in key:
                loss += ret_dict[key]

        ret_dict['loss'] = loss
        ret_dict['nce_temperature'] = self.calc_nce_loss.tau

        return ret_dict

    def encode_video(
            self,
            images: torch.Tensor,
            images_mask: torch.Tensor,
        ):
        # encode each frame embeddings
        frames_emb, images_mask = self.encode_image(images, images_mask=images_mask)

        # Get the video temporal information
        rets = self.transfer_language(
            frames_emb,
            images_mask,
        )

        return rets

    def encode_image(self,
                     images: torch.Tensor,
                     images_mask: torch.Tensor = None,
                     visual_embeds: torch.tensor = None
                     ):
        """
            Pure image encoder.
            TODO: support more embeds: with spatial/timesformers and so on.
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(1)

        if images_mask is None:
            if visual_embeds is None:
                images_mask = torch.ones(images.shape[0:2], device=images.device, dtype=torch.long)
            else:
                images_mask = torch.ones(visual_embeds.shape[0:2], device=visual_embeds.device, dtype=torch.long)

        N, F, C, H, W = images.shape
        images = torch.reshape(images, [N * F, C, H, W])
        ret_dict = self.vision_tower(images, return_dict=self.enable_patch_embedding)
        frames_emb = ret_dict
        frames_emb = frames_emb.reshape([N, F, -1])
        return frames_emb, images_mask

    def encode_text(self,
                    input_ids: torch.Tensor,
                    input_mask: torch.Tensor,
                    input_segment_ids: torch.Tensor = None,
                    ret_compress: bool = True,
                    ):
        if input_segment_ids is None:
            input_segment_ids = torch.zeros_like(input_ids, device=input_ids.device)
        t_out = self.text_tower(input_ids=input_ids, input_segment_ids=input_segment_ids, input_mask=input_mask)

        if ret_compress and self.compress_embedding_size > 0:
            return self.compress_text_projector(t_out)
        return t_out

    def transfer_language(self, visual_embeds, visual_mask):
        ret_dict = dict()
        # TODO: 改成做cross attention
        visual_embeds = self.v_projector(visual_embeds)
        visual_embedding, visual_mask = self.tokenize_visual(visual_embeds, visual_mask)
        visual_embedding = self.coencoder_tower.partial_embedding_forward(visual_embedding, visual_mask)
        ret_dict['visual_embs'] = visual_embedding
        ret_dict['visual_masks'] = visual_mask
        return ret_dict

    def tokenize_visual(self, visual_embeds, visual_mask, position_ids=None):
        # 1. token
        if self.need_visual_ln:
            visual_embeds = self.visual_ln(visual_embeds)
        bsz, visual_length = visual_embeds.size()[:2]
        # 2. 纯视觉因为是没有input_ids这些，需要在最开始补一个 [IMG] 的 token
        # TODO: 可以选择是否要加
        img_embeds = self.gen_img_token_emb(bsz, visual_embeds.device)
        inputs_embeds = torch.cat([img_embeds, visual_embeds], dim=1)
        length = visual_length + 1
        # 3. mask 多加一个 [IMG] 的位置
        img_token_mask = (torch.sum(visual_mask, dim=1, keepdim=True) > 0).long()
        input_mask = torch.cat([img_token_mask, visual_mask], dim=1)

        # 4. 先 project， 去掉，因为已经对齐embedding 维度了
        # if self.project_embedding_first and self.proj_embedding_hidden:
        #     inputs_embeds = self.proj_embedding_hidden(inputs_embeds)

        # 5. position embedding
        if position_ids is None:
            position_ids = torch.arange(0, length, dtype=torch.long, device=visual_embeds.device).expand(bsz, length)
        # position_embeddings = self.v_token_embedder_positions(position_ids) # fix
        #position_embeddings = self.text_tower.embedding.token_embedder_positions(position_ids)
        # NOTE: changed by zxz
        # NOTE: the text tower position embed dim may not same as visual dim here
        position_embeddings = self.coencoder_tower.embedding.token_embedder_positions(position_ids)
        # 6. segment embedding
        segment_embeddings = self.v_segment_embeddings(
            torch.zeros_like(input_mask,
                             device=input_mask.device,
                             dtype=torch.long))
        # 7. 后处理
        embeddings = inputs_embeds + position_embeddings + segment_embeddings
        return embeddings, input_mask

    def gen_img_token_emb(self, bsz, device):
        img_token = torch.zeros((bsz, 1), device=device, dtype=torch.long)
        img_embeds = self.img_embedder_tokens(img_token)
        return img_embeds


if __name__ == "__main__":
    import sys
    from IPython import embed
    sys.path.append('/mnt/bn/jiny-ttls-i18n-fr1q/MM-Embedding')
    from config import cfg
    cfg.update_cfg('/mnt/bn/jiny-ttls-i18n-fr1q/MM-Embedding/experiments/config_finetune_mmembed_v3.yaml')

    print(cfg)
    model = VideoMusicTextCLIP(cfg, gpuwise_nce=False)
    checkpoint_path = '/mnt/bn/jiny-ttls-i18n-fr1q/models/Multimodal-Embedding/model_state_epoch_170000.th'
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    load_res = model.load_state_dict(state_dict, strict=False)
    print(load_res)

    # video_input = torch.randn(1, 8, 3, 224, 224)
    # video_mask = torch.ones(1, 8)
    # music_input = torch.randn(1, 128)
    # res = model(video_input, video_mask, music_input, 0, 1)
    # print(res)
    