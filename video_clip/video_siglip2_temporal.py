


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
import math

import sys
sys.path.append('/opt/tiger/MMPretrain')
from utils import is_dist_avail_and_initialized
from IPython import embed

try:
    from .siglip2_temporal import *
except:
    from siglip2_temporal import *


from transformers import AutoConfig, Siglip2Model, Siglip2TextModel, Siglip2VisionModel

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







class VideoSigLIP2Temporal(torch.nn.Module):
    def __init__(self,
                 model_path,
                 load_vision: bool = True,
                 load_text: bool = True,
                 gpuwise_nce: bool = True,
                 queue_size: int = 0,
                 **kwargs):
        super().__init__()
        # module init
        
        self.logit_scale = nn.Parameter(torch.randn([1]))
        self.logit_bias = nn.Parameter(torch.randn([1]))

        self.init_modules(model_path, load_vision, load_text)
        self.hidden_size = self.config.vision_config.hidden_size

        self.gpuwise_nce = gpuwise_nce  # GPU 同步 batch 负例
        # self.enable_memory_bank = config.get('network.enable_memory_bank', False)
        # self.enable_momentum = config.get('network.enable_momentum', False)
        # self.init_loss(config)

        if queue_size > 0:
            self.register_buffer("video_queue", torch.randn(queue_size, self.hidden_size))
            self.register_buffer("text_queue", torch.randn(queue_size, self.hidden_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        
        self.queue_size = queue_size
    
        
    
    def init_modules(self, model_path, load_vision=True, load_text=True):
        self.config = AutoConfig.from_pretrained(model_path)
        self.vision_tower = Siglip2VisionTower(config = self.config.vision_config)
        self.text_tower = Siglip2TextTower(config = self.config.text_config)
        self.vision_head = Siglip2MultiheadAttentionPoolingHead(config = self.config.vision_config)
        
        logit_scale_init = torch.log(torch.tensor(10.0))
        self.logit_scale.data.fill_(logit_scale_init)
        self.logit_bias.data.fill_(-10.0)

        
        if(load_vision):
            print(f"Loading the pre-trained vision checkpoint from {model_path}")
            vision_model = Siglip2VisionModel.from_pretrained(model_path)
            self.vision_tower.load_state_dict(vision_model.state_dict(),strict = False)
            print(f"Loading the pre-trained vision head from {model_path}")
            self.vision_head.load_state_dict(vision_model.vision_model.head.state_dict(),strict = False)
        if(load_text):
            print(f"Loading the pre-trained text checkpoint from {model_path}")
            text_model = Siglip2TextModel.from_pretrained(model_path)  
            scale = self.text_tower.text_model.embeddings.position_embedding_n.weight.shape[0] // text_model.text_model.embeddings.position_embedding.weight.shape[0]
            self.text_tower.text_model.embeddings.position_embedding_n.weight.data = text_model.text_model.embeddings.position_embedding.weight.data.repeat(scale,1)
            self.text_tower.load_state_dict(text_model.state_dict(),strict = False)

    def freeze_modules(self, freeze_vision=True, freeze_text=True):
        if freeze_vision:
            for param in self.vision_tower.parameters():
                param.requires_grad = False
            
        if freeze_text:
            for name, param in self.text_tower.named_parameters():
                if name.find("position_embedding_n") != -1:
                    param.requires_grad = True
                elif name.find("segment_embedding") != -1:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    @torch.no_grad()
    def _dequeue_and_enqueue(self, video_feat, text_feat):
        # gather keys before updating queue
        device = text_feat.device
        video_feats = concat_all_gather(video_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = text_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.video_queue[ptr:ptr + batch_size, :] = video_feats
        self.text_queue[ptr:ptr + batch_size, :] = text_feats
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr 

    def calc_siglip_loss(self,
        image_embeds: torch.Tensor, 
        text_embeds: torch.Tensor,  
    ):
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # print("image_embeds.shape:",image_embeds.shape)
        # print("text_embeds.shape:",text_embeds.shape)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))

        logit_scale, logit_bias = self.logit_scale.to(text_embeds.device), self.logit_bias.to(text_embeds.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias
        
        # eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
        # m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
        m1_diag1 = -torch.ones(logits_per_text.size()).to(logits_per_text.device)
        m1_diag1.fill_diagonal_(1)
        loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()

        return loss



    def forward(
            self,
            samples,
            model_ema: torch.nn.Module = None,
            rank: int = 0,
            world_size: int = 1,
        ):

        
        pixel_values = samples['pixel_values']
        pixel_attention_mask = samples['pixel_attention_mask']
        spatial_shapes = samples['spatial_shapes']
        text_input_ids = samples['text_input_ids']
        
        ret_dict = dict()
        # text
        t_embs = self.encode_text(input_ids=text_input_ids)

        cls_visual_embedding = self.encode_video(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes)

        # loss
        if self.queue_size > 0:
            with torch.no_grad():
                
                
                cls_visual_embedding_m = model_ema.encode_video(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes)
                cls_visual_embedding_all_m = torch.cat([cls_visual_embedding_m, self.video_queue.clone().detach()], dim=0)

                # get text embedding from momentum encoder
                t_embs_m = model_ema.encode_text(input_ids=text_input_ids)
                t_emb_m_all = torch.cat([t_embs_m, self.text_queue.clone().detach()], dim=0)
            
            self._dequeue_and_enqueue(cls_visual_embedding_m, t_embs_m)
            cls_visual_embedding_all = cls_visual_embedding

            # calculate the siglip loss
            siglip_loss = self.calc_siglip_loss(image_embeds=cls_visual_embedding_all, text_embeds=t_emb_m_all)
            ret_dict['siglip_loss'] = siglip_loss

        else:
            if self.gpuwise_nce:
                cls_visual_embedding_all = allgather(cls_visual_embedding, rank, world_size)
                text_embedding_all = allgather(t_embs, rank, world_size)
            else:
                cls_visual_embedding_all = cls_visual_embedding
                text_embedding_all = t_embs

            siglip_loss = self.calc_siglip_loss(image_embeds=cls_visual_embedding_all, text_embeds=text_embedding_all)
            ret_dict['siglip_loss'] = siglip_loss
        

        loss = 0
        for key in ret_dict.keys():
            if 'loss' in key:
                loss += ret_dict[key]

        ret_dict['loss'] = loss
        ret_dict['logit_scale'] = self.logit_scale
        ret_dict['logit_bias'] = self.logit_bias

        return ret_dict

    def encode_video(
            self,
            pixel_values: torch.FloatTensor, # B, F, 256, 768
            pixel_attention_mask: torch.Tensor, # B, F, 256
            spatial_shapes: torch.LongTensor, # B, F, 2
        ):
        # encode each frame embeddings
        hidden_states = self.vision_tower(pixel_values, pixel_attention_mask, spatial_shapes)
        pooler_output = self.vision_head(hidden_states)

        return pooler_output # B, 768

    def encode_text(self,
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor = None,
                    ):
        
        pooler_output = self.text_tower(input_ids = input_ids, attention_mask = attention_mask)

        return pooler_output # B, 768

if __name__ == "__main__":
    import sys
    from transformers import AutoProcessor, AutoTokenizer
    from PIL import Image
    import requests
    
    model_path = "/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"
    videosig = VideoSigLIP2Temporal(model_path = model_path, gpuwise_nce = False)
    videosig.freeze_modules()
    videosig.to("cuda")
    processor = AutoProcessor.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base")
    tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base")

    url = "/mnt/bn/yexiaoyu-test/data/000000039769.jpg"
    image = [Image.open(url) for i in range(8)]
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.unsqueeze(0)
    pixel_attention_mask = inputs.pixel_attention_mask.unsqueeze(0)
    spatial_shapes= inputs.spatial_shapes.unsqueeze(0)
    print("pixel_values.shape", pixel_values.shape)


    text = ["two tabby cats sleeping on a pink bed"]

    text_inputs = tokenizer(text, padding="max_length", max_length = 256, return_tensors="pt")
    text_input_ids = text_inputs['input_ids']

    # output = videosig.encode_video(pixel_values, pixel_attention_mask, spatial_shapes)
    # print("output.shape", output.shape)

    # text_output = videosig.encode_text(text_input_ids, text_attention_masks)
    # print("text_output.shape", text_output.shape)

    samples = {}

    samples['pixel_values'] = pixel_values.to("cuda")
    samples['pixel_attention_mask'] = pixel_attention_mask.to("cuda")
    samples['spatial_shapes'] = spatial_shapes.to("cuda")
    samples['text_input_ids'] = text_input_ids.to("cuda")

    res = videosig(samples)
    print("res", res)

    for name,param in videosig.named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)
    
    # video_input = torch.randn(1, 8, 3, 224, 224)
    # video_mask = torch.ones(1, 8)
    # music_input = torch.randn(1, 128)
    # res = model(video_input, video_mask, music_input, 0, 1)
    # print(res)
    