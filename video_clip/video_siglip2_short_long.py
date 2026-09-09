


#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from torch.autograd import Function
from typing import List
from copy import deepcopy
import pickle
import math

import sys
sys.path.append('/opt/tiger/MMPretrain')
from utils import is_dist_avail_and_initialized
from IPython import embed

try:
    from .siglip2_short_long import *
except:
    from siglip2_short_long import *


from transformers import AutoConfig, Siglip2Model, Siglip2TextModel, Siglip2VisionModel



class AllGather(Function):
    """
    A custom PyTorch autograd.Function to perform all_gather with gradient support.
    """
    @staticmethod
    def forward(ctx, tensor):
        output = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(output, tensor)
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        # The gradients for all_gather are essentially a scatter operation.
        # We need to distribute the grad_output back to the individual ranks.
        world_size = dist.get_world_size()

        # Calculate the size of the original tensor
        # grad_output will have shape (batch_size * world_size, ...)
        # We need to divide it by world_size to get the original batch size
        original_tensor_shape = list(grad_output.shape)
        original_tensor_shape[0] = original_tensor_shape[0] // world_size

        # Determine the chunk for the current rank
        rank = dist.get_rank()
        grad_input_for_this_rank = grad_output.narrow(0, rank * original_tensor_shape[0], original_tensor_shape[0])

        return grad_input_for_this_rank

def custom_all_gather(tensor):
    """
    Wrapper function to use the custom AllGather autograd.Function.
    """
    return AllGather.apply(tensor)







class VideoSigLIP2ShortLong(torch.nn.Module):
    def __init__(self,
                 model_path,
                 load_vision: bool = True,
                 load_text: bool = True,
                 gpuwise_nce: bool = True,
                 use_frame_mask: bool = True,
                 queue_size: int = 0,
                 interpolate: int = 4,
                 mean_loss: bool = False,
                 **kwargs):
        super().__init__()
        # module init
        
        self.interpolate = interpolate
        
        self.logit_scale = nn.Parameter(torch.randn([1]))
        self.logit_bias = nn.Parameter(torch.randn([1]))

        self.init_modules(model_path, load_vision, load_text)
        self.hidden_size = self.config.vision_config.hidden_size

        self.gpuwise_nce = gpuwise_nce  # GPU 同步 batch 负例
        self.mean_loss = mean_loss
        self.use_frame_mask = use_frame_mask

        if gpuwise_nce:
            print("using gpuwise_nce")

        if use_frame_mask:
            print("using frame mask")
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
        self.text_tower = Siglip2TextTower(config = self.config.text_config, interpolate = self.interpolate)
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
            
            # scale = self.text_tower.text_model.embeddings.position_embedding_n.weight.shape[0] // text_model.text_model.embeddings.position_embedding.weight.shape[0]
            # self.text_tower.text_model.embeddings.position_embedding_n.weight.data = text_model.text_model.embeddings.position_embedding.weight.data.repeat(scale,1)

            # interpolate long pos embedding
            self.text_tower.load_state_dict(text_model.state_dict(),strict = False)

            positional_embedding_pre = text_model.text_model.embeddings.position_embedding.weight.data
            
            length, dim = positional_embedding_pre.shape
            keep_len = 20

            k = self.interpolate
            posisitonal_embedding_new = torch.zeros([k*length-(k-1)*keep_len, dim], dtype=positional_embedding_pre.dtype)

            for i in range(keep_len):
                posisitonal_embedding_new[i] = positional_embedding_pre[i]
            for i in range(length-1-keep_len):
                posisitonal_embedding_new[k*i + keep_len] = positional_embedding_pre[i + keep_len]
                for j in range(1, k):
                    posisitonal_embedding_new[k*i + j + keep_len] = (k-j)*positional_embedding_pre[i + keep_len]/k + j*positional_embedding_pre[i+1+keep_len]/k
            
            for j in range(k):
                posisitonal_embedding_new[k*length -(k-1)*keep_len - (k-j)] = positional_embedding_pre[length-1] + j*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/k
                    
            self.text_tower.text_model.embeddings.position_embedding_n.weight.data = posisitonal_embedding_new
            print("posisitonal_embedding_new.shape", posisitonal_embedding_new.shape)

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

    def calc_siglip_loss(self,
        image_embeds: torch.Tensor, 
        text_embeds: torch.Tensor,  
        rank,
        world_size,
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

        # if m1_diag1.shape[1] != m1_diag1.shape[0]:
        #     assert world_size * m1_diag1.shape[0] == m1_diag1.shape[1]
        #     m1_diag1[:,rank * m1_diag1.shape[0] : (rank + 1) * m1_diag1.shape[0]] = 2 * torch.eye(m1_diag1.shape[0], device=logits_per_text.device) - 1
        # else:
        m1_diag1.fill_diagonal_(1)

        loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
        if self.mean_loss:
            nll = -torch.mean(loglik, dim=-1)
        else:
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
        short_input_ids = samples['short_input_ids']
        long_input_ids = samples['long_input_ids']

        if self.use_frame_mask:
            frame_mask = samples['frame_mask']
        else:
            frame_mask = None
        
        ret_dict = dict()
        # text
        short_embs = self.encode_text(input_ids=short_input_ids)
        long_embs = self.encode_text(input_ids=long_input_ids)

        cls_visual_embedding = self.encode_video(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes, frame_mask = frame_mask)

        # loss
        
        if self.gpuwise_nce:
            cls_visual_embedding_all = custom_all_gather(cls_visual_embedding)
            short_embs_all = custom_all_gather(short_embs)
            long_embs_all = custom_all_gather(long_embs)
        else:
            cls_visual_embedding_all = cls_visual_embedding
            short_embs_all = short_embs
            long_embs_all = long_embs

        short_text_loss = self.calc_siglip_loss(image_embeds=cls_visual_embedding_all, text_embeds=short_embs_all, rank = rank, world_size=world_size)
        long_text_loss = self.calc_siglip_loss(image_embeds=cls_visual_embedding_all, text_embeds=long_embs_all, rank = rank, world_size=world_size)
        ret_dict['short_text_loss'] = short_text_loss
        ret_dict['long_text_loss'] = long_text_loss
        

        loss = 0
        for key in ret_dict.keys():
            if 'loss' in key:
                loss += ret_dict[key]

        ret_dict['loss'] = loss
        # ret_dict['loss'] = short_text_loss

        return ret_dict

    def encode_video(
            self,
            pixel_values: torch.FloatTensor, # B, F, 256, 768
            pixel_attention_mask: torch.Tensor, # B, F, 256
            spatial_shapes: torch.LongTensor, # B, F, 2
            frame_mask: torch.Tensor = None, # B, 256
        ):
        # encode each frame embeddings
        hidden_states = self.vision_tower(pixel_values, pixel_attention_mask, spatial_shapes)
        pooler_output = self.vision_head(hidden_states, frame_mask = frame_mask, pixel_mask = pixel_attention_mask)

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
    videosig = VideoSigLIP2ShortLong(model_path = model_path, gpuwise_nce = False)
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

    short_inputs = tokenizer(text, padding="max_length", max_length = 64, return_tensors="pt")
    short_input_ids = short_inputs['input_ids']

    long_inputs = tokenizer(text, padding="max_length", max_length = 196, return_tensors="pt")
    long_input_ids = long_inputs['input_ids']

    # output = videosig.encode_video(pixel_values, pixel_attention_mask, spatial_shapes)
    # print("output.shape", output.shape)

    # text_output = videosig.encode_text(text_input_ids, text_attention_masks)
    # print("text_output.shape", text_output.shape)

    samples = {}

    samples['pixel_values'] = pixel_values.to("cuda")
    samples['pixel_attention_mask'] = pixel_attention_mask.to("cuda")
    samples['spatial_shapes'] = spatial_shapes.to("cuda")
    samples['short_input_ids'] = short_input_ids.to("cuda")
    samples['long_input_ids'] = long_input_ids.to("cuda")

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
    