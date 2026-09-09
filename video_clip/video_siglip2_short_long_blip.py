


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
    from .siglip2_short_long_blip import *
except:
    from siglip2_short_long_blip import *


from transformers import AutoConfig, AutoTokenizer
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







class VideoSigLIP2ShortLongBLIP(torch.nn.Module):
    def __init__(self,
                 model_path,
                 blip_path,
                 gpuwise_nce: bool = True,
                 use_frame_mask: bool = True,
                 queue_size: int = 0,
                 interpolate:int = 6,
                 
                 **kwargs):# use_dwa: bool = True,
        super().__init__()
        # module init
        
        self.logit_scale = nn.Parameter(torch.randn([1]))
        self.logit_bias = nn.Parameter(torch.randn([1]))

        self.logit_scale_fusion = nn.Parameter(torch.randn([1]))
        self.logit_bias_fusion = nn.Parameter(torch.randn([1]))

        self.interpolate = interpolate

        self.init_modules(model_path,blip_path)
        self.hidden_size = self.config.vision_config.hidden_size

        self.gpuwise_nce = gpuwise_nce  # GPU 同步 batch 负例
        self.use_frame_mask = use_frame_mask
        if gpuwise_nce:
            print("using gpuwise_nce")
        if use_frame_mask:
            print("using frame mask")
        # self.use_dwa = use_dwa

        if queue_size > 0:
            self.register_buffer("video_queue", torch.randn(queue_size, self.hidden_size))
            self.register_buffer("text_queue", torch.randn(queue_size, self.hidden_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        
        self.queue_size = queue_size
        
        # #动态权重
        # self.hist = [deque(maxlen=2) for _ in range(3)]
        # self.T = 2.0

    def init_weight(self):

        logit_scale_init = torch.log(torch.tensor(10.0))
        self.logit_scale.data.fill_(logit_scale_init)
        self.logit_bias.data.fill_(-10.0)

        self.logit_scale_fusion.data.fill_(logit_scale_init)
        self.logit_bias_fusion.data.fill_(-10.0)

        def _init_weights(module):
            if isinstance(module, Siglip2VisionEmbeddings):
                width = (
                    self.config.vision_config.hidden_size
                    if isinstance(self.config, Siglip2Config)
                    else self.config.hidden_size
                )
                nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
            elif isinstance(module, nn.Embedding):
                default_flax_embed_init(module.weight)
            elif isinstance(module, Siglip2Attention) or isinstance(module, Siglip2CrossAttention):
                nn.init.xavier_uniform_(module.q_proj.weight)
                nn.init.xavier_uniform_(module.k_proj.weight)
                nn.init.xavier_uniform_(module.v_proj.weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
                nn.init.zeros_(module.q_proj.bias)
                nn.init.zeros_(module.k_proj.bias)
                nn.init.zeros_(module.v_proj.bias)
                nn.init.zeros_(module.out_proj.bias)
            elif isinstance(module, Siglip2MLP):
                nn.init.xavier_uniform_(module.fc1.weight)
                nn.init.xavier_uniform_(module.fc2.weight)
                nn.init.normal_(module.fc1.bias, std=1e-6)
                nn.init.normal_(module.fc2.bias, std=1e-6)
            elif isinstance(module, Siglip2MultiheadAttentionPoolingHead) or isinstance(module, Siglip2FusionPoolingHead):
                nn.init.xavier_uniform_(module.probe_2.data)
                nn.init.xavier_uniform_(module.attention_2.in_proj_weight.data)
                nn.init.zeros_(module.attention_2.in_proj_bias.data)
            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        
        self.apply(_init_weights)
    
    
    def init_modules(self, model_path, blip_path):
        self.blip_config = AutoConfig.from_pretrained(blip_path)
        self.config = AutoConfig.from_pretrained(model_path)

        tokenizer = AutoTokenizer.from_pretrained(blip_path)

        self.pad_token_id = tokenizer.pad_token_id

        self.vision_tower = Siglip2VisionTower(config = self.config.vision_config)
        self.text_tower = Siglip2TextTower(config = self.config.text_config,interpolate = self.interpolate)
        self.vision_head = Siglip2MultiheadAttentionPoolingHead(config = self.config.vision_config)

        # fuse title and video
        self.fusion_module = Siglip2FusionHead(config = self.config.vision_config)

        # pooling layer for fused video-title embedding
        self.fusion_head = Siglip2FusionPoolingHead(config = self.config.vision_config)

        # video-grounded text decoder
        self.text_decoder = Siglip2GroundedTextTransformer(config = self.blip_config, encoder_dim = self.config.vision_config.hidden_size)
        
        self.init_weight()

    def freeze_modules(self, freeze_vision=True, freeze_text=True):
        for name,param in self.text_decoder.named_parameters(): #freeze the LLM decoder
            if name.find("language_projection") != -1 or name.find("language_layernorm") != -1:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        if freeze_vision:
            for param in self.vision_tower.parameters():
                param.requires_grad = False
            
        if freeze_text:
            for name, param in self.text_tower.named_parameters():
                if name.find("position_embedding_n") != -1:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def calc_siglip_loss(self,
        image_embeds: torch.Tensor, 
        text_embeds: torch.Tensor, 
        logit_scale,
        logit_bias, 
        rank,
        world_size,
    ):
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)



        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))

        logit_scale, logit_bias = logit_scale.to(text_embeds.device), logit_bias.to(text_embeds.device)
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

        hdfs_batch = samples['pixel_values'].shape[0]
        pixel_values = torch.cat((samples['pixel_values'],samples['internvid_pixel_values']), dim = 0)
        pixel_attention_mask = torch.cat((samples['pixel_attention_mask'],samples['internvid_pixel_attention_mask']),dim = 0)
        spatial_shapes = torch.cat((samples['spatial_shapes'], samples['internvid_spatial_shapes']), dim = 0)
        if self.use_frame_mask:
            frame_mask = torch.cat((samples['frame_mask'], samples['internvid_frame_mask']), dim = 0)
        else:
            frame_mask = None

        short_input_ids = torch.cat((samples['short_input_ids'],samples['internvid_short_input_ids']),dim = 0) # short-vision contrastive
        long_input_ids = torch.cat((samples['long_input_ids'], samples['internvid_long_input_ids']),dim = 0)# long-vision contrastive


        fusion_input_ids = samples['fusion_input_ids'] # fusion-vision contrastive

        blip_input_ids = samples['blip_input_ids'] #lm loss
        blip_attention_mask = samples['blip_attention_mask'] #lm loss

        title_input_ids = samples['title_input_ids'] # title + sticker + ocr + asr (B, 4, seq_len)
        batch_size, concat, seq_len = title_input_ids.shape
        
        # segment_ids = torch.zeros_like(title_input_ids)# 0: title
        # segment_ids = segment_ids.view(batch_size, concat* seq_len).contiguous()
        # segment_ids[:, 64:128] = 1    # 1: sticker
        # segment_ids[:, 128:192] = 2   # 2: ocr
        # segment_ids[:, 192:256] = 3   # 3: asr
        
        title_input_ids = title_input_ids.view(batch_size * concat, seq_len).contiguous()
        #title_attention_mask = title_attention_mask.view(batch_size * concat, seq_len).contiguous()
        
        ret_dict = dict()
        # short text
        _, short_pooled = self.encode_text(input_ids=short_input_ids)

        # long text
        _, long_pooled = self.encode_text(input_ids=long_input_ids)

        # fusion text
        _, fusion_pooled = self.encode_text(input_ids=fusion_input_ids)

        # title text
        title_embeds, _ = self.encode_text(input_ids = title_input_ids)

        # title + sticker + ocr + asr (B, 4 * seq_len = 256, hidden_dim)
        title_embeds = title_embeds.view(batch_size, concat, seq_len, -1).contiguous()

        title_embeds = title_embeds.view(batch_size, concat * seq_len, -1).contiguous()
        

        #video
        video_embeds, video_pooled = self.encode_video(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes, frame_mask = frame_mask)

        video_for_fusion = video_embeds[:hdfs_batch] #only takes the hdfs dataset
        if self.use_frame_mask:
            frame_mask_for_fusion = frame_mask[:hdfs_batch]
        else:
            frame_mask_for_fusion = None

        #fusion vision
        fusion_vision_embeds, fusion_vision_pooled = self.fuse_video_title(vision_embed = video_for_fusion,title_embed = title_embeds,  
                                                            vision_attn_mask=frame_mask_for_fusion, title_attn_mask = None, segment_ids = None)
        # video_grounded text decoder
        lm_loss = self.video_grounded_decode(input_ids = blip_input_ids,
                                            attention_mask = blip_attention_mask,
                                            encoder_embeddings = fusion_vision_embeds, #fused embeddings
                                            encoder_attention_mask = None)

        ret_dict['lm_loss'] = lm_loss

        

        # loss
        
        if self.gpuwise_nce:
            video_pooled_all = custom_all_gather(video_pooled)
            fusion_vision_pooled_all = custom_all_gather(fusion_vision_pooled)
            short_pooled_all = custom_all_gather(short_pooled)
            long_pooled_all = custom_all_gather(long_pooled)
            fusion_pooled_all = custom_all_gather(fusion_pooled)

        else:
            video_pooled_all = video_pooled
            fusion_vision_pooled_all = fusion_vision_pooled
            short_pooled_all = short_pooled
            long_pooled_all = long_pooled
            fusion_pooled_all = fusion_pooled

        video_short_loss = self.calc_siglip_loss(image_embeds=video_pooled_all, text_embeds=short_pooled_all, logit_scale = self.logit_scale, logit_bias = self.logit_bias, rank = rank, world_size = world_size)
        ret_dict['video_short_loss'] = video_short_loss

        video_long_loss = self.calc_siglip_loss(image_embeds=video_pooled_all, text_embeds=long_pooled_all, logit_scale = self.logit_scale, logit_bias = self.logit_bias,rank = rank, world_size = world_size)
        ret_dict['video_long_loss'] = video_long_loss

        fusion_title_loss = self.calc_siglip_loss(image_embeds=fusion_vision_pooled_all, text_embeds=fusion_pooled_all, logit_scale = self.logit_scale_fusion, logit_bias = self.logit_bias_fusion, rank = rank, world_size = world_size)
        ret_dict['fusion_title_loss'] = fusion_title_loss

        
        loss = 0
        for key in ret_dict.keys():
            if 'loss' in key:
                loss += ret_dict[key]

        ret_dict['loss'] = loss

        return ret_dict
    
    def video_grounded_decode(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_embeddings: Optional[torch.Tensor] = None, #fused embeddings
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        decoder_input_ids = input_ids.clone()

        labels = input_ids.clone()
        labels[input_ids == self.blip_config.image_token_id] = -100
        labels[input_ids == self.pad_token_id] = -100

        lm_loss = self.text_decoder(input_ids = decoder_input_ids, 
                                                    labels = labels, 
                                                    attention_mask = attention_mask,
                                                    encoder_embeddings = encoder_embeddings,
                                                    encoder_attention_mask = encoder_attention_mask)
        
        return lm_loss
    
    def decode(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_embeddings: Optional[torch.Tensor] = None, #fused embeddings
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):

        logits = self.text_decoder.decode(input_ids = input_ids,
                                                    attention_mask = attention_mask,
                                                    encoder_embeddings = encoder_embeddings,
                                                    encoder_attention_mask = encoder_attention_mask)
        
        return logits
    
    def fuse_video_title(self,
        vision_embed: torch.Tensor, 
        title_embed: torch.Tensor, 
        segment_ids: torch.Tensor,
        vision_attn_mask=None, 
        title_attn_mask = None,
        
        ):

        #B, 256(8*32), 768
        fused_hidden_states = self.fusion_module(vision_embed = vision_embed, title_embed = title_embed, title_attn_mask = title_attn_mask, vision_attn_mask = vision_attn_mask,segment_ids = segment_ids)
        fused_pooler_output = self.fusion_head(fused_hidden_states, attention_mask = vision_attn_mask)

        return fused_hidden_states, fused_pooler_output


    def encode_video(
            self,
            pixel_values: torch.FloatTensor, # B, F, 256, 768
            pixel_attention_mask: torch.Tensor, # B, F, 256
            spatial_shapes: torch.LongTensor, # B, F, 2
            frame_mask: torch.Tensor
        ):
        # encode each frame embeddings
        vision_embeds = self.vision_tower(pixel_values, pixel_attention_mask, spatial_shapes)
        hidden_states, pooler_output = self.vision_head(vision_embeds, frame_mask = frame_mask, pixel_mask = pixel_attention_mask)

        return hidden_states, pooler_output # B, 768

    def encode_text(self,
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor = None,
                    ):
        
        hidden_states, pooler_output = self.text_tower(input_ids = input_ids, attention_mask = attention_mask)

        return hidden_states, pooler_output # B, 768

if __name__ == "__main__":
    import sys
    from transformers import AutoProcessor, AutoTokenizer, Blip2Model
    from PIL import Image
    import requests
    
    model_path = "/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"
    pretrained_path = "/mnt/bn/yexiaoyu-test/checkpoints/debug_6_layer_temporal/checkpoint-30000"
    blip_path = "/mnt/bn/yexiaoyu-test/models/huggingface/blip2"
    videosig = VideoSigLIP2TemporalFusionBLIP(model_path = model_path, blip_path = blip_path, gpuwise_nce = False)
    
    print("load pretrained stage one model from: {}".format(pretrained_path))

    videosig.load_state_dict(torch.load(pretrained_path + '/pytorch_model.bin'),strict = False)
    videosig.fusion_head.load_state_dict(videosig.vision_head.state_dict(),strict = False)
    blip_model = Blip2Model.from_pretrained(blip_path)
    videosig.text_decoder.language_model.load_state_dict(blip_model.language_model.state_dict())
    
    videosig.freeze_modules()
    videosig.to("cuda")
    

    for name,param in videosig.named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)
    
    # video_input = torch.randn(1, 8, 3, 224, 224)
    # video_mask = torch.ones(1, 8)
    # music_input = torch.randn(1, 128)
    # res = model(video_input, video_mask, music_input, 0, 1)
    # print(res)
    