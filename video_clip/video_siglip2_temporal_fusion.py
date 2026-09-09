


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
    from .siglip2_temporal_fusion import *
except:
    from siglip2_temporal_fusion import *


from transformers import AutoConfig, AutoTokenizer
from collections import deque

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







class VideoSigLIP2TemporalFusion(torch.nn.Module):
    def __init__(self,
                 model_path,
                 tokenizer_path,
                 gpuwise_nce: bool = True,
                 queue_size: int = 0,
                 use_dwa: bool = True,
                 **kwargs):
        super().__init__()
        # module init
        
        self.logit_scale = nn.Parameter(torch.randn([1]))
        self.logit_bias = nn.Parameter(torch.randn([1]))

        self.logit_scale_fusion = nn.Parameter(torch.randn([1]))
        self.logit_bias_fusion = nn.Parameter(torch.randn([1]))

        self.init_modules(model_path,tokenizer_path)
        self.hidden_size = self.config.vision_config.hidden_size

        self.gpuwise_nce = gpuwise_nce  # GPU 同步 batch 负例
        self.use_dwa = use_dwa

        if queue_size > 0:
            self.register_buffer("video_queue", torch.randn(queue_size, self.hidden_size))
            self.register_buffer("text_queue", torch.randn(queue_size, self.hidden_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        
        self.queue_size = queue_size
        
        #动态权重
        self.hist = [deque(maxlen=2) for _ in range(3)]
        self.T = 2.0

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
    
    @torch.no_grad()
    def dwa_weights(self,losses):
        for i, li in enumerate(losses):
            self.hist[i].append(li.detach())
        if any(len(h) < 2 for h in self.hist):
            w = torch.ones(3, device=losses[0].device) / 3.0
            return w
        ratios = torch.tensor([h[-1] / (h[-2] + 1e-8) for h in self.hist], device=losses[0].device)
        torch.clamp_(ratios, 0.1, 10.0)
        exps = torch.exp(ratios / self.T)
        w = 3 * exps / exps.sum()
        return w
    
    def init_modules(self, model_path, tokenizer_path):
        self.config = AutoConfig.from_pretrained(model_path)

        #preparing labels for LM Loss
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.pad_token_id = tokenizer.pad_token_id

        self.vision_tower = Siglip2VisionTower(config = self.config.vision_config)
        self.text_tower = Siglip2TextTower(config = self.config.text_config)
        self.vision_head = Siglip2MultiheadAttentionPoolingHead(config = self.config.vision_config)

        # fuse title and video
        self.fusion_module = Siglip2FusionHead(config = self.config.vision_config)

        # pooling layer for fused video-title embedding
        self.fusion_head = Siglip2FusionPoolingHead(config = self.config.vision_config)

        # video-grounded text decoder
        self.text_decoder = Siglip2GroundedTextTransformer(config = self.config.text_config)
        print("tokenizer.vocab_size", tokenizer.vocab_size)
        
        self.init_weight()

    def freeze_modules(self, freeze_vision=True, freeze_text=True):
        if freeze_vision:
            for param in self.vision_tower.parameters():
                param.requires_grad = False
            
        if freeze_text:
            for name, param in self.text_tower.named_parameters():
                if name.find("position_embedding_n") != -1:
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
        logit_scale,
        logit_bias, 
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

        
        pixel_values = samples['pixel_values']
        pixel_attention_mask = samples['pixel_attention_mask']
        spatial_shapes = samples['spatial_shapes']

        caption_input_ids = samples['caption_input_ids']#caption的segmentid是zero可以直接传 
        bert_input_ids = samples['bert_input_ids']
        bert_attention_mask = samples['bert_attention_mask']#在text encoder的时候不要传

        title_input_ids = samples['title_input_ids']
        title_segment_ids = samples['title_segment_ids']
        title_attention_mask = samples['title_attention_mask']#在text encoder的时候不要传
        
        ret_dict = dict()
        # caption
        caption_embeds, caption_pooled = self.encode_text(input_ids=caption_input_ids)

        # title
        title_embeds, title_pooled = self.encode_text(input_ids = title_input_ids, segment_ids = title_segment_ids)

        #video
        video_embeds, video_pooled = self.encode_video(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes)

        

        #fusion
        fused_embeds, fused_pooled = self.fuse_video_title(vision_embed = video_embeds,title_embed = title_embeds,  
                                                            vision_attn_mask=None, title_attn_mask = title_attention_mask)
        # video_grounded text decoder
        lm_loss = self.video_grounded_decode(input_ids = bert_input_ids,
                                            attention_mask = bert_attention_mask,
                                            encoder_embeddings = fused_embeds, #fused embeddings
                                            encoder_attention_mask = None)

        ret_dict['LM_loss'] = lm_loss

        

        # loss
        
        if self.gpuwise_nce:
            video_pooled_all = allgather(video_pooled, rank, world_size)
            fused_pooled_all = allgather(fused_pooled, rank, world_size)
            caption_pooled_all = allgather(caption_pooled, rank, world_size)

        else:
            video_pooled_all = video_pooled
            fused_pooled_all = fused_pooled
            caption_pooled_all = caption_pooled

        video_caption_loss = self.calc_siglip_loss(image_embeds=video_pooled_all, text_embeds=caption_pooled_all, logit_scale = self.logit_scale, logit_bias = self.logit_bias)
        ret_dict['video_caption_loss'] = video_caption_loss

        fused_caption_loss = self.calc_siglip_loss(image_embeds=fused_pooled_all, text_embeds=caption_pooled_all, logit_scale = self.logit_scale_fusion, logit_bias = self.logit_bias_fusion)
        ret_dict['fused_caption_loss'] = fused_caption_loss

        if self.use_dwa:
            weights = self.dwa_weights([video_caption_loss,fused_caption_loss,lm_loss])
            loss = weights[0] * video_caption_loss + weights[1] * fused_caption_loss + weights[2] * lm_loss
        else:
            loss = 0
            for key in ret_dict.keys():
                if 'loss' in key:
                    loss += ret_dict[key]

        ret_dict['loss'] = loss

        return ret_dict
    
    def video_grounded_decode(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        encoder_embeddings: Optional[torch.Tensor] = None, #fused embeddings
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        decoder_input_ids = input_ids.clone()
        labels = decoder_input_ids.masked_fill(decoder_input_ids == self.pad_token_id, -100)

        hidden_states, lm_loss = self.text_decoder(input_ids = decoder_input_ids, 
                                                    labels = labels, 
                                                    attention_mask = attention_mask,
                                                    encoder_embeddings = encoder_embeddings,
                                                    encoder_attention_mask = encoder_attention_mask)
        
        return lm_loss
    
    def decode(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        encoder_embeddings: Optional[torch.Tensor] = None, #fused embeddings
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):

        hidden_states = self.text_decoder.decode(input_ids = input_ids,
                                                    attention_mask = attention_mask,
                                                    encoder_embeddings = encoder_embeddings,
                                                    encoder_attention_mask = encoder_attention_mask)
        
        return hidden_states
    
    def fuse_video_title(self,
        vision_embed: torch.Tensor, 
        title_embed: torch.Tensor, 
        vision_attn_mask=None, 
        title_attn_mask = None
        ):

        #B, 256(8*32), 768
        fused_hidden_states = self.fusion_module(vision_embed = vision_embed, title_embed = title_embed, title_attn_mask = title_attn_mask, vision_attn_mask = vision_attn_mask)
        fused_pooler_output = self.fusion_head(fused_hidden_states)

        return fused_hidden_states, fused_pooler_output


    def encode_video(
            self,
            pixel_values: torch.FloatTensor, # B, F, 256, 768
            pixel_attention_mask: torch.Tensor, # B, F, 256
            spatial_shapes: torch.LongTensor, # B, F, 2
        ):
        # encode each frame embeddings
        vision_embeds = self.vision_tower(pixel_values, pixel_attention_mask, spatial_shapes)
        hidden_states, pooler_output = self.vision_head(vision_embeds) # B, 256(8*32), 768

        return hidden_states, pooler_output # B, 768

    def encode_text(self,
                    input_ids: torch.Tensor,
                    segment_ids: torch.Tensor = None,
                    attention_mask: torch.Tensor = None,
                    ):
        
        hidden_states, pooler_output = self.text_tower(input_ids = input_ids, segment_ids = segment_ids, attention_mask = attention_mask)

        return hidden_states, pooler_output # B, 768

if __name__ == "__main__":
    import sys
    from transformers import AutoProcessor, AutoTokenizer
    from PIL import Image
    import requests
    
    model_path = "/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"
    pretrained_path = "/mnt/bn/yexiaoyu-test/checkpoints/debug_6_layer_temporal/checkpoint-30000"
    tokenizer_path = "/mnt/bn/yexiaoyu-test/models/huggingface/bert_uncased"
    videosig = VideoSigLIP2TemporalFusion(model_path = model_path, tokenizer_path = tokenizer_path, gpuwise_nce = False)
    
    print("load pretrained stage one model from: {}".format(pretrained_path))

    videosig.load_state_dict(torch.load(pretrained_path + '/pytorch_model.bin'),strict = False)
    videosig.fusion_head.load_state_dict(videosig.vision_head.state_dict(),strict = False)
    videosig.text_decoder.load_state_dict(videosig.text_tower.text_model.state_dict(),strict = False)
    #videosig.freeze_modules()
    videosig.to("cuda")
    processor = AutoProcessor.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base")
    tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base")
    bert_tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/bert_uncased")

    url = "/mnt/bn/yexiaoyu-test/data/000000039769.jpg"
    image = [Image.open(url) for i in range(16)]
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.unsqueeze(0).view(2,8,256,768)
    pixel_attention_mask = inputs.pixel_attention_mask.unsqueeze(0).view(2,8,256)
    spatial_shapes= inputs.spatial_shapes.unsqueeze(0).view(2,8,2)
    print("pixel_values.shape", pixel_values.shape)


    text = ["two tabby cats sleeping on a pink bed", "two tabby cats sleeping on a pink bed"]

    text_inputs = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_attention_mask = True, return_tensors="pt")
    bert_inputs = bert_tokenizer(text, padding="max_length", truncation=True, max_length=256, return_attention_mask = True, return_tensors="pt")
    caption_input_ids = text_inputs['input_ids']
    title_attention_mask = text_inputs['attention_mask']
    bert_input_ids = bert_inputs['input_ids']
    bert_attention_mask = bert_inputs['attention_mask']#在text encoder的时候不要传
    # print("bert_attention_mask:", bert_attention_mask.shape)

    title_input_ids = caption_input_ids.clone()
    title_segment_ids = torch.zeros_like(title_input_ids)
    #title_attention_mask = bert_attention_mask.clone()


    # output = videosig.encode_video(pixel_values, pixel_attention_mask, spatial_shapes)
    # print("output.shape", output.shape)

    # text_output = videosig.encode_text(text_input_ids, text_attention_masks)
    # print("text_output.shape", text_output.shape)

    samples = {}

    samples['pixel_values'] = pixel_values.to("cuda")
    samples['pixel_attention_mask'] = pixel_attention_mask.to("cuda")
    samples['spatial_shapes'] = spatial_shapes.to("cuda")

    samples['caption_input_ids'] = caption_input_ids.to("cuda")
    samples['bert_input_ids'] = bert_input_ids.to("cuda")
    samples['bert_attention_mask'] = bert_attention_mask.to("cuda")

    samples['title_input_ids'] = title_input_ids.to("cuda")
    samples['title_segment_ids'] = title_segment_ids.to("cuda")
    samples['title_attention_mask'] = title_attention_mask.to("cuda")


    res = videosig(samples)
    print("res", res)

    # for name,param in videosig.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.requires_grad)
    
    # video_input = torch.randn(1, 8, 3, 224, 224)
    # video_mask = torch.ones(1, 8)
    # music_input = torch.randn(1, 128)
    # res = model(video_input, video_mask, music_input, 0, 1)
    # print(res)
    