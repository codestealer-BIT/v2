


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
    from .siglip2_short_long_ema import *
except:
    from siglip2_short_long_ema import *


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







class VideoSigLIP2ShortLongEMA(torch.nn.Module):
    def __init__(self,
                 model_path,
                 load_vision: bool = True,
                 load_text: bool = True,
                 gpuwise_nce: bool = True,
                 queue_size: int = 0,
                 interpolate: int = 4,
                 **kwargs):
        super().__init__()
        # module init
        
        self.interpolate = interpolate
        
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
    
    def init_weight(self):

        logit_scale_init = torch.log(torch.tensor(10.0))
        self.logit_scale.data.fill_(logit_scale_init)
        self.logit_bias.data.fill_(-10.0)

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
    
    def init_modules(self, model_path, load_vision=True, load_text=True):
        self.config = AutoConfig.from_pretrained(model_path)
        self.vision_tower = Siglip2VisionTower(config = self.config.vision_config)
        self.text_tower = Siglip2TextTower(config = self.config.text_config, interpolate = self.interpolate)
        self.vision_head = Siglip2MultiheadAttentionPoolingHead(config = self.config.vision_config)
        self.predict_head = Siglip2PredictorHead(config = self.config.vision_config)
        self.target_head = Siglip2TargetHead(config = self.config.vision_config)
        
        self.init_weight()
        
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

    def calc_masked_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ):
        # how many masked tokens are here
        mask_token_ratio =  target_mask.sum().item() * target.shape[-1]

        # target_mask = target_mask.view(target_mask.shape[0], -1, target_mask.shape[-1]) # B, F, seq_len -> B, F * seq_len
        target_mask = target_mask.unsqueeze(-1).expand_as(target)

        prediction = torch.where(target_mask.bool(), prediction, 0.0) # only calc the targets
        target = torch.where(target_mask.bool(),target, 0.0)

        loss = nn.L1Loss(reduction = 'sum')

        masked_loss = loss(prediction, target) / mask_token_ratio

        return masked_loss




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
        
        ret_dict = dict()
        # text
        short_embs = self.encode_text(input_ids=short_input_ids)
        long_embs = self.encode_text(input_ids=long_input_ids)

        cls_visual_embedding = self.encode_video(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes)

        # loss
        
        if self.gpuwise_nce:
            cls_visual_embedding_all = allgather(cls_visual_embedding, rank, world_size)
            short_embedding_all = allgather(short_embs, rank, world_size)
            long_embedding_all = allgather(long_embs, rank, world_size)
        else:
            cls_visual_embedding_all = cls_visual_embedding
            short_embedding_all = short_embs
            long_embedding_all = long_embs

        short_text_loss = self.calc_siglip_loss(image_embeds=cls_visual_embedding_all, text_embeds=short_embs)
        long_text_loss = self.calc_siglip_loss(image_embeds=cls_visual_embedding_all, text_embeds=long_embs)
        ret_dict['short_text_loss'] = short_text_loss
        ret_dict['long_text_loss'] = long_text_loss

        #self-distillation:
        if model_ema is not None: # is instance of this model
            context_mask = samples['context_mask']
            target_mask = samples['target_mask']

            #use context_mask as attention mask for student encoder so that it can't attend to the target tokens
            masked_embedding = self.vision_tower(pixel_values, context_mask, spatial_shapes)
            prediction = self.predict_head(input_embed = masked_embedding,target_mask = target_mask,context_mask = context_mask, attention_mask = pixel_attention_mask)

            target_embedding = model_ema.vision_tower(pixel_values, pixel_attention_mask, spatial_shapes)
            target = self.target_head(target_embedding)

            masked_loss = self.calc_masked_loss(prediction, target, target_mask)
        
            ret_dict['mask_modeling_loss'] = masked_loss


        

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
    