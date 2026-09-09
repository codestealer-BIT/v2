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
from collections import OrderedDict, defaultdict

import sys
sys.path.append('/mnt/bn/jiny-ttls-i18n-fr1q/MMPretrain')
from utils import is_dist_avail_and_initialized
from IPython import embed

try:
    from .siglip2_short_long import *
except:
    from siglip2_short_long import *


from transformers import AutoConfig


class MusicFeatureFusion(nn.Module):
    """
    音乐特征融合网络，将得到的音乐特征融合成一个综合的音乐表示
    """
    def __init__(self, high_input_dim=768, low_input_dim=128, hidden_dim=512, output_dim=128, add_lyrics_ue=False, add_title_ue=False):
        super(MusicFeatureFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        print("MusicFeatureFusion add_lyrics_ue", add_lyrics_ue)
        print("MusicFeatureFusion add_title_ue", add_title_ue)

        # 每种特征的独立编码器
        self.content_encoder = nn.Sequential(
            nn.Linear(low_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.cover_encoder = nn.Sequential(
            nn.Linear(low_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.attribute_encoder = nn.Sequential(
            nn.Linear(high_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.caption_encoder = nn.Sequential(
            nn.Linear(high_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.total_head_num = 4
        if add_lyrics_ue:
            self.lyrics_encoder = nn.Sequential(
                nn.Linear(low_input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.total_head_num += 1
        else:
            self.lyrics_encoder = None

        if add_title_ue:
            self.title_encoder = nn.Sequential(
                nn.Linear(low_input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.total_head_num += 1
        else:
            self.title_encoder = None

        # 注意力机制用于特征加权
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.total_head_num, self.total_head_num),
            nn.Softmax(dim=1)
        )
        print("total_head_num", self.total_head_num)
        # 融合后的特征处理
        self.fusion_processor = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim, eps=1e-5)  # 最后使用Post LayerNorm稳定表示
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, music_content, music_cover, music_attribute, music_caption, music_lyrics_ue=None, music_title_ue=None):
        """
        前向传播过程
        
        参数:
            music_content: 音乐内容特征, shape: (batch_size, 128)
            music_cover: 音乐封面特征, shape: (batch_size, 128)
            music_attribute: 音乐属性特征, shape: (batch_size, 768)
            music_caption: 音乐caption特征, shape: (batch_size, 768)
            music_lyrics_ue:  音乐歌词ue特征, shape: (batch_size, 768)
            music_title_ue:  音乐title ue特征, shape: (batch_size, 768)
        返回:
            fused_feature: 融合后的音乐表示, shape: (batch_size, output_dim)
        """
        # 编码每种特征
        content_feat = self.content_encoder(music_content)  # (batch_size, hidden_dim)
        cover_feat = self.cover_encoder(music_cover)        # (batch_size, hidden_dim)
        attribute_feat = self.attribute_encoder(music_attribute)
        caption_feat = self.caption_encoder(music_caption)
        lyrics_feat = None
        if self.lyrics_encoder:
            lyrics_feat = self.lyrics_encoder(music_lyrics_ue)
        title_feat = None
        if self.title_encoder:
            title_feat = self.title_encoder(music_title_ue)
        all_muisc_features = [content_feat, cover_feat, attribute_feat, caption_feat, lyrics_feat, title_feat]
        all_muisc_features = [t for t in all_muisc_features if t is not None]
        concat_feat = torch.cat(all_muisc_features, dim=1)  # (batch_size, 4*hidden_dim)

        # 拼接特征用于注意力计算        
        # 计算注意力权重
        attn_weights = self.attention(concat_feat)  # (batch_size, 4)
        attn_weights = attn_weights.unsqueeze(-1)   # (batch_size, 4, 1)
        
        # 应用注意力权重
        weighted_content = content_feat * attn_weights[:, 0, :]  # (batch_size, hidden_dim)
        weighted_cover = cover_feat * attn_weights[:, 1, :]      # (batch_size, hidden_dim)
        weighted_attribute = attribute_feat * attn_weights[:, 2, :]      # (batch_size, hidden_dim)
        weighted_caption = caption_feat * attn_weights[:, 3, :]      # (batch_size, hidden_dim)
        next_attention_pos = 4
        weighted_lyrics = None
        if lyrics_feat is not None:
            weighted_lyrics = lyrics_feat * attn_weights[:, next_attention_pos, :]
            next_attention_pos += 1
        weighted_title = None
        if title_feat is not None:
            weighted_title = title_feat * attn_weights[:, next_attention_pos, :]
            next_attention_pos += 1
        all_weighted_feat = [weighted_content, weighted_cover, weighted_attribute, weighted_caption, weighted_lyrics, weighted_title]
        all_weighted_feat = [t for t in all_weighted_feat if t is not None]
        # 特征融合 - 采用元素级相加与mean pooling融合的组合策略
        sum_fused = sum(all_weighted_feat)  # 元素级相加
        mean_fused = torch.mean(torch.stack(all_weighted_feat), dim=0)  # 融合
        fused = sum_fused + mean_fused  # 组合两种融合结果
        
        # 处理融合后的特征得到最终表示
        final_feat = self.fusion_processor(fused)  # (batch_size, output_dim)
        
        return final_feat

class UserContextFusion(nn.Module):
    def __init__(self, visual_dim, num_langs=512, num_countries=512, emb_dim=32):
        super().__init__()
        self.language_embedding = nn.Embedding(num_langs, emb_dim)
        self.country_embedding = nn.Embedding(num_countries, emb_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim + emb_dim * 2, visual_dim),
            nn.LayerNorm(visual_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(visual_dim)

    def forward(self, x, lang_ids, country_ids):
        lang_feat = self.language_embedding(lang_ids)
        country_feat = self.country_embedding(country_ids)
        
        combined = torch.cat([x, lang_feat, country_feat], dim=-1)
        
        residual = self.mlp(combined)
        return self.norm(x + residual)

class VideoSigLIP2ForMusicOnlyMatchingV2(torch.nn.Module):
    def __init__(self,
                 model_path,
                 gpuwise_nce: bool = True,
                 queue_size: int = 0,
                 interpolate: int = 6,
                 mean_loss: bool = False,
                 use_frame_mask: bool = True,
                 low_quality_score=4,
                 pos_ins_thresh= 6,
                 neg_ins_thresh= 6,
                 add_lyrics_ue= False,
                 add_title_ue= False,
                 add_user_lang_and_country_code: bool = False,
                 huber_loss_delta = 1.0,
                 mbias_delta = 0.01,
                 **kwargs):
        super().__init__()
        print("VideoSigLIP2ForMusicOnlyMatching add_lyrics_ue", add_lyrics_ue)
        print("VideoSigLIP2ForMusicOnlyMatching add_title_ue", add_title_ue)
        print("mbias_delta", mbias_delta)
        # module init
        
        self.interpolate = interpolate
        self.compress_embedding_size = 128
        self.visual_output_dim = 768
        
        self.logit_scale = nn.Parameter(torch.randn([1]))
        self.logit_bias = nn.Parameter(torch.randn([1]))

        self.init_modules(model_path)
        self.hidden_size = self.config.vision_config.hidden_size
        self.gpuwise_nce = gpuwise_nce  # GPU 同步 batch 负例
        self.mean_loss = mean_loss
        self.use_frame_mask = use_frame_mask

        self.add_lyrics_ue = add_lyrics_ue
        self.add_title_ue = add_title_ue
        self.huber_loss_delta = huber_loss_delta
        self.mbias_delta = mbias_delta
        self.music_tower = MusicFeatureFusion(
            high_input_dim=768,
            low_input_dim=self.compress_embedding_size,
            hidden_dim=512,
            output_dim=self.compress_embedding_size,
            add_lyrics_ue=self.add_lyrics_ue,
            add_title_ue=self.add_title_ue,
        )

        if self.compress_embedding_size > 0:
            self.compress_vision_projector = torch.nn.Sequential(
                nn.LayerNorm(self.visual_output_dim, eps=1e-5),
                torch.nn.Linear(self.visual_output_dim, self.visual_output_dim//2),
                torch.nn.GELU(),
                torch.nn.Linear(self.visual_output_dim//2, self.compress_embedding_size)
            )
        self.queue_size = queue_size
        self.low_quality_score = low_quality_score   # 将小于等于此分的样本当作负例

        self.pos_ins_thresh = pos_ins_thresh
        self.neg_ins_thresh = neg_ins_thresh

        self.add_user_lang_and_country_code = add_user_lang_and_country_code
        if self.add_user_lang_and_country_code:
            self.user_fusion_layer = UserContextFusion(
                visual_dim=128,
                num_langs=256, 
                num_countries=256,
                emb_dim=32
            )
        # 初始化权重
        self._initialize_weights()

        label_freq = torch.tensor([0.35, 0.20, 0.31, 0.08, 0.06])

        label_weights = 1.0 / torch.sqrt(label_freq)
        label_weights = label_weights / label_weights.mean()
        self.register_buffer("label_weights", label_weights)

    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        logit_scale_init = torch.log(torch.tensor(2.0))
        self.logit_scale.data.fill_(logit_scale_init)
        self.logit_bias.data.fill_(0.0)
    
    def init_modules(self, model_path):
        self.config = AutoConfig.from_pretrained(model_path)
        self.vision_tower = Siglip2VisionTower(config = self.config.vision_config)
        self.vision_head = Siglip2MultiheadAttentionPoolingHead(config = self.config.vision_config)
        self.text_tower = Siglip2TextTower(config = self.config.text_config, interpolate = self.interpolate)



    def freeze_modules(self, freeze_vision=False, freeze_text=True):
        if freeze_vision:
            for param in self.vision_tower.parameters():
                param.requires_grad = False

            for param in self.vision_head.parameters():
                param.requires_grad = False

        if freeze_text:
            for name, param in self.text_tower.named_parameters():
                param.requires_grad = False

    def calc_regression_loss(self, image_embeds, music_embeds, label):
        image_embeds = F.normalize(image_embeds, dim=-1)
        music_embeds = F.normalize(music_embeds, dim=-1)

        cosine_sim = torch.sum(image_embeds * music_embeds, dim=-1)

        logit_scale = self.logit_scale.exp()

        # 映射到 [1, 5]
        pred_score = 1 + 4 * torch.sigmoid(cosine_sim * logit_scale + self.logit_bias).squeeze(-1)
        weights = self.label_weights[label.long() - 1]

        loss = (weights * F.smooth_l1_loss(
            pred_score, label.float(), reduction='none'
        )).mean()

        # loss = F.smooth_l1_loss(
        #     cosine_sim, label.float(), reduction='none'
        # ).mean()

        return loss, pred_score

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

        if self.use_frame_mask:
            frame_mask = samples['frame_mask']
        else:
            frame_mask = None


        music_item_ids = samples['music_ids']
        music_content_embedding = samples['music_content_ue_vector']
        music_cover_embedding = samples['music_cover_ue_vector']
        music_lyrics_ue = samples['music_lyrics_ue_vector']
        music_title_ue = samples['music_title_ue_vector']

        video_music_match_score = samples['video_music_match_score']
        video_music_match_labels = []

        for i_b in range(len(video_music_match_score)):
            score = video_music_match_score[i_b].item()
            video_music_match_labels.append(score)
        video_music_match_labels = torch.tensor(video_music_match_labels, dtype=music_cover_embedding.dtype).to(pixel_values.device)

        # music caption and attribute feature
        with torch.no_grad():
            music_attribute_input_ids = samples['music_attribute_input_ids']
            music_caption_input_ids = samples['music_caption_input_ids']
            music_attribute_embedding = self.encode_text(input_ids=music_attribute_input_ids)
            music_caption_embedding = self.encode_text(input_ids=music_caption_input_ids)

        music_embedding = self.music_tower(music_content_embedding, music_cover_embedding, music_attribute_embedding, music_caption_embedding, music_lyrics_ue=music_lyrics_ue, music_title_ue=music_title_ue)
        
        ret_dict = dict()
        cls_visual_embedding = self.encode_video(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes, frame_mask=frame_mask)
        if self.compress_embedding_size > 0:
            cls_visual_embedding = self.compress_vision_projector(cls_visual_embedding)

        if self.add_user_lang_and_country_code:
            user_languages = samples["user_lang_code"]
            user_countries = samples["user_country_code"] 
            lang_indices = torch.tensor(
                user_languages,
                dtype=torch.long, device=pixel_values.device
            )
            country_indices = torch.tensor(
                user_countries,
                dtype=torch.long, device=pixel_values.device
            )

            cls_visual_embedding = self.user_fusion_layer(
                cls_visual_embedding, lang_indices, country_indices
            )

        # video music matching loss
        video_music_matching_loss, pred_score = self.calc_regression_loss(cls_visual_embedding, music_embedding, video_music_match_labels)
        ret_dict['video_music_matching_loss'] = video_music_matching_loss
        ret_dict['pred_score'] = pred_score
        loss = 0
        for key in ret_dict.keys():
            if 'loss' in key:
                loss += ret_dict[key]

        ret_dict['loss'] = loss

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

    @torch.no_grad()
    def extract_music_embeds(self, music_content_embedding, music_cover_embedding, music_attribute_input_ids, music_caption_input_ids, music_lyrics_embedding=None, music_title_embedding=None):
        music_attribute_embedding = self.encode_text(input_ids=music_attribute_input_ids)
        music_caption_embedding = self.encode_text(input_ids=music_caption_input_ids)
        music_embedding = self.music_tower(music_content_embedding, music_cover_embedding, music_attribute_embedding, music_caption_embedding, music_lyrics_ue=music_lyrics_embedding, music_title_ue=music_title_embedding)
        music_embedding = F.normalize(music_embedding, dim=-1)
        return music_embedding

    @torch.no_grad()
    def extract_video_embeds(self, 
            pixel_values: torch.FloatTensor, # B, F, 256, 768
            pixel_attention_mask: torch.Tensor, # B, F, 256
            spatial_shapes: torch.LongTensor, # B, F, 2
            frame_mask: torch.Tensor = None, # B, 256):
            user_languages= [],
            user_countries= [],
        ):
        cls_visual_embedding = self.encode_video(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes, frame_mask=frame_mask)
        if self.compress_embedding_size > 0:
            cls_visual_embedding = self.compress_vision_projector(cls_visual_embedding)
        if self.add_user_lang_and_country_code:
            assert(len(user_languages) == len(user_countries))
            assert(len(user_languages) == pixel_values.size(dim=0))
            lang_indices = torch.tensor(
                user_languages,
                dtype=torch.long, device=pixel_values.device
            )
            country_indices = torch.tensor(
                user_countries,
                dtype=torch.long, device=pixel_values.device
            )
            cls_visual_embedding = self.user_fusion_layer(
                cls_visual_embedding, lang_indices, country_indices
            )

        cls_visual_embedding = F.normalize(cls_visual_embedding, dim=-1)

        return cls_visual_embedding


if __name__ == "__main__":
    import sys
    from transformers import AutoProcessor, AutoTokenizer
    from transformers import pipeline
    from PIL import Image
    import requests
    
    model_path = "/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"
    videosig = VideoSigLIP2ForMusicOnlyMatching(model_path=model_path, gpuwise_nce = True)
    videosig.freeze_modules()
    videosig.to("cuda")
    processor = AutoProcessor.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base")
    tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base")

    url = "/mnt/bn/yexiaoyu-test/data/000000039769.jpg"
    image = [Image.open(url) for i in range(8)]
    text = ["two tabby cats sleeping on a pink bed"]
    inputs = processor(images=image, text=text, return_tensors="pt")
    pixel_values = inputs.pixel_values.unsqueeze(0)
    pixel_attention_mask = inputs.pixel_attention_mask.unsqueeze(0)
    spatial_shapes= inputs.spatial_shapes.unsqueeze(0)
    print("pixel_values.shape", pixel_values.shape)
    

    short_inputs = tokenizer(text, padding="max_length", return_tensors="pt")
    # short_input_ids = short_inputs['input_ids']

    # long_inputs = tokenizer(text, padding="max_length", max_length = 196, return_tensors="pt")
    # long_input_ids = long_inputs['input_ids']

    output = videosig.encode_video(pixel_values.to("cuda"), pixel_attention_mask.to("cuda"), spatial_shapes)
    print("output.shape", output.shape)
    embed()