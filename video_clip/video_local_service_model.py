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


class VideLocalServiceCLIP(torch.nn.Module):
    def __init__(self,
                 config = None,
                 gpuwise_nce: bool = True,
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

        self.classifier = nn.Sequential(
            nn.Linear(self.compress_embedding_size * 6, self.compress_embedding_size * 12),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(self.compress_embedding_size * 12, self.compress_embedding_size * 6),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(self.compress_embedding_size * 6, 1),
            # nn.Sigmoid(),
        )

        self.criterion = nn.BCEWithLogitsLoss() 

        # init weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.classifier.apply(_basic_init)

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
    def predict_label(
        self,
        samples,
    ):
        images = samples['frames']
        images_mask = samples['frames_mask']
        label = samples['label']

        title_input_ids = samples['title_input_ids']
        title_segment_ids = samples['title_segment_ids']
        title_input_masks = samples['title_input_masks']
        ocr_input_ids = samples['ocr_input_ids']
        ocr_segment_ids = samples['ocr_segment_ids']
        ocr_input_masks = samples['ocr_input_masks']
        asr_input_ids = samples['asr_input_ids']
        asr_segment_ids = samples['asr_segment_ids']
        asr_input_masks = samples['asr_input_masks']
        user_input_ids = samples['user_input_ids']
        user_segment_ids = samples['user_segment_ids']
        user_input_masks = samples['user_input_masks']
        poi_input_ids = samples['poi_input_ids']
        poi_segment_ids = samples['poi_segment_ids']
        poi_input_masks = samples['poi_input_masks']

        
        ret_dict = dict()
        text_input_ids = torch.cat([title_input_ids, ocr_input_ids, asr_input_ids, user_input_ids, poi_input_ids], dim=0)
        text_segment_ids = torch.cat([title_segment_ids, ocr_segment_ids, asr_segment_ids, user_segment_ids, poi_segment_ids], dim=0)
        text_input_masks = torch.cat([title_input_masks, ocr_input_masks, asr_input_masks, user_input_masks, poi_input_masks], dim=0)

        # text
        t_embs = self.encode_text(input_ids=text_input_ids, input_segment_ids=text_segment_ids, input_mask=text_input_masks)
        t_emb = t_embs[:, 0].contiguous()  # [bs * 5, dim]

        # Get the cls token of each feature
        title_cls, ocr_cls, asr_cls, user_cls, poi_cls = torch.split(t_emb, [title_input_ids.shape[0], ocr_input_ids.shape[0], asr_input_ids.shape[0], user_input_ids.shape[0], poi_input_ids.shape[0]], dim=0)

        # doc
        doc_ret = self.encode_video(
            images=images,
            images_mask=images_mask,
        )
        visual_embedding = doc_ret['visual_embs'] # seq embedding

        cls_visual_embedding = visual_embedding[:, 0].contiguous()
        if self.compress_embedding_size > 0:
            cls_visual_embedding = self.compress_vision_projector(cls_visual_embedding)

        pred_logits = self.classifier(torch.cat([cls_visual_embedding, title_cls, ocr_cls, asr_cls, user_cls, poi_cls], dim=1))
        pred_score = F.sigmoid(pred_logits)
        return pred_score

    def forward(
            self,
            samples,
            model_ema: torch.nn.Module = None,
            rank: int = 0,
            world_size: int = 1,
        ):

        images = samples['frames']
        images_mask = samples['frames_mask']
        label = samples['label']

        title_input_ids = samples['title_input_ids']
        title_segment_ids = samples['title_segment_ids']
        title_input_masks = samples['title_input_masks']
        ocr_input_ids = samples['ocr_input_ids']
        ocr_segment_ids = samples['ocr_segment_ids']
        ocr_input_masks = samples['ocr_input_masks']
        asr_input_ids = samples['asr_input_ids']
        asr_segment_ids = samples['asr_segment_ids']
        asr_input_masks = samples['asr_input_masks']
        user_input_ids = samples['user_input_ids']
        user_segment_ids = samples['user_segment_ids']
        user_input_masks = samples['user_input_masks']
        poi_input_ids = samples['poi_input_ids']
        poi_segment_ids = samples['poi_segment_ids']
        poi_input_masks = samples['poi_input_masks']

        
        ret_dict = dict()
        text_input_ids = torch.cat([title_input_ids, ocr_input_ids, asr_input_ids, user_input_ids, poi_input_ids], dim=0)
        text_segment_ids = torch.cat([title_segment_ids, ocr_segment_ids, asr_segment_ids, user_segment_ids, poi_segment_ids], dim=0)
        text_input_masks = torch.cat([title_input_masks, ocr_input_masks, asr_input_masks, user_input_masks, poi_input_masks], dim=0)

        # text
        t_embs = self.encode_text(input_ids=text_input_ids, input_segment_ids=text_segment_ids, input_mask=text_input_masks)
        t_emb = t_embs[:, 0].contiguous()  # [bs * 5, dim]

        # Get the cls token of each feature
        title_cls, ocr_cls, asr_cls, user_cls, poi_cls = torch.split(t_emb, [title_input_ids.shape[0], ocr_input_ids.shape[0], asr_input_ids.shape[0], user_input_ids.shape[0], poi_input_ids.shape[0]], dim=0)

        # doc
        doc_ret = self.encode_video(
            images=images,
            images_mask=images_mask,
        )
        visual_embedding = doc_ret['visual_embs'] # seq embedding

        cls_visual_embedding = visual_embedding[:, 0].contiguous()
        if self.compress_embedding_size > 0:
            cls_visual_embedding = self.compress_vision_projector(cls_visual_embedding)

        pred_logits = self.classifier(torch.cat([cls_visual_embedding, title_cls, ocr_cls, asr_cls, user_cls, poi_cls], dim=1))
        ret_dict['bce_loss'] = self.criterion(pred_logits, label.float())

        loss = 0
        for key in ret_dict.keys():
            if 'loss' in key:
                loss += ret_dict[key]

        ret_dict['loss'] = loss
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
    model = VideLocalServiceCLIP(cfg, gpuwise_nce=False)
    checkpoint_path = '/mnt/bn/jiny-ttls-i18n-fr1q/models/Multimodal-Embedding/model_state_epoch_170000.th'
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    load_res = model.load_state_dict(state_dict, strict=False)
    print(load_res)

    # video_input = torch.randn(1, 8, 3, 224, 224)
    # video_mask = torch.ones(1, 8)
    # music_input = torch.randn(1, 128)
    # res = model(video_input, video_mask, music_input, 0, 1)
    # print(res)
    