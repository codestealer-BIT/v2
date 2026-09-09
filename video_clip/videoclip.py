#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist

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


class VideoCLIP(torch.nn.Module):
    def __init__(self,
                 config = None,
                 gpuwise_nce: bool = True,
                 **kwargs):
        super().__init__()
        # module init
        self.compress_embedding_size = config.network.get('compress_embedding_size', -1)
        self.enable_patch_embedding = config.get('vision.enable_patch_emb', False)
        self.init_text_config(config)
        self.init_visual_config(config)
        self.init_transfer_module(config)

        self.gpuwise_nce = gpuwise_nce  # GPU 同步 batch 负例
        self.enable_memory_bank = config.get('train.enable_memory_bank', False)
        self.enable_momentum = config.get('network.enable_momentum', False)
        
        self.enable_moco = config.get('network.enable_moco', False)
        self.enable_simclr = config.get('network.enable_simclr', False)

        self.init_loss(config)

    def init_loss(self, config):
        self.main_loss_weight = config.get('network.main_loss_weight', 1.0)
        self.extra_loss_weight = config.get('network.extra_loss_weight', 0.2)

        self.dual_loss = config.get('network.dual_loss.enable', False)
        if self.dual_loss:
            self.dual_loss_weight = config.get('network.dual_loss.dual_loss_weight', 0.5)

        init_tau = config.get('train.init_tau', 0.07)
        tau_clamp = config.get('train.tau_clamp', 4.6051)
        self.calc_nce_loss = LearnableNTXentLoss(init_tau=init_tau, clamp=tau_clamp)

    def init_visual_config(self, config):
        self.visual_type = config.network.visual_type
        self.vision_convert_dtype = config.network.get('vision_convert_dtype', '')
        self.vision_tower = create_vision_tower(self.visual_type, config)
        self.return_visual_embeds = config.network.get('ret_visual_embeds', False)
        self.use_high_level_visual = config.network.get('use_high_level_visual', False)

    def init_text_config(self, config):
        self.text_type = config.network.text_type
        self.text_tower, text_model_config = create_text_tower(self.text_type, config)
        self.text_output_dim = text_model_config['dim']
        self.text_embedding_dim = text_model_config['embedding_dim']
        self.text_model_config = text_model_config

        self.mm_encoder = config.network.get('mm_encoder', False)
        self.mm_ca_encoder = config.network.get('mm_ca_encoder', False)
        self.mm_hl_ca_encoder = config.network.get('mm_hl_ca_encoder', False)
        self.mmencoder_sg_input = config.network.get('mm_encoder_sg_input', '')

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

            if self.mm_hl_ca_encoder:
                self.mm_encoder_ln = config.network.get('mm_encoder_ln', True)
                self.post_hlca_normalize = config.network.get('post_hlca_normalize', False)
                self.fusion_weight = config.network.get('fusion_weight', 1.0)
                self.enable_text_usefulness = config.network.get('enable_text_usefulness', False)
                self.add_visual_for_usefulness = config.network.get('add_visual_for_usefulness', False)
                optimize_softmax = config.network.get('optimize_softmax', False)
                optimize_softmax_v2 = config.network.get('optimize_softmax_v2', False)
                keep_origin = config.network.get('keep_origin', False)
                mm_num_layers = config.network.get('mm_num_layers', 1)
                mm_num_heads = config.network.get('mm_num_heads', 8)
                self.mm_cross_attention = CrossAttentionDecoder(dim=self.visual_output_dim,
                                                                context_dim=self.text_output_dim,
                                                                num_layers=mm_num_layers,
                                                                dim_head=self.text_output_dim,
                                                                heads=mm_num_heads,
                                                                layer_norm=self.mm_encoder_ln,
                                                                optimize_softmax=optimize_softmax,
                                                                keep_origin=keep_origin,
                                                                post_hlca_normalize=self.post_hlca_normalize,
                                                                optimize_softmax_v2=optimize_softmax_v2)

    def forward(self,
                images: torch.Tensor,
                images_mask: torch.Tensor,
                input_ids: torch.Tensor,
                input_mask: torch.Tensor,
                input_segment_ids: torch.Tensor = None,
                mmtext_input_ids: torch.Tensor = None,
                mmtext_input_mask: torch.Tensor = None,
                mmtext_input_segment_ids: torch.Tensor = None,
                mask_mmtext_input: torch.Tensor = None,
                extext_input_ids: torch.Tensor = None,
                extext_input_mask: torch.Tensor = None,
                extext_input_segment_ids: torch.Tensor = None,
                encode_only=False,
                *args, **kwargs):
        
        ret_dict = dict()
        # query
        t_embs = self.encode_text(input_ids=input_ids, input_segment_ids=input_segment_ids, input_mask=input_mask)
        t_emb = t_embs[:, 0].contiguous()

        # title + sticker + hashtag
        et_emb = None
        if extext_input_ids is not None:
            et_embs = self.encode_text(input_ids=extext_input_ids, input_segment_ids=extext_input_segment_ids, input_mask=extext_input_mask)
            et_emb = et_embs[:, 0].contiguous()

        # doc
        doc_ret = self.encode_doc(images=images,
                                  images_mask=images_mask,
                                  mmtext_input_ids=mmtext_input_ids,
                                  mmtext_input_mask=mmtext_input_mask,
                                  mmtext_input_segment_ids=mmtext_input_segment_ids,
                                  mask_mmtext_input=mask_mmtext_input)

        doc_emb = doc_ret['doc_emb']

        visual_embedding = None
        visual_mask = None

        if 'visual_embs' in doc_ret:
            visual_embedding = doc_ret['visual_embs'] # seq embedding
            visual_mask = doc_ret['visual_masks']

        if self.dual_loss:
            if self.mm_hl_ca_encoder:
                cls_visual_embedding = visual_embedding[:, 0].contiguous()
            else:
                cls_visual_embedding = torch.mean(visual_embedding, dim=1)
            if self.compress_embedding_size > 0:
                cls_visual_embedding = self.compress_vision_projector(cls_visual_embedding)

        if encode_only:
            ret_dict['t_emb'] = t_emb
            ret_dict['doc_emb'] = doc_emb
            return ret_dict

        # loss
        if self.gpuwise_nce:
            t_emb = allgather(t_emb, self.trainer.rank, self.trainer.world_size)
            doc_emb = allgather(doc_emb, self.trainer.rank, self.trainer.world_size)
            if et_emb is not None:
                et_emb = allgather(et_emb, self.trainer.rank, self.trainer.world_size)
            if self.dual_loss:
                cls_visual_embedding = allgather(cls_visual_embedding, self.trainer.rank, self.trainer.world_size)
        valid_idx = None

        nce_loss = self.calc_nce_loss(v_emb=doc_emb, t_emb=t_emb, valid_idx=valid_idx)
        ret_dict['nce_loss'] = nce_loss * self.main_loss_weight

        if et_emb is not None:
            extra_nce_loss = self.calc_nce_loss(v_emb=doc_emb, t_emb=et_emb, valid_idx=valid_idx)
            ret_dict['extra_nce_loss'] = extra_nce_loss * self.extra_loss_weight

        if self.dual_loss:
            visual_nce_loss = self.calc_nce_loss(v_emb=cls_visual_embedding, t_emb=t_emb, valid_idx=valid_idx)
            ret_dict['visual_nce_loss'] = visual_nce_loss * self.dual_loss_weight

        loss = 0
        for key in ret_dict.keys():
            if 'loss' in key:
                loss += ret_dict[key]

        ret_dict['loss'] = loss
        ret_dict['nce_temperature'] = self.calc_nce_loss.tau

        return ret_dict

    def encode_doc(self,
                   images: torch.Tensor,
                   images_mask: torch.Tensor,
                   mmtext_input_ids: torch.Tensor = None,
                   mmtext_input_mask: torch.Tensor = None,
                   mmtext_input_segment_ids: torch.Tensor = None,
                   mask_mmtext_input: torch.Tensor = None):
        frames_emb, images_mask = self.encode_image(images, images_mask=images_mask)
        rets = self.transfer_language(self.transfer_type,
                                frames_emb,
                                images_mask,
                                input_ids=mmtext_input_ids,
                                input_segment_ids=mmtext_input_segment_ids,
                                input_mask=mmtext_input_mask,
                                mask_mmtext_input=mask_mmtext_input)

        doc_embedding = rets['doc_emb']
        if self.compress_embedding_size > 0:
            doc_embedding = self.compress_vision_projector(doc_embedding)
        rets['doc_emb'] = doc_embedding.contiguous()
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

    def transfer_language(self, transfer_type, visual_embeds, visual_mask, input_ids=None, input_segment_ids=None, input_mask=None, mask_mmtext_input=None):
        ret_dict = dict()
        
        if 'with_text_encode' in transfer_type:
            # TODO: 改成做cross attention
            visual_embeds = self.v_projector(visual_embeds)
            
            if self.mm_hl_ca_encoder: # 共用text_tower, 最后一层视觉跟文本做ca
                visual_embedding, visual_mask = self.tokenize_visual(visual_embeds, visual_mask)
                visual_embedding = self.coencoder_tower.partial_embedding_forward(visual_embedding, visual_mask)

                if input_ids != None:
                    text_embedding = self.encode_text(input_ids=input_ids, input_mask=input_mask, input_segment_ids=input_segment_ids, ret_compress=False)
                    if mask_mmtext_input is not None:
                        text_mask = mask_mmtext_input
                    else:
                        text_mask = input_mask
                else:
                    text_embedding = torch.zeros((visual_embedding.shape[0], 128, self.text_output_dim), dtype=visual_embedding.dtype, device=visual_embedding.device)
                    text_mask = torch.zeros((visual_embedding.shape[0], 128), dtype=visual_mask.dtype, device=visual_mask.device)

                # query=text, 取跟text相似的视觉 增强视觉;  query=visual，取跟视觉相似的文本，增强视觉； #应该走后者，且两者都应该用sigmoid? 如果要做softmax的话，无论哪个是query，都应该是对文本侧做softmax
                # last_hidden_state = visual_embedding + self.mm_cross_attention(query=text_embedding, context_kv=visual_embedding, x_mask=text_mask, context_mask=visual_mask)
                tv_embedding = self.mm_cross_attention(query=visual_embedding, context_kv=text_embedding, query_mask=visual_mask, context_mask=text_mask)
                fusion_weight = self.fusion_weight
                last_hidden_state = visual_embedding + fusion_weight * tv_embedding

            ret_dict['doc_emb'] = last_hidden_state[:, 0]
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
    model = VideoCLIP(cfg)
    model_state_dict = model.state_dict()
    model_keys = sorted(list(model_state_dict.keys()))
    for key in model_keys:
        print(key, model_state_dict[key].shape)