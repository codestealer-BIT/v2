import sys
sys.path.append('/mnt/bn/jiny-ttls-i18n-fr1q/ptx__')
from typing import List, Optional, Tuple, Dict
from functools import partial
from einops import rearrange
import logging
import math
import pickle as pkl
from numpy import dtype

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptx.ops.embedding import Embedding
from ptx.ops.layernorm import LayerNormTypes, LayerNorm, FTLayerNorm
from ptx.ops.transformer import (
    BareTransformerEncoder,
    TransformerEncoder, Config as TransformerConfig,
    DeepSpeedTransformerEncoder,
    LSTransformerEncoder,
    PositionWiseFeedForward,
)

try:
    from .embeddings import *
except:
    from embeddings import *


def init_weights(module, initializer_range=0.02):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, (torch.nn.Embedding, Embedding)):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if getattr(module, 'padding_idx', None) is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, LayerNorm):
        # ptx.ops.layernorm init itself
        pass
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def _extend_ffn_moe(
    self: TransformerEncoder,
    n_ffn_types: Optional[int] = 1,
    n_ffn_types_by_layer: Optional[Dict[str, int]] = None,
):
    self._n_ffn_types = n_ffn_types
    self._n_ffn_types_by_layer = n_ffn_types_by_layer
    if n_ffn_types_by_layer:
        for block_idx, block in enumerate(self.blocks):
            if str(block_idx) in n_ffn_types_by_layer:
                n_ffn_types_cur_block = n_ffn_types_by_layer[str(block_idx)]
                if n_ffn_types_cur_block > 1:
                    block._use_n_ffn_types = True
                    block.pwff = nn.ModuleList([PositionWiseFeedForward(block.config) for _ in range(n_ffn_types_cur_block)])
    elif isinstance(n_ffn_types, int) and n_ffn_types > 1:
        for block in self.blocks:
            block._use_n_ffn_types = True
            block.pwff = nn.ModuleList([PositionWiseFeedForward(block.config) for _ in range(n_ffn_types)])


class BertBare(nn.Module):
    default_encoder_class = TransformerEncoder
    default_transformer_config_class = TransformerConfig

    def __init__(self,
                 vocab_size: int,
                 dim: int,
                 dim_ff: int,
                 n_segments: int,
                 n_heads: int,
                 n_layers: int,
                 max_len: int = 512,
                 embedding_dim: int = None,
                 p_drop_hidden: float = 0.1,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12,
                 layernorm_type: str = 'v0',
                 layernorm_fp16: bool = False,
                 use_realformer: bool = False,
                 attention_clamp_inf: bool = False,
                 p_drop_attn: float = 0.1,
                 token_embedding_dim: int = None,
                 embedding_dropout: float = None,
                 act: str = 'gelu_new',
                 padding_index: int = 0,
                 use_deepspeed_transformer: bool = False,
                 use_ls_transformer: bool = False,
                 use_bare_transformer: bool = False,
                 extra_transformer_config: Optional[Dict] = None,
                 omit_other_output: bool = False,
                 pool: bool = True,
                 adaptive_option: Optional[Dict] = None,
                 n_ffn_types: Optional[int] = 1,
                 n_ffn_types_by_layer: Optional[Dict[str, int]] = None,
                 ddp_params_and_buffers_to_ignore: Optional[List[str]] = None,
                 use_dist_token_update: bool = False,
                 max_seq_len: int = 512,
                 dist_token_update_option: Optional[Dict] = None,
                 **kwargs
                 ):
        super().__init__()

        self.initializer_range = initializer_range
        self.vocab_size = vocab_size

        embedding_dim = embedding_dim or dim

        self.embedding = BertEmbedding(
            dim=embedding_dim, vocab_size=self.vocab_size,
            n_segments=n_segments, max_len=max_len,
            p_drop_hidden=p_drop_hidden if embedding_dropout is None else embedding_dropout,
            padding_index=padding_index,
            layer_norm_eps=layer_norm_eps,
            token_embedding_dim=token_embedding_dim,
            layernorm_type=layernorm_type,
            adaptive_option=adaptive_option,
            use_dist_token_update=use_dist_token_update,
            max_seq_len=max_seq_len,
            dist_token_update_option=dist_token_update_option,
        )
        # text
        if layernorm_fp16:
            self.embedding.norm._simply_cast = True

        self.proj_embedding_hidden = None
        if embedding_dim != dim:
            self.proj_embedding_hidden = torch.nn.Linear(embedding_dim, dim)

        self.use_deepspeed_transformer = use_deepspeed_transformer
        self.use_ls_transformer = use_ls_transformer
        self.use_bare_transformer = use_bare_transformer
        self.extra_transformer_config = extra_transformer_config or {}
        tsfm_cls = self.default_encoder_class
        if use_deepspeed_transformer:
            tsfm_cls = DeepSpeedTransformerEncoder
        if use_ls_transformer:
            tsfm_cls = LSTransformerEncoder
        if use_bare_transformer:
            tsfm_cls = BareTransformerEncoder

        self._omit_other_output = omit_other_output

        self.encoder = tsfm_cls(self.default_transformer_config_class(
            n_layers, dim, n_heads, dim_ff, act=act,
            layernorm_type=layernorm_type,
            use_realformer=use_realformer,
            clamp_inf_nan=attention_clamp_inf,
            p_drop_hidden=p_drop_hidden, p_drop_attn=p_drop_attn,
            return_layers=list(range(n_layers + 1)) if not self._omit_other_output else [],
            layer_norm_eps=layer_norm_eps,
            **self.extra_transformer_config,
        ))

        _extend_ffn_moe(self.encoder, n_ffn_types, n_ffn_types_by_layer)

        self._use_moe = self.encoder.config.use_moe
        self._moe_l_aux_factor = self.encoder.config.moe_l_aux_factor

        if pool:
            self.pooler = BertPooler(dict(
                dim=dim,
            )) # add pooler

        self.apply(partial(init_weights, initializer_range=self.initializer_range))

        self.padding_index = padding_index

        if use_dist_token_update:
            if not ddp_params_and_buffers_to_ignore:
                ddp_params_and_buffers_to_ignore = ["embedding.token_embedder_tokens.weight"]
        if ddp_params_and_buffers_to_ignore:
            if not use_dist_token_update:
                assert "embedding.token_embedder_tokens.weight" not in ddp_params_and_buffers_to_ignore
            self._ddp_params_and_buffers_to_ignore = ddp_params_and_buffers_to_ignore

    def forward(self, input_ids, input_segment_ids, input_mask, ret_cls=True):
        embeddings = self.embedding(
            input_ids=input_ids,
            token_type_ids=input_segment_ids,
            position_ids=None,
            embed_only=True,
        )
        return self.partial_embedding_forward(embeddings, input_mask)

    def partial_embedding_forward(self, embeddings, input_mask):
        embeddings = self.embedding.norm(embeddings)
        embeddings = self.embedding.dropout(embeddings)
        embeddings = self.proj_embedding_hidden(embeddings)
        text_encoder_out = self.encoder(embeddings, input_mask)
        return text_encoder_out.last_hidden_state