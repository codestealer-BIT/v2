from typing import List, Optional, Tuple, Dict
from einops import rearrange
import sys
sys.path.append('/mnt/bn/jiny-ttls-i18n-fr1q/ptx__')

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptx.ops.embedding import Embedding
from ptx.ops.layernorm import LayerNormTypes, LayerNorm
from ptx.ops.adaptive import AdaptiveEmbedding


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class FFN(nn.Module):
    def __init__(self,
                 embed_dims,
                 feedforward_channels=1024,
                 num_fcs=2,
                 active='ReLU',
                 ffn_drop=0):
        super(FFN, self).__init__()
        self.embed_dims = embed_dims
        self.output_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        if active == 'ReLU':
            self.activate = nn.ReLU(inplace=True)
        elif active == 'GELU':
            self.activate = nn.GELU()
        elif active == 'SwiGLU':
            self.activate = SwiGLU()
            self.feedforward_channels = feedforward_channels * 2
            self.output_dims = self.output_dims * 2
        
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, self.feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop)
                )
            )
            in_channels = feedforward_channels

        layers.append(
            nn.Sequential(
                nn.Linear(in_channels, self.output_dims),
                self.activate,
                nn.Dropout(ffn_drop)
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x, identity=None):
        out = self.layers(x)
        if identity is not None:
            return out + identity
        else:
            return out


class BertPooler(torch.nn.Module):
    def __init__(self, option):
        """
        Args:
            option:
                dim:
        """
        super().__init__()
        self.dense = torch.nn.Linear(option.get('dim'), option.get('dim'))
        self.activation = torch.nn.Tanh()
        self.use_ft_linear = bool(option.get('use_ft_linear_amap'))

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        if self.use_ft_linear:
            first_token_tensor = first_token_tensor.contiguous()
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class CrossAttentionDecoder(nn.Module):
    def __init__(self,
                 dim,
                 num_layers=1,
                 context_dim=None,
                 dim_head=64,
                 heads=8,
                 parallel_ff=False,
                 ff_mult=4,
                 layer_norm=True,
                 optimize_softmax=False,
                 keep_origin=False,
                 post_hlca_normalize=False,
                 optimize_softmax_v2=False):
        super().__init__()
        self.num_layers = num_layers

        layers = []
        for _ in range(num_layers):
            layers.append(SpecialCrossAttention(dim=dim,
                                                context_dim=context_dim,
                                                dim_head=dim_head,
                                                heads=heads,
                                                layer_norm=layer_norm,
                                                optimize_softmax=optimize_softmax,
                                                keep_origin=keep_origin,
                                                post_hlca_normalize=post_hlca_normalize,
                                                optimize_softmax_v2=optimize_softmax_v2))
        self.layers = nn.Sequential(*layers)

    def forward(self, query, context_kv, query_mask=None, context_mask=None):
        for i in range(self.num_layers):
            query = self.layers[i](query, context_kv, query_mask=query_mask, context_mask=context_mask)
        return query
    
class SpecialCrossAttention(nn.Module):
    def __init__(self, 
                 dim,
                 context_dim=None,
                 dim_head=64,
                 heads=8,
                 parallel_ff=False,
                 ff_mult=4,
                 layer_norm=True,
                 optimize_softmax=False,
                 keep_origin=False,
                 post_hlca_normalize=False,
                 optimize_softmax_v2=False):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        if context_dim is None:
            context_dim = dim
        
        self.optimize_softmax = optimize_softmax
        self.optimize_softmax_v2 = optimize_softmax_v2
        if self.optimize_softmax_v2:
            assert not self.optimize_softmax
        self.keep_origin = keep_origin
        self.layer_norm = layer_norm
        self.post_hlca_normalize = post_hlca_normalize
        if layer_norm:
            self.norm = LayerNorm(dim)
            # self.context_norm = LayerNorm(context_dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        ) if parallel_ff else None
    
    def _exapnded_mask(self, query_mask, context_mask):
        bsz, src_len = context_mask.size()
        tgt_len = query_mask.shape[1]
        context_expanded_mask = context_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
        if context_mask != None:
            x_expanded_mask = query_mask[:, None, :, None].expand(bsz, 1, tgt_len, src_len)
            expanded_mask = x_expanded_mask * context_expanded_mask
        else:
            expanded_mask = context_expanded_mask
        # inverted_mask = (1.0 - expanded_mask) * -100000
        return expanded_mask

    def forward(self, query, context_kv, query_mask=None, context_mask=None):
        if self.layer_norm:
            x = self.norm(query)
            context = context_kv
        else:
            x = F.normalize(query, dim=-1)
            context = F.normalize(context_kv, dim=-1) #self.context_norm(context_kv)

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q * self.scale

        k, v = self.to_kv(context).chunk(2, dim=-1)

        sim = torch.einsum('b h i d, b j d -> b h i j', q, k)
        # sim = sim - sim.amax(dim=-1, keepdim=True)

        if query_mask is None:
            query_mask = torch.ones(query.shape[:2], device=query.device, dtype=torch.long)

        if context_mask is not None:
            expanded_mask = self._exapnded_mask(query_mask, context_mask)
            sim += (1.0 - expanded_mask) * -1000       

        if self.optimize_softmax:
            sim = sim - sim.amax(dim=-1, keepdim=True)# this is buggy, +1 would be useless if maximum only inside sim
            attn = torch.exp(sim) / (torch.sum(torch.exp(sim), dim=-1, keepdims=True) + 1.0)
        elif self.optimize_softmax_v2:#bug fix: maximum should also consider 0, which accords to +1 after exp
            _maxs = torch.maximum(sim.amax(dim=-1, keepdim=True), torch.tensor(0., device=sim.device))
            sim = sim - _maxs
            attn = torch.exp(sim) / (torch.sum(torch.exp(sim), dim=-1, keepdims=True) + torch.exp(-_maxs))
        else:
            sim = sim.softmax(dim=-1)
            sim_min, _ = (sim + (1 - expanded_mask) * 100).min(dim=-1, keepdims=True)
            attn = (sim - sim_min) * expanded_mask

        if self.keep_origin:
            if self.post_hlca_normalize:
                context_kv = F.normalize(context_kv, dim=-1)
            out = torch.einsum('b h i j, b j d -> b h i d', attn, context_kv).mean(dim=1)
        else:
            out = torch.einsum('b h i j, b j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)

            if self.ff is not None:
                out = out + self.ff(x)

            if self.post_hlca_normalize:
                out = F.normalize(out, dim=-1)
        return out


class BertEmbedding(torch.nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        n_segments: int,
        max_len: int,
        p_drop_hidden: float = 0.1,
        padding_index: int = None,
        layer_norm_eps: float = 1e-12,
        token_embedding_dim: int = None,
        output_dim: int = None,
        adaptive_option: Optional[dict] = None,
        layernorm_type: str = 'v0',
        position_offset: int = 0,
        use_dist_token_update: bool = False,
        max_seq_len: int = 512,
        dist_token_update_option: Optional[Dict] = None,
    ):
        super().__init__()

        token_embedding_dim = token_embedding_dim or dim

        dist_token_update_option = dist_token_update_option or {}
        dist_token_method = dist_token_update_option.get('method', 'v0')
        use_sparse_embedding = dist_token_update_option.get('use_sparse_embedding')
        if use_sparse_embedding is None:
            if not use_dist_token_update:
                use_sparse_embedding = False
            else:
                use_sparse_embedding = dist_token_method != 'v0'

        if token_embedding_dim != dim:
            assert not adaptive_option, 'Cannot use `adaptive_option` when `token_embedding_dim` is set.'
            self.token_embedder_tokens = Embedding(vocab_size, token_embedding_dim, padding_index=padding_index)
            self.token_embedding_proj = nn.Linear(token_embedding_dim, dim)
        else:
            self.token_embedding_proj = None
            if not adaptive_option:
                self.token_embedder_tokens = Embedding(vocab_size, dim, padding_index=padding_index, sparse=use_sparse_embedding)
            else:
                self.token_embedder_tokens = AdaptiveEmbedding(vocab_size, dim, padding_idx=padding_index, **adaptive_option)

        self.position_offset = position_offset or 0
        self.token_embedder_positions = Embedding(max_len + self.position_offset, dim) if max_len else None
        self.token_embedder_segments = Embedding(n_segments, dim) if n_segments else None

        self.dim = dim
        self.output_dim = output_dim or dim
        if self.output_dim != self.dim:
            self.out_proj = nn.Linear(self.dim, self.output_dim, bias=False)
        else:
            self.out_proj = None

        self.norm = (LayerNormTypes[layernorm_type])(self.output_dim, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(p_drop_hidden)

        # if max_len:
        #     # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        #     self.register_buffer('position_ids', torch.arange(max_len + self.position_offset).expand((1, -1)))

        self.add_pos_embedding = True
        self.add_seg_embedding = True

        dist_token_absent_grad_zero = dist_token_update_option.get('set_absent_grad_zero', True)
        dist_token_clip_only = dist_token_update_option.get('use_clip_only', False)
        dist_token_grad_by_count = dist_token_update_option.get('avg_grad_by_count', True)
        dist_token_grad_fp32 = dist_token_update_option.get('use_grad_fp32', True)
        self._use_dist_token_update = use_dist_token_update
        self._max_seq_len = max_seq_len
        self._cached_input_ids = None
        self._use_cached_input_ids = self._use_dist_token_update and (dist_token_method == 'v0')
        if self._use_dist_token_update:
            from ptx.distributed import get_world_size, create_process_group, get_rank

            world_size = get_world_size()
            cur_rank = get_rank()
            pg = None
            if dist_token_method == 'v1':
                pg = create_process_group(backend='gloo')

            def all_gather_embedding_grad_v0(grad):
                if getattr(self, '_cached_input_ids', None) is None:
                    return grad
                emp_grad = torch.zeros_like(grad)
                if dist_token_absent_grad_zero or dist_token_clip_only:
                    cur_ids = self._cached_input_ids.unique()
                    emp_grad[cur_ids] = grad[cur_ids]
                    grad = emp_grad
                    if dist_token_clip_only:
                        self._cached_input_ids = None
                        return grad
                cur_ids = F.pad(self._cached_input_ids, (0, self._max_seq_len - self._cached_input_ids.size(1), 0, 0), 'constant', -1)
                # print(f'local input_ids: {self._cached_input_ids.shape}')
                self._cached_input_ids = None
                all_ids = [torch.full_like(cur_ids, -1) for _ in range(world_size)]
                torch.distributed.all_gather(all_ids, cur_ids)  # use `all_gather_object` instead?
                avg_grad_by = world_size
                if not dist_token_grad_by_count:
                    uni_ids = torch.cat(all_ids).unique()  # dedup, sort
                    if uni_ids[0].item() == -1:
                        uni_ids = uni_ids[1:]  # remove `-1`
                else:
                    uni_ids, uni_counts = torch.cat(all_ids).unique(return_counts=True)  # dedup, sort
                    if uni_ids[0].item() == -1:
                        uni_ids = uni_ids[1:]  # remove `-1`
                        uni_counts = uni_counts[1:]
                    avg_grad_by = uni_counts.unsqueeze(1)
                cur_grad = grad[uni_ids]
                cur_grad[cur_grad != cur_grad] = 0
                # print(f'before allreduce: {uni_ids.shape}, {cur_grad.max()}, {cur_grad.min()}')
                if dist_token_grad_fp32:
                    cur_grad = cur_grad.float()
                torch.distributed.all_reduce(cur_grad)
                # print(f'after allreduce: {uni_ids.shape}, {cur_grad.max()}, {cur_grad.min()}')
                all_grad = cur_grad / avg_grad_by
                if dist_token_grad_fp32:
                    emp_grad[uni_ids] = all_grad.to(dtype=emp_grad.dtype)
                else:
                    emp_grad[uni_ids] = all_grad
                # TODO: check nan/inf ?
                # emp_grad[emp_grad != emp_grad] = 0
                return emp_grad

            def all_gather_embedding_grad_v1(grad):
                # print(f'grad before, {grad.is_coalesced()}')
                # print(f'grad index: {grad._indices()}')
                # print(f'grad value: {grad._values()}')
                if dist_token_grad_fp32:
                    grad = grad.float()
                torch.distributed.all_reduce(grad, group=pg)
                grad = grad / world_size
                if dist_token_grad_fp32:
                    grad = grad.half()
                grad = grad.coalesce()
                # grad = grad.to_dense()  # Cannot, because grad tensor type must not change
                # print(grad)
                return grad

            def all_gather_embedding_grad_v2(grad):
                # print(f'grad before, {grad.is_coalesced()}')
                grad = grad.coalesce()
                all_grads = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(all_grads, grad)
                a_grad = None
                for i_grad in all_grads:
                    if not dist_token_grad_fp32:
                        i_grad = i_grad.to(device=grad.device)
                    else:
                        i_grad = i_grad.to(device=grad.device, dtype=torch.float32)
                    # print(f'grad after, {i_grad.is_coalesced()}')
                    # i_grad = i_grad.coalesce()
                    # i_indices = i_grad.indices()
                    # i_values = i_grad.values()
                    if a_grad is None:
                        a_grad = i_grad
                    else:
                        a_grad.add_(i_grad)
                grad = a_grad / world_size
                if dist_token_grad_fp32:
                    grad = grad.half()
                grad = grad.coalesce()
                # print(grad)
                return grad

            def all_gather_embedding_grad_v3(grad):
                # print(f'grad before, {grad.is_coalesced()}')
                grad = grad.coalesce()
                cur_size = grad.size()
                cur_indices = grad.indices()
                cur_indices = cur_indices.squeeze(0)
                cur_values = grad.values()

                cur_seqlen = cur_indices.size(0)
                all_seqlen = [torch.tensor([0], device=grad.device) for _ in range(world_size)]
                torch.distributed.all_gather(all_seqlen, torch.tensor([cur_seqlen], device=grad.device))

                max_seqlen = torch.cat(all_seqlen).max().item()
                if max_seqlen > cur_seqlen:
                    cur_idx = torch.cat((cur_indices, torch.full((max_seqlen - cur_seqlen,), -1, dtype=cur_indices.dtype, device=cur_indices.device)))
                    cur_val = torch.cat((cur_values, torch.zeros((max_seqlen - cur_seqlen, cur_values.size(1)), dtype=cur_values.dtype, device=cur_values.device)))
                else:
                    cur_idx = cur_indices
                    cur_val = cur_values

                all_ids = [torch.full_like(cur_idx, -1) for _ in range(world_size)]
                torch.distributed.all_gather(all_ids, cur_idx)

                all_grads = [torch.zeros_like(cur_val) for _ in range(world_size)]
                torch.distributed.all_gather(all_grads, cur_val)

                for i, (idx, val) in enumerate(zip(all_ids, all_grads)):
                    if i == cur_rank:
                        continue
                    this_seqlen = all_seqlen[i].item()
                    this_grad = torch.sparse_coo_tensor(idx[:this_seqlen].unsqueeze(0), val[:this_seqlen], size=cur_size, dtype=grad.dtype, device=grad.device)
                    grad.add_(this_grad)

                grad = grad / world_size
                grad = grad.coalesce()
                # print(grad)
                return grad

            dist_token_methods = {
                'v0': all_gather_embedding_grad_v0,
                'v1': all_gather_embedding_grad_v1,
                'v2': all_gather_embedding_grad_v2,
                'v3': all_gather_embedding_grad_v3,
            }

            self.token_embedder_tokens.weight.register_hook(dist_token_methods[dist_token_method])

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, mask=None, embed_only: bool = False):
        if isinstance(input_ids, dict):
            token_type_ids = input_ids.get('segments')
            position_ids = input_ids.get('positions')
            inputs_embeds = input_ids.get('token_embeddings')
            input_ids = input_ids.get('tokens')

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if self.training and self._use_cached_input_ids:
            self._cached_input_ids = input_ids

        if inputs_embeds is None:
            embeddings = self.token_embedder_tokens(input_ids)
            if self.token_embedding_proj is not None:
                embeddings = self.token_embedding_proj(embeddings)
        else:
            embeddings = inputs_embeds

        if (self.token_embedder_positions is not None) and self.add_pos_embedding:
            if position_ids is None:
                # slice_pos_ids模式会导致evaluation时发生数值混乱，导致self.token_embedder_positions(position_ids)越界【原因未明】
                # position_ids = slice_pos_ids(self.position_ids, input_ids)
                position_ids = torch.arange(input_shape[-1]).expand((1, -1)).to(input_ids.device)
            if self.position_offset:
                position_ids = position_ids + self.position_offset
            position_embeddings = self.token_embedder_positions(position_ids)
            embeddings += position_embeddings

        if (self.token_embedder_segments is not None) and self.add_seg_embedding:
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=embeddings.device)
            token_type_embeddings = self.token_embedder_segments(token_type_ids)
            embeddings += token_type_embeddings

        if embed_only:
            return embeddings

        if self.out_proj is not None:
            embeddings = self.out_proj(embeddings)

        embeddings = self.norm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            embeddings *= mask.to(embeddings.dtype)

        embeddings = self.dropout(embeddings)
        return embeddings
