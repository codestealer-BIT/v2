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
from .video_siglip2_for_music_only_matching import VideoSigLIP2ForMusicOnlyMatching
from .video_siglip2_for_music_negative_filtering import VideoSigLIP2ForMusicNegativeFiltering

class AllGather(torch.autograd.Function):
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


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors, supporting different batch sizes across ranks.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not is_dist_avail_and_initialized():
        return tensor

    world_size = dist.get_world_size()
    
    # 获取当前rank上tensor的batch size
    local_batch_size = tensor.size(0)
    
    # 收集所有rank的batch size
    all_batch_sizes = [torch.tensor([0], device=tensor.device) for _ in range(world_size)]
    dist.all_gather(all_batch_sizes, torch.tensor([local_batch_size], device=tensor.device))
    all_batch_sizes = [x.item() for x in all_batch_sizes]
    
    # 为所有tensor创建存储
    if tensor.ndim == 0:
        # 处理标量tensor的情况
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        return torch.stack(gathered_tensors)
    else:
        # 为不同大小的tensor创建合适的存储空间
        gathered_tensors = []
        for batch_size in all_batch_sizes:
            # 创建与当前rank tensor类型和设备相同，但batch size不同的tensor
            tensor_shape = (batch_size,) + tensor.size()[1:]
            gathered_tensor = torch.zeros(tensor_shape, dtype=tensor.dtype, device=tensor.device)
            gathered_tensors.append(gathered_tensor)
        
        # 收集所有tensor
        dist.all_gather(gathered_tensors, tensor)
        
        # 拼接所有收集到的tensor
        return torch.cat(gathered_tensors, dim=0)


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


class MusicFeatureFusion(nn.Module):
    """
    音乐特征融合网络，将得到的音乐特征融合成一个综合的音乐表示
    """
    def __init__(self, high_input_dim=768, low_input_dim=128, hidden_dim=512, output_dim=128):
        super(MusicFeatureFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
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
        
        # 注意力机制用于特征加权
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 4, 4),
            nn.Softmax(dim=1)
        )
        
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
    
    def forward(self, music_content, music_cover, music_attribute, music_caption):
        """
        前向传播过程
        
        参数:
            music_content: 音乐内容特征, shape: (batch_size, 128)
            music_cover: 音乐封面特征, shape: (batch_size, 128)
            music_attribute: 音乐属性特征, shape: (batch_size, 768)
            music_caption: 音乐caption特征, shape: (batch_size, 768)
        返回:
            fused_feature: 融合后的音乐表示, shape: (batch_size, output_dim)
        """
        # 编码每种特征
        content_feat = self.content_encoder(music_content)  # (batch_size, hidden_dim)
        cover_feat = self.cover_encoder(music_cover)        # (batch_size, hidden_dim)
        attribute_feat = self.attribute_encoder(music_attribute)
        caption_feat = self.caption_encoder(music_caption)
        
        # 拼接特征用于注意力计算
        concat_feat = torch.cat([content_feat, cover_feat, attribute_feat, caption_feat], dim=1)  # (batch_size, 4*hidden_dim)
        
        # 计算注意力权重
        attn_weights = self.attention(concat_feat)  # (batch_size, 4)
        attn_weights = attn_weights.unsqueeze(-1)   # (batch_size, 4, 1)
        
        # 应用注意力权重
        weighted_content = content_feat * attn_weights[:, 0, :]  # (batch_size, hidden_dim)
        weighted_cover = cover_feat * attn_weights[:, 1, :]      # (batch_size, hidden_dim)
        weighted_attribute = attribute_feat * attn_weights[:, 2, :]      # (batch_size, hidden_dim)
        weighted_caption = caption_feat * attn_weights[:, 3, :]      # (batch_size, hidden_dim)
        
        # 特征融合 - 采用元素级相加与mean pooling融合的组合策略
        sum_fused = weighted_content + weighted_cover + weighted_attribute + weighted_caption  # 元素级相加
        mean_fused = torch.mean(torch.stack([weighted_content, weighted_cover, weighted_attribute, weighted_caption]), dim=0)  # 融合
        fused = sum_fused + mean_fused  # 组合两种融合结果
        
        # 处理融合后的特征得到最终表示
        final_feat = self.fusion_processor(fused)  # (batch_size, output_dim)
        
        return final_feat


class VideoMusicMatchingLoss(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_category=3):
        """
        >= 7 : high quality
        4, 5, 6: meduium quality
        1, 2, 3: very bad
        """
        super(VideoMusicMatchingLoss, self).__init__()
        # 定义第一个线性层，输入维度是两个输入向量维度之和
        self.layer1 = nn.Linear(input_dim * 2, hidden_dim)
        # 定义激活函数
        self.activation = nn.GELU()
        # 定义输出层
        self.layer2 = nn.Linear(hidden_dim, num_category)

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, video_embeds, music_embeds, labels):
        # 沿维度1拼接两个输入向量
        combined_vec = torch.cat((video_embeds, music_embeds), dim=1) #
        # 通过隐藏层和激活函数
        hidden_output = self.activation(self.layer1(combined_vec))
        # 通过输出层得到最终的 logits
        outputs = self.layer2(hidden_output)

        loss = self.loss_function(outputs, labels)

        return loss

class UserContextAdapter(nn.Module):
    def __init__(self,
                 cls_dim: int = 128,
                 lang_emb_dim: int = 32,
                 country_emb_dim: int = 32,
                 num_langs: int = 256,
                 num_countries: int = 256,
                 hidden_dim: int = 128,
                 method: str = 'gate'):
        super().__init__()
        assert method in ('gate', 'film', 'residual'), 'method must be one of gate/film/residual'
        self.method = method
        self.cls_dim = cls_dim

        # embedding tables
        self.language_embedding = nn.Embedding(num_langs, lang_emb_dim)
        self.country_embedding = nn.Embedding(num_countries, country_emb_dim)

        # shared MLP to produce a compact user context vector
        self.context_mlp = nn.Sequential(
            nn.Linear(lang_emb_dim + country_emb_dim, hidden_dim),
            nn.GELU()
        )

        if method == 'gate':
            # gate -> produce scalar gate per-dim or per-vector
            # two variants: per-dim (vector gate) or per-scalar. we use per-dim for more flexibility
            self.gate_proj = nn.Sequential(
                nn.Linear(hidden_dim, cls_dim),
                nn.Sigmoid()
            )
            # adapter to produce residual user vector in cls_dim
            self.res_proj = nn.Sequential(
                nn.Linear(hidden_dim, cls_dim),
                nn.Tanh()
            )
        elif method == 'film':
            # produce scale and bias
            self.scale_proj = nn.Linear(hidden_dim, cls_dim)
            self.bias_proj = nn.Linear(hidden_dim, cls_dim)
        elif method == 'residual':
            # project context to cls_dim and add (with learnable multiplier)
            self.res_proj = nn.Sequential(
                nn.Linear(hidden_dim, cls_dim),
                nn.GELU()
            )
            self.alpha = nn.Parameter(torch.tensor(0.0, requires_grad=True))  # start from no-change

        # init small weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, cls_visual_embedding: torch.Tensor, lang_indices: torch.Tensor, country_indices: torch.Tensor):
        """
        cls_visual_embedding: (B, cls_dim)
        lang_indices: (B,) long
        country_indices: (B,) long
        返回: (B, cls_dim)
        """
        B = cls_visual_embedding.size(0)
        assert lang_indices.dim() == 1 and country_indices.dim() == 1
        device = cls_visual_embedding.device

        lang_emb = self.language_embedding(lang_indices.to(device))
        country_emb = self.country_embedding(country_indices.to(device))

        ctx = torch.cat([lang_emb, country_emb], dim=-1)  # (B, lang_emb_dim+country_emb_dim)
        ctx = self.context_mlp(ctx)  # (B, hidden_dim)

        if self.method == 'gate':
            gate = self.gate_proj(ctx)  # (B, cls_dim) in (0,1)
            user_res = self.res_proj(ctx)  # (B, cls_dim) in (-1,1)
            # 混合：保守地把 vision 与 user_res 融合
            out = cls_visual_embedding * (1 - gate) + user_res * gate
            return out

        elif self.method == 'film':
            scale = self.scale_proj(ctx)  # (B, cls_dim)
            bias = self.bias_proj(ctx)    # (B, cls_dim)
            # small residual scale init near 0
            out = cls_visual_embedding * (1 + scale) + bias
            return out

        elif self.method == 'residual':
            user_res = self.res_proj(ctx)
            out = cls_visual_embedding + self.alpha * user_res
            return out


class VideoSigLIP2ForMusicNegativeFilteringBM(torch.nn.Module):
    def __init__(self,
                 model_path,
                 gpuwise_nce: bool = True,
                 queue_size: int = 0,
                 interpolate: int = 6,
                 mean_loss: bool = False,
                 use_frame_mask: bool = True,
                 low_quality_score=4,
                 historical_model_path: str = None,            # 新增：历史模型权重路径
                 historical_score_threshold: float = 0.1,      # 新增：历史模型判定阈值（可调整）
                 false_positive_filter: bool = None, # 对正样本中低于low_quality_score的处理方式, None 不做处理, True 过滤, False 设为负样本
                 **kwargs):
        super().__init__()
        # module init
        print("start init VideoSigLIP2ForMusicNegativeFilteringBM")
        print("historical_model_path", historical_model_path)
        print("historical_score_threshold", historical_score_threshold)

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

        self.music_tower = MusicFeatureFusion(
            high_input_dim=768,
            low_input_dim=self.compress_embedding_size,
            hidden_dim=512,
            output_dim=self.compress_embedding_size,
        )

        if self.compress_embedding_size > 0:
            self.compress_vision_projector = torch.nn.Sequential(
                nn.LayerNorm(self.visual_output_dim, eps=1e-5),
                torch.nn.Linear(self.visual_output_dim, self.visual_output_dim//2),
                torch.nn.GELU(),
                torch.nn.Linear(self.visual_output_dim//2, self.compress_embedding_size)
            )

        self.queue_size = queue_size
        self.cal_video_music_matching_loss = VideoMusicMatchingLoss(input_dim=self.compress_embedding_size, num_category=3)
        self.low_quality_score = low_quality_score   # 将小于等于此分的样本当作负例
        self.false_positive_filter = false_positive_filter # 对正样本中低于low_quality_score的处理方式, None 不做处理, True 过滤, False 设为负样本
        # 历史模型相关
        self.historical_model = None
        self.historical_score_threshold = historical_score_threshold
        print("historical_model_path check", historical_model_path)

        # cls_dim = self.compress_embedding_size if self.compress_embedding_size > 0 else self.visual_output_dim
        # 初始化权重
        self._initialize_weights()
        # self.user_context_adapter = UserContextAdapter(cls_dim=cls_dim, lang_emb_dim=32, country_emb_dim=32, method=user_info_fusion_method)

        self.historical_model_loaded = False
        if historical_model_path is not None:
            print("historical_model_path is not None")
            hist_model = VideoSigLIP2ForMusicOnlyMatching(
                model_path=model_path,
                gpuwise_nce=False,
                queue_size=0,
                interpolate=self.interpolate,
                mean_loss=self.mean_loss,
                use_frame_mask=self.use_frame_mask,
                low_quality_score=self.low_quality_score,
            )
            state_dict = torch.load(historical_model_path, map_location="cpu")
            hist_model.load_state_dict(state_dict, strict=True)
            # 冻结并设为 eval
            hist_model.eval()
            for p in hist_model.parameters():
                p.requires_grad = False
            device = self.logit_scale.device
            hist_model.to(device)
            object.__setattr__(self, "historical_model", hist_model)
            self._non_persistent_buffers_set.add("historical_model")

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
    
    def init_modules(self, model_path):
        self.config = AutoConfig.from_pretrained(model_path)
        self.vision_tower = Siglip2VisionTower(config = self.config.vision_config)
        self.vision_head = Siglip2MultiheadAttentionPoolingHead(config = self.config.vision_config)
        self.text_tower = Siglip2TextTower(config = self.config.text_config, interpolate = self.interpolate)
        
        logit_scale_init = torch.log(torch.tensor(10.0))
        self.logit_scale.data.fill_(logit_scale_init)
        self.logit_bias.data.fill_(-10.0)

    def freeze_modules(self, freeze_vision=False, freeze_text=True):
        if freeze_vision:
            for param in self.vision_tower.parameters():
                param.requires_grad = False

            for param in self.vision_head.parameters():
                param.requires_grad = False

        if freeze_text:
            for name, param in self.text_tower.named_parameters():
                param.requires_grad = False

    def calc_siglip_loss(self,
        image_embeds: torch.Tensor, 
        music_embeds: torch.Tensor,  
        music_mask: torch.Tensor,  
        matching_scores: torch.Tensor, 
        rank,
        world_size,
        hist_video_embeds: torch.Tensor = None,
        hist_music_embeds: torch.Tensor = None
    ):
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        music_embeds = music_embeds / music_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_text = torch.matmul(music_embeds, image_embeds.t().to(music_embeds.device))
        logit_scale, logit_bias = self.logit_scale.to(music_embeds.device), self.logit_bias.to(music_embeds.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

        # ===== 构造正负mask =====
        m1_diag1 = -torch.ones_like(logits_per_text)
        m1_diag1[music_mask.to(logits_per_text.device)] = 1   # 正例 +1，其余负例 -1

        # exclude_mask: 标记哪些 pair 要过滤（True 表示过滤）
        exclude_mask = torch.zeros_like(logits_per_text, dtype=torch.bool, device=logits_per_text.device)
        filtered_negatives = 0
        if self.false_positive_filter is not None:
            if self.false_positive_filter == False: # 正例中小于low_quality_score的pair置为负例
                matching_score_mask = torch.diag(matching_scores <= self.low_quality_score)
                m1_diag1[matching_score_mask] = -1 
            else: # 正例中小于low_quality_score的pair mask 掉
                matching_score_mask = torch.diag(matching_scores <= self.low_quality_score)
                exclude_mask[matching_score_mask] = True 
            
            # # third option: use both mask & set negative
            # true_neg_threshold = 2
            # neg_diag = torch.diag(matching_scores <= true_neg_threshold)
            # m1_diag1[neg_diag] = -1 
            # mask_diag = torch.diag((matching_scores > true_neg_threshold) & (matching_scores <= self.low_quality_score))
            # exclude_mask[mask_diag] = True # not using this for training


        with torch.no_grad():
            N = exclude_mask.size(0)
            diag_mask = torch.eye(N, dtype=torch.bool, device=exclude_mask.device)
            diag_exclude = exclude_mask[diag_mask]  # [N]，对角线上被mask的标志
            kept_pos_count = (~diag_exclude).sum().item()  # 取反后求和 = 未被mask的数量
        # print("未被过滤正例数", kept_pos_count)
        # ===== 历史模型过滤负例 =====
        if (hist_video_embeds is not None) and (hist_music_embeds is not None) and (self.historical_model is not None):
            hist_music = F.normalize(hist_music_embeds, dim=-1)
            hist_video = F.normalize(hist_video_embeds, dim=-1)

            with torch.no_grad():
                hist_sims = torch.mm(hist_music, hist_video.t())

            # 找出负例 (非对角线)
            neg_mask = ~torch.eye(hist_sims.size(0), dtype=torch.bool, device=hist_sims.device)

            off_diag_sims = hist_sims[neg_mask]
            # print("数量:", off_diag_sims.numel())
            # print("均值:", off_diag_sims.mean().item())
            # print("方差:", off_diag_sims.var().item())
            # print("最小值:", off_diag_sims.min().item())
            # print("最大值:", off_diag_sims.max().item())
            # 在负例中，历史模型打分 > 阈值的进行过滤
            high_sim_mask = (hist_sims > self.historical_score_threshold) & neg_mask
            # print("hist_sims:", hist_sims, hist_sims.shape)
            # print("self.historical_score_threshold:", self.historical_score_threshold)
            # print("high_sim_mask:", high_sim_mask, high_sim_mask.shape)
            exclude_mask[high_sim_mask] = True
            filtered_negatives = high_sim_mask.sum().item()
            # print("过滤的负例数：", filtered_negatives)
        # ===== 打点统计 =====
        total_pairs = m1_diag1.numel()
        total_negatives = total_pairs - m1_diag1.size(0)
        total_negatives = max(total_negatives, 1)        
        filtered_ratio = filtered_negatives / total_negatives
        # print("total_negatives", total_negatives)
        # print("total_pairs", total_pairs)
        # print("total_pairs", total_pairs)
        # print("filtered_negatives", filtered_negatives)

        global_filtered_ratio = torch.tensor([filtered_ratio], device=logits_per_text.device)
        if is_dist_avail_and_initialized():
            dist.all_reduce(global_filtered_ratio, op=dist.ReduceOp.AVG)
        filtered_ratio = global_filtered_ratio.item()

        # ===== loss 计算 =====
        loglik = F.logsigmoid(m1_diag1 * logits_per_text)
        weight_mask = (~exclude_mask).float()
        if self.mean_loss:
            row_counts = torch.sum(weight_mask, dim=-1).clamp(min=1.0)
            nll = -torch.sum(loglik * weight_mask, dim=-1) / row_counts
        else:
            nll = -torch.sum(loglik * weight_mask, dim=-1)
        loss = nll.mean()

        # ===== 返回结果 =====
        return loss, {
            "total_pairs": total_pairs,
            "total_negatives": total_negatives,
            "filtered_negatives": filtered_negatives,
            "filtered_ratio": filtered_ratio,
        }


    def create_music_mask(self, query_ids, gallery_ids):
        """
        创建query与gallery之间的music id匹配mask矩阵
        
        参数:
            query_ids: 字符串类型的query music id列表，长度为m
            gallery_ids: 字符串类型的gallery music id列表，长度为n
            
        返回:
            mask: m×n的PyTorch张量，mask[i][j] = 1表示query_ids[i] == gallery_ids[j]，否则为0
        """
        # 收集所有唯一的music id并分配整数映射
        all_ids = list(set(query_ids + gallery_ids))
        id_mapping = {id_str: idx for idx, id_str in enumerate(all_ids)}
        
        # 将字符串id转换为整数id
        query_int = [id_mapping[id_str] for id_str in query_ids]
        gallery_int = [id_mapping[id_str] for id_str in gallery_ids]
        
        # 转换为张量并扩展维度以进行广播比较
        query_tensor = torch.tensor(query_int).unsqueeze(1)  # 形状: (m, 1)
        gallery_tensor = torch.tensor(gallery_int).unsqueeze(0)  # 形状: (1, n)
        
        # 生成mask矩阵
        mask = query_tensor == gallery_tensor  # 形状: (m, n)
        
        return mask

    def forward(
            self,
            samples,
            model_ema: torch.nn.Module = None,
            rank: int = 0,
            world_size: int = 1,
        ):
        if not self.historical_model_loaded and self.historical_model is not None:
            device = next(self.parameters()).device
            self.historical_model.to(device)
            self.historical_model_loaded = True
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

        video_music_match_score = samples['video_music_match_score']
        video_music_match_labels = []

        for i_b in range(len(video_music_match_score)):
            score = video_music_match_score[i_b].item()
            if score >= 7:
                video_music_match_labels.append(2)
            elif score >=5 and score <=6:
                video_music_match_labels.append(1)
            else:
                video_music_match_labels.append(0)

        video_music_match_labels = torch.LongTensor(video_music_match_labels).to(pixel_values.device)

        # music caption and attribute feature
        with torch.no_grad():
            music_attribute_input_ids = samples['music_attribute_input_ids']
            music_caption_input_ids = samples['music_caption_input_ids']
            music_attribute_embedding = self.encode_text(input_ids=music_attribute_input_ids)
            music_caption_embedding = self.encode_text(input_ids=music_caption_input_ids)

        music_embedding = self.music_tower(music_content_embedding, music_cover_embedding, music_attribute_embedding, music_caption_embedding)
        
        ret_dict = dict()
        cls_visual_embedding = self.encode_video(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes, frame_mask=frame_mask)
        if self.compress_embedding_size > 0:
            cls_visual_embedding = self.compress_vision_projector(cls_visual_embedding)

        # user_languages = samples["user_lang_code"]
        # user_countries = samples["user_country_code"] 
        # lang_indices = torch.tensor(
        #     user_languages,
        #     dtype=torch.long, device=pixel_values.device
        # )
        # country_indices = torch.tensor(
        #     user_countries,
        #     dtype=torch.long, device=pixel_values.device
        # )
        # cls_visual_embedding = self.user_context_adapter(cls_visual_embedding, lang_indices, country_indices)
        # # print("user_languages", user_languages, "user_countries", user_countries)
        # # print("lang_indices", lang_indices, "video_country_name", country_indices)

        # video music matching loss
        video_music_matching_loss = self.cal_video_music_matching_loss(cls_visual_embedding, music_embedding, video_music_match_labels)
        ret_dict['video_music_matching_loss'] = video_music_matching_loss

        # video music siglip loss
        if self.gpuwise_nce:
            cls_visual_embedding_all = custom_all_gather(cls_visual_embedding)
            music_embedding_all = custom_all_gather(music_embedding)
            all_music_item_ids = concat_all_gather_common_object(music_item_ids, device=cls_visual_embedding.device, concat=True)
            all_matching_scores = concat_all_gather_common_object(video_music_match_score.tolist(), device=cls_visual_embedding.device, concat=True)
            music_mask = self.create_music_mask(all_music_item_ids, all_music_item_ids)
        else:
            cls_visual_embedding_all = cls_visual_embedding
            music_embedding_all = music_embedding
            music_mask = self.create_music_mask(music_item_ids, music_item_ids)
            all_matching_scores = video_music_match_score.tolist()

        # 获取列表长度
        music_mask = music_mask.to(cls_visual_embedding.device)
        all_matching_scores = torch.LongTensor(all_matching_scores).to(cls_visual_embedding.device)

        # 在计算完当前模型的 embedding 之后，准备历史模型的 embedding（如果启用）
        hist_video_embedding_all = None
        hist_music_embedding_all = None

        if self.historical_model is not None:
            # 使用历史模型在本地先计算 embedding（历史模型已经被冻结）
            with torch.no_grad():
                hist_cls_visual_embedding_local = self.historical_model.extract_video_embeds(
                    pixel_values=pixel_values,
                    pixel_attention_mask=pixel_attention_mask,
                    spatial_shapes=spatial_shapes,
                    frame_mask=frame_mask
                )
                hist_music_embedding_local = self.historical_model.extract_music_embeds(
                    music_content_embedding=music_content_embedding,
                    music_cover_embedding=music_cover_embedding,
                    music_attribute_input_ids=samples['music_attribute_input_ids'],
                    music_caption_input_ids=samples['music_caption_input_ids']
                )

            # 若使用 gpuwise_nce，则需要 all_gather 历史 embedding 与当前模型一致
            if self.gpuwise_nce:
                hist_cls_visual_embedding_all = custom_all_gather(hist_cls_visual_embedding_local)
                hist_music_embedding_all = custom_all_gather(hist_music_embedding_local)
            else:
                hist_cls_visual_embedding_all = hist_cls_visual_embedding_local
                hist_music_embedding_all = hist_music_embedding_local

            # 将历史 embedding 送回给 calc_siglip_loss（会用于排除判断）
            hist_video_embedding_all = hist_cls_visual_embedding_all
            hist_music_embedding_all = hist_music_embedding_all

        # 计算 video-music siglip loss：将历史 embedding 传入用于可能的排除判断
        video_music_siglip_loss, neg_stat = self.calc_siglip_loss(
            image_embeds=cls_visual_embedding_all,
            music_embeds=music_embedding_all,
            music_mask=music_mask,
            matching_scores=all_matching_scores,
            rank=rank,
            world_size=world_size,
            hist_video_embeds=hist_video_embedding_all,
            hist_music_embeds=hist_music_embedding_all
        )
        ret_dict['video_music_siglip_loss'] = video_music_siglip_loss

        # ✅ 增加打点监控信息
        ret_dict.update({
            "filtered_negatives": neg_stat["filtered_negatives"],
            "filtered_ratio": neg_stat["filtered_ratio"],
        })
        
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
    def extract_music_embeds(self, music_content_embedding, music_cover_embedding, music_attribute_input_ids, music_caption_input_ids):
        music_attribute_embedding = self.encode_text(input_ids=music_attribute_input_ids)
        music_caption_embedding = self.encode_text(input_ids=music_caption_input_ids)
        music_embedding = self.music_tower(music_content_embedding, music_cover_embedding, music_attribute_embedding, music_caption_embedding)
        music_embedding = F.normalize(music_embedding, dim=-1)
        return music_embedding

    @torch.no_grad()
    def extract_video_embeds(self, 
            pixel_values: torch.FloatTensor, # B, F, 256, 768
            pixel_attention_mask: torch.Tensor, # B, F, 256
            spatial_shapes: torch.LongTensor, # B, F, 2
            frame_mask: torch.Tensor = None, # B, 256):
            # user_lang_code: List[int] = [],
            # user_country_code: List[int] = [],
        ):
        cls_visual_embedding = self.encode_video(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes, frame_mask=frame_mask)
        if self.compress_embedding_size > 0:
            cls_visual_embedding = self.compress_vision_projector(cls_visual_embedding)

        # user_languages = user_lang_code
        # user_countries = user_country_code
        # lang_indices = torch.tensor(
        #     user_languages,
        #     dtype=torch.long, device=pixel_values.device
        # )
        # country_indices = torch.tensor(
        #     user_countries,
        #     dtype=torch.long, device=pixel_values.device
        # )
        # cls_visual_embedding = self.user_context_adapter(cls_visual_embedding, lang_indices, country_indices)

        cls_visual_embedding = F.normalize(cls_visual_embedding, dim=-1)

        return cls_visual_embedding


if __name__ == "__main__":
    import sys
    from transformers import AutoProcessor, AutoTokenizer
    from transformers import pipeline
    from PIL import Image
    import requests
    
    model_path = "/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"
    videosig = VideoSigLIP2ForMusicNegativeFiltering(model_path=model_path, gpuwise_nce = True)
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