
import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.init import _calculate_fan_in_and_fan_out

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import ModelOutput, auto_docstring, can_return_tuple
from transformers.models.siglip2.configuration_siglip2 import Siglip2Config, Siglip2TextConfig, Siglip2VisionConfig
from transformers.models.siglip2.modeling_siglip2 import Siglip2PreTrainedModel



@dataclass
@auto_docstring(
    custom_intro="""
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.
    """
)
class Siglip2VisionOutput(ModelOutput):
    r"""
    image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The image embeddings obtained by applying the projection layer to the pooler_output.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for text model's outputs that also contains a pooling of the last hidden states.
    """
)
class Siglip2TextOutput(ModelOutput):
    r"""
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The text embeddings obtained by applying the projection layer to the pooler_output.
    """

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
@auto_docstring
class Siglip2Output(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
        Contrastive loss for image-text similarity.
    logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
        The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
        similarity scores.
    logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
        The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
        similarity scores.
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The text embeddings obtained by applying the projection layer to the pooled output of [`Siglip2TextTower`].
    image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The image embeddings obtained by applying the projection layer to the pooled output of [`Siglip2VisionTower`].
    text_model_output (`BaseModelOutputWithPooling`):
        The output of the [`Siglip2TextTower`].
    vision_model_output (`BaseModelOutputWithPooling`):
        The output of the [`Siglip2VisionTower`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: Optional[torch.FloatTensor] = None
    logits_per_text: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:
        """
        Resize positional embeddings to image-specific size and pad to a fixed size.

        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            spatial_shapes (`torch.LongTensor`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
            max_length (`int`):
                Maximum length of the positional embeddings to pad resized positional embeddings to

        Returns:
            `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for bfloat16/float16 on CPU
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i]
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)

            # Cast to original dtype
            resized_embeddings = resized_embeddings.to(source_dtype)

            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def forward(self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (batch_size, max_num_patches, num_channels * patch_size * patch_size)
            spatial_shapes (`list[tuple[int, int]]`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
        """

        # Apply patch embeddings to already patchified pixel values
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # Get positional resized and padded positional embeddings
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes, max_length=pixel_values.shape[1]
        )

        # Add positional embeddings to patch embeddings
        embeddings = patch_embeds + resized_positional_embeddings
        return embeddings




def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Siglip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        


        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Siglip2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Union[Siglip2VisionConfig, Siglip2TextConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Siglip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Siglip2EncoderLayer`].

    Args:
        config: Siglip2Config
    """

    def __init__(self, config: Siglip2Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    @can_return_tuple
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

        return hidden_states


class Siglip2VisionTransformer(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        # self.head = Siglip2MultiheadAttentionPoolingHead(config)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        r"""
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width) of the input images.
        """
        #pixel_values: B, F, ...
        #attention_mask: B, F, 256
        #spatial_shapes: B, F, 2
        # if(pixel_values.ndim == 3): # image, for debug only
        #     pixel_values = pixel_values.unsqueeze(0)
        
        
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        batch_size = pixel_values.shape[0]
        num_frames = pixel_values.shape[1]

        pixel_values = pixel_values.view(batch_size * num_frames, *pixel_values.shape[2:]).contiguous()
        attention_mask = attention_mask.view(batch_size * num_frames, -1).contiguous()
        spatial_shapes = spatial_shapes.view(batch_size * num_frames, *spatial_shapes.shape[2:]).contiguous()

        hidden_states = self.embeddings(pixel_values, spatial_shapes) 
        
        #hidden_states.shape = B*F,256,768

        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        else:
            encoder_attention_mask = attention_mask
        
        #进来以后是 B*F,256,hidden_size

        #hidden_states.shape = B*F,256,768
        hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # last_hidden_states.shape = B,F,256,768
        # last_hidden_state = hidden_state.view(batch_size,num_frames,-1,hidden_state.shape[-1]).contiguous()
        
        hidden_state = self.post_layernorm(hidden_state)

        return hidden_state


class Siglip2TextEmbeddings(nn.Module):
    def __init__(self, config: Siglip2TextConfig, interpolate = 4, max_segments = 4):
        super().__init__()
        embed_dim = config.hidden_size
        self.max_position_embeddings=64 * interpolate - (interpolate-1) * 20
        self.switch_len = 64
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)
        self.position_embedding_n = nn.Embedding(self.max_position_embeddings, embed_dim)
        self.segment_embedding = nn.Embedding(max_segments, embed_dim)
        default_flax_embed_init(self.position_embedding_n.weight)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(self.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        segment_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        max_position_embedding = self.position_embedding_n.weight.shape[0]

        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
        
        if seq_length <= self.switch_len:
            #print("using position_embedding for short text")
            position_embeddings = self.position_embedding(position_ids)
        else:
            position_embeddings = self.position_embedding_n(position_ids)
        embeddings = inputs_embeds + position_embeddings

        if segment_ids is not None:
            segment_embeddings = self.segment_embedding(segment_ids)
            embeddings += segment_embeddings

        return embeddings


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)


def trunc_normal_tf_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> torch.Tensor:
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \text{mean} \\leq b`.

    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsequently scaled and shifted by the mean and std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def default_flax_embed_init(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="normal")


class Siglip2TextTransformer(nn.Module):
    def __init__(self, config: Siglip2TextConfig, interpolate: int = 4):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = Siglip2TextEmbeddings(config, interpolate)
        self.encoder = Siglip2Encoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.head = nn.Linear(embed_dim, config.projection_size)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # note: Siglip2's text model does not use a causal mask, unlike the original CLIP model.
        # expand attention_mask
        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # The model uses the last token's hidden state, which may be padding.
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.head(pooled_output)

        return pooled_output


@auto_docstring(
    custom_intro="""
    The text model from Siglip2 without any head or projection on top.
    """
)
class Siglip2TextTower(Siglip2PreTrainedModel):
    config: Siglip2TextConfig

    def __init__(self, config: Siglip2TextConfig, interpolate: int = 4):
        super().__init__(config)
        self.text_model = Siglip2TextTransformer(config, interpolate)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, Siglip2TextTower

        >>> model = Siglip2TextTower.from_pretrained("google/siglip2-base-patch16-224")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224")

        >>> # important: make sure to set padding="max_length" as that's how the model was trained
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: Siglip2VisionConfig,probe_size: int = 32, num_frames: int = 8, num_head_layers = 6):
        super().__init__()

        self.probe_n = nn.Parameter(torch.randn(1, probe_size, config.hidden_size))#probe for 32
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True, add_zero_attn=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_frames = num_frames

        self.absolute_frame_pos_embed = nn.Parameter(torch.zeros((1, num_frames * probe_size, config.hidden_size)))
        self.head_layers = nn.ModuleList([Siglip2EncoderLayer(config) for _ in range(num_head_layers)])

        self.probe_2 = nn.Parameter(torch.randn(1, 1, config.hidden_size))#probe for 1
        self.attention_2 = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp_2 = Siglip2MLP(config)

        self.init_weight()
    
    def init_weight(self):
        def _init_weights(module):
            if isinstance(module, Siglip2Attention):
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
            elif isinstance(module, Siglip2MultiheadAttentionPoolingHead):
                nn.init.xavier_uniform_(module.probe_2.data)
                nn.init.xavier_uniform_(module.attention_2.in_proj_weight.data)
                nn.init.zeros_(module.attention_2.in_proj_bias.data)
                lecun_normal_(module.absolute_frame_pos_embed.data)
            elif isinstance(module, nn.Embedding):
                default_flax_embed_init(module.weight)
            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        
        self.apply(_init_weights)


    def forward(self, hidden_state: torch.Tensor, frame_mask: Optional[torch.Tensor] = None, pixel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        #Attention Pooling: each frame extract 16 tokens
        batch_frame = hidden_state.shape[0]
        #hidden.shape B*F,32, hidden_size
        probe_1 = self.probe_n.repeat(batch_frame, 1, 1)

        if pixel_mask is not None:
            pixel_mask = pixel_mask.view(batch_frame, -1).contiguous()
            target_len, source_len = probe_1.shape[1], hidden_state.shape[1]
            pixel_mask = _prepare_4d_attention_mask(pixel_mask, hidden_state.dtype, target_len)#B*F, 1, 32, 256
            
            pixel_mask = pixel_mask.repeat(1, self.num_heads, 1, 1)#B*F,num_heads, 32, 256
            pixel_mask = pixel_mask.view(-1, target_len, source_len)

        
        hidden_state = self.attention(probe_1, hidden_state, hidden_state, attn_mask=pixel_mask)[0]


        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        # Last_hidden_state.shape B,F*32,hidden_size
        hidden_state = hidden_state.contiguous().view(-1, frame_mask.shape[1], hidden_state.shape[-1])

        # Add frame positional embedding B,F*32,hidden_size
        hidden_state = hidden_state + self.absolute_frame_pos_embed[:, :frame_mask.shape[1]]

        if frame_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            frame_attention_mask = _prepare_4d_attention_mask(frame_mask, hidden_state.dtype)
        else:
            frame_attention_mask = None

        # Transformer layers
        for layer in self.head_layers:

            layer_outputs = layer(
                hidden_state,
                attention_mask=frame_attention_mask
            )

            # B,F*16,hidden_size
            hidden_state = layer_outputs[0]

        # Final attention pooling: 
        # B,F*16,hidden_size -> B,1,hidden_size
        batch = hidden_state.shape[0]

        probe_2 = self.probe_2.repeat(batch, 1, 1)

        if frame_mask is not None:
            target_len, source_len = probe_2.shape[1], hidden_state.shape[1]
            frame_mask = _prepare_4d_attention_mask(frame_mask, hidden_state.dtype, target_len)
            frame_mask = frame_mask.repeat(1, self.num_heads, target_len, 1)
            frame_mask = frame_mask.view(-1, target_len, source_len)

        hidden_state = self.attention_2(probe_2, hidden_state, hidden_state, attn_mask=frame_mask)[0]

        residual = hidden_state
        hidden_state = self.layernorm_2(hidden_state)
        hidden_state = residual + self.mlp_2(hidden_state)

        return hidden_state[:, 0]

@auto_docstring(
    custom_intro="""
    The vision model from Siglip2 without any head or projection on top.
    """
)
class Siglip2VisionTower(Siglip2PreTrainedModel):
    config: Siglip2VisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__(config)

        self.vision_model = Siglip2VisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width) of the input images.

        Examples:
        """
        #prepare pixel_mask for cls_token

        return self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )







__all__ = [
    "Siglip2PreTrainedModel",
    "Siglip2TextTower",
    "Siglip2VisionTower",
    "Siglip2MultiheadAttentionPoolingHead",
]