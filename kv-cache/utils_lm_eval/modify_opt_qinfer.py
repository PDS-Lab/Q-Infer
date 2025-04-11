import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.opt.modeling_opt import OPTAttention
import types


__all__ = ['convert_kvcache_opt_heavy_recent', 'OPTAttention_Mask']

use_gumbel = False
use_tau = True
itr_count = 0
quantized_ratio = 0.5

def set_itr_count(count):
    global itr_count
    itr_count = count

def partial_quantize_to_qint8(
    k,
    token_quantized_mask
):
    # 提取需要量化的部分
    k_to_quantize = k * token_quantized_mask

    # 确保 k_to_quantize 不为空
    if k_to_quantize.numel() == 0:
        return k

    # 将 BF16 转换为 FP32，以便进行量化
    k_to_quantize_fp32 = k_to_quantize.to(torch.float32)
    # 计算量化参数（scale 和 zero_point）
    quantized_min = -128
    quantized_max = 127
    scale = (k_to_quantize_fp32.max() - k_to_quantize_fp32.min()) / (quantized_max - quantized_min)
    # 确保 scale 不为零
    if scale == 0:
        scale = 1e-9
    zero_point = torch.round(quantized_min + k_to_quantize_fp32.min() / scale)
    # 确保 zero_point 在合法范围内
    zero_point = torch.clamp(zero_point, quantized_min, quantized_max).to(torch.int8)
    # 量化
    k_quantized = torch.quantize_per_tensor(k_to_quantize_fp32, scale, zero_point, torch.qint8)
    # 将量化后的部分替换回原始 k 张量
    k_quantized = k_quantized.dequantize().to(torch.bfloat16)
    k = k * torch.logical_not(token_quantized_mask) + k_quantized * token_quantized_mask
    return k.to(torch.float16)


def local_heavy_hitter_mask(attn_weights, heavy_budget, no_padding_seq_length=None, tau_init=1.0, tau_delta=0.02, quantized_tokens = 0):

    global itr_count
    # attn_weights (head, query, keys)
    dtype_attn_weights = attn_weights.dtype
    seq_length = attn_weights.shape[-1]
    if no_padding_seq_length is None:
        padding_length = 0
    else:
        padding_length = seq_length - no_padding_seq_length

    offset = torch.finfo(attn_weights.dtype).min

    # whp
    if use_gumbel:
        if use_tau:
            tmp_attn = nn.functional.gumbel_softmax(attn_weights, tau=tau_init + (itr_count * tau_delta), hard=False, dim=-1).to(dtype_attn_weights)
        else:
            # TypeError: gumbel_softmax() got an unexpected keyword argument 'dtype'
            tmp_attn = nn.functional.gumbel_softmax(attn_weights, hard=False, dim=-1).to(dtype_attn_weights)
    else:
        tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype_attn_weights)

    accumulated_attention_score = torch.sum(tmp_attn[:,padding_length:heavy_budget+padding_length,:], dim=-2) #(head, keys)
    # 将超出预算部分的累积注意力分数置为0
    accumulated_attention_score[:,heavy_budget+padding_length:] = 0

    if padding_length > 0:
        accumulated_attention_score[:,:padding_length] = 0

    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom[:, padding_length:heavy_budget+padding_length, padding_length:heavy_budget+padding_length] = True

    mask_quantized = torch.zeros_like(attn_weights, dtype=torch.bool)

    # 生成掩码，对于每个查询，计算其对所有键的注意力权重，并选择累积注意力分数最高的 heavy_budget-1 个键
    for token_index in range(heavy_budget+padding_length, seq_length):
        # whp
        if use_gumbel:
            if use_tau:
                tmp_attn_index = nn.functional.gumbel_softmax(attn_weights[:,token_index,:], tau=tau_init + (itr_count * tau_delta), hard=False, dim=-1).to(dtype_attn_weights)
            else:
                tmp_attn_index = nn.functional.gumbel_softmax(attn_weights[:,token_index,:], hard=False, dim=-1).to(dtype_attn_weights)
        else:
            tmp_attn_index = nn.functional.softmax(attn_weights[:,token_index,:], dim=-1, dtype=torch.float32).to(dtype_attn_weights)
        
        _, tmp_topk_index = accumulated_attention_score.topk(k=heavy_budget-1, dim=-1)
        # 注意 -0 = 0
        if quantized_tokens > 0:
            quantized_topk_index = tmp_topk_index[:,-quantized_tokens:]
        else:
            quantized_topk_index = torch.empty((tmp_topk_index.shape[0], 0)).cuda()
        zeros_index = torch.zeros_like(tmp_attn_index, dtype=torch.bool)
        mask_bottom_index = zeros_index.scatter(-1, tmp_topk_index, True) #(head, keys)
        mask_bottom_index[:, token_index] = True
        mask_quantized_index = zeros_index.scatter(-1, quantized_topk_index, True) #(head, keys)

        mask_bottom[:,token_index,:] = mask_bottom_index
        mask_quantized[:,token_index,:] = mask_quantized_index

        accumulated_attention_score += tmp_attn_index
        # 更新累积注意力分数，并将其乘以当前的掩码
        accumulated_attention_score = accumulated_attention_score * mask_bottom_index

    # 将掩码转换为下三角矩阵，确保只保留对角线及其以下的部分
    mask_bottom = torch.tril(mask_bottom, diagonal=0)
    mask_quantized = torch.tril(mask_quantized, diagonal=0)

    itr_count = itr_count + 1

    return mask_bottom, mask_quantized


def sanity_check(mask):
    # mask (head, query, key)
    ones = torch.ones_like(mask)
    ones = torch.triu(ones, diagonal=0)
    mask_bottom = torch.logical_or(mask, ones)

    error_cnt = 0
    for i in range(mask_bottom.shape[1]-1):
        index = mask_bottom[:,i,:].eq(0).unsqueeze(1)
        index[:,i:]=0
        error_cnt += (mask_bottom[:,i:,:] * index).sum().item()
    print(error_cnt)
    return error_cnt


class OPTAttention_Mask(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        heavy_ratio: float,
        recent_ratio: float,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.attn = None

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.heavy_budget_ratio = heavy_ratio
        self.recent_budget_ratio = recent_ratio

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)   # (bsz * self.num_heads, sequence_length, self.head_dim)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        #current_attn_weights = attn_weights.clone()

        ### Heavy + Recent
        heavy_budget = int(self.heavy_budget_ratio * attn_weights.shape[-1])
        global quantized_ratio
        quantized_tokens = int(heavy_budget * quantized_ratio)
        heavy_budget = heavy_budget + quantized_tokens
        quantized_tokens = quantized_tokens * 2
        recent_budget = int(self.recent_budget_ratio * attn_weights.shape[-1])

        # Heavy Hitter Mask
        if heavy_budget > 0:
            mask_bottom, quantized_mask = local_heavy_hitter_mask(attn_weights, heavy_budget, None, quantized_tokens=quantized_tokens) # Default: No padding applied to input
        else:
            mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)

        # Recent Mask
        ones = torch.ones_like(attn_weights, dtype=torch.bool)
        ones = torch.triu(ones, diagonal=-recent_budget)
        mask_bottom = torch.logical_or(mask_bottom, ones)

        # Combine h2o+recent and apply casual mask
        mask_bottom = torch.tril(mask_bottom, diagonal=0)

        ##### 重计算 K, attn_weights
        nk = key_states
        #nk = partial_quantize_to_qint8(nk, quantized_mask[:,:,0].unsqueeze(-1).expand(-1, -1, nk.shape[-1]))
        # print(query_states.dtype, nk.dtype)
        attn_weights = torch.bmm(query_states, nk.transpose(1, 2).to(query_states.dtype))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        #####

        # TODO: which is better? attn_weights[~mask_bottom] = torch.finfo(attn_weights.dtype).min
        attn_weights[~mask_bottom] = torch.min(attention_mask)

        # print(key_states.shape, value_states.shape, attn_weights.shape, mask_bottom.shape)
        # torch.Size([32, 94, 128]) torch.Size([32, 94, 128]) torch.Size([32, 94, 94]) torch.Size([32, 94, 94])

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        self.attn = attn_weights.clone()
        
        #current_attn_weights = nn.functional.softmax(current_attn_weights, dim = -1, dtype=torch.float32).to(torch.float16)
        #cos_sim = torch.sum(current_attn_weights * attn_weights, dim = -1) / (torch.norm(current_attn_weights, dim = -1) * torch.norm(attn_weights, dim = -1))

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # 量化 V
        nv = value_states
        #nv = partial_quantize_to_qint8(nv, quantized_mask[:,:,0].unsqueeze(-1).expand(-1, -1, nv.shape[-1]))

        # attn_output = torch.bmm(attn_probs, value_states)
        attn_output = torch.bmm(attn_probs, nv)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

# whp
def layer_forward_keyformer(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

    residual = hidden_states

    # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
    if self.do_layer_norm_before:
        hidden_states = self.self_attn_layer_norm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        past_key_value=past_key_value,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
    )
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states

    # 350m applies layer norm AFTER attention
    if not self.do_layer_norm_before:
        hidden_states = self.self_attn_layer_norm(hidden_states)

    # Fully Connected
    hidden_states_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual = hidden_states

    # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
    if self.do_layer_norm_before:
        hidden_states = self.final_layer_norm(hidden_states)

    hidden_states = self.fc1(hidden_states)
    hidden_states = self.activation_fn(hidden_states)
    hidden_states = self.fc2(hidden_states)

    hidden_states = (residual + hidden_states).view(hidden_states_shape)

    # 350m applies layer norm AFTER attention
    if not self.do_layer_norm_before:
        hidden_states = self.final_layer_norm(hidden_states)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs



def convert_kvcache_opt_heavy_recent(model, config):

    for idx, module in enumerate(model.model.decoder.layers):
        if (idx != 0) and (idx != 1):
            #if use_gumbel:
            #    model.model.decoder.layers[idx].forward = types.MethodType(layer_forward_keyformer, model.model.decoder.layers[idx])
            model.model.decoder.layers[idx].self_attn = OPTAttention_Mask(
                embed_dim=module.embed_dim,
                num_heads=config.num_attention_heads,
                heavy_ratio = config.heavy_ratio,
                recent_ratio = config.recent_ratio,
                dropout=config.attention_dropout,
                is_decoder=True,
                bias=config.enable_bias,
            )
        else:
            print("skip layer: ", idx)
    return model

