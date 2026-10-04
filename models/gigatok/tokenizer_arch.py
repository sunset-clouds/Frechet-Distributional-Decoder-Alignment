"""GigaTok VQ_SS256 tokenizer, flattened from the upstream release.

Concatenated in dependency order; intra-package imports removed, bodies otherwise
unchanged.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, pack, unpack
from typing import Optional, List
import os
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
from collections import OrderedDict
from typing import List
from torch import einsum
from einops import rearrange, reduce, pack, unpack
from dataclasses import dataclass
from dataclasses import dataclass, field


# ------------------------------------------------------------------------------
# gigatok_utils/drop_path.py
# ------------------------------------------------------------------------------
# from timm.models.layers import DropPath

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


# ------------------------------------------------------------------------------
# gigatok_utils/rope.py
# ------------------------------------------------------------------------------
#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=1):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=0):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)    # (grid_size**2, head_dim // 2, 2)
    if cls_token_num > 0:
        cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
        return cond_cache 
    else:
        return cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, bs_first=True):
    # if bs_first
        # x: (bs, seq_len, n_head, head_dim)
        # freqs_cis (seq_len, head_dim // 2, 2)
    # else
        # x: (seq_len, bs, n_head, head_dim) 
        # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    if bs_first:
        freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    else:
        freqs_cis = freqs_cis.view(xshaped.size(1), 1, 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)

    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


# ------------------------------------------------------------------------------
# tokenizer/tokenizer_image/vq/blocks.py
# ------------------------------------------------------------------------------
"""Building blocks for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Reference: 
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
"""



# check the version of pytorch, 
# if pytorch version >= 2.2.0, then flash_attention can be used 
if torch.__version__ >= "2.2.0":
    HAS_FLASH_ATTENTION_V2 = True
    # print("flash_attention v2 can be used.")
else:
    HAS_FLASH_ATTENTION_V2 = False
    # print("flash_attention v2 is not supported.")


############################################################################
## Util Functions
############################################################################


def sample_multi_level_1d_tokens(x, n_level, first_grow="row"):
    """
    Args:
        x: [B, C, H, W]
        n_level: token_length = 2^n_level = 1 + 2^0 + ... + 2^(n_level - 1)
            corresponding to token_group_idx: 0, 1, ..., n_level (summed to be n_level + 1 groups)
    Returns:
        x_1d: [B, 2^n_level, C]
    note: log_2(W) >= n_level // 2 if first_grow == 'row'
    on the condition of: 
        if first_grow == 'row':
            W % 2^(n_level//2) == 0 and H % 2^((n_level-1)//2) == 0
        elif first_grow == 'col':
            H % 2^(n_level//2) == 0 and W % 2^((n_level-1)//2) == 0
    """

    def _get_kernel_size(token_group_idx, max_h, max_w, first_grow="row"):
        # level_idx should start from 0
        level_idx = max(token_group_idx - 1, 0)
        if first_grow == "row":
            # first split in the row dimension (first increase w_block_num)
            w_block_num = 2 ** ((level_idx + 1) // 2)
            h_block_num = 2 ** (level_idx // 2)
            w_block_size = max_w // w_block_num
            h_block_size = max_h // h_block_num
        elif first_grow == "col":
            # first split in the col dimension (first increase h_block_num)
            h_block_num = 2 ** ((level_idx + 1) // 2)
            w_block_num = 2 ** (level_idx // 2)
            h_block_size = max_h // h_block_num
            w_block_size = max_w // w_block_num
        else:
            raise ValueError(f"Invalid first_grow: {first_grow}, choose from 'row' or 'col'")
        assert w_block_size > 0 and h_block_size > 0, f"Invalid level idx: {level_idx}, max_h: {max_h}, max_w: {max_w}"
        return (h_block_size, w_block_size)

    x_groups = []
    for i in range(n_level + 1):
        kernel_size = _get_kernel_size(i, x.shape[2], x.shape[3], first_grow=first_grow)
        cur_group_x = F.avg_pool2d(x, kernel_size=kernel_size)
        cur_group_x = rearrange(cur_group_x, "b c h w -> b (h w) c")
        x_groups.append(cur_group_x)

    x_1d = torch.cat(x_groups, dim=1)
    return x_1d


def last_level_1d_features_to_2d_maps(x_1d, n_level, max_h, max_w, first_grow="row"):
    """
    Args:
        x_1d: [B, 2^n_level, C]
        n_level: token_length = 2^n_level = 1 + 2^0 + ... + 2^(n_level - 1)
            corresponding to token_group_idx: 0, 1, ..., n_level (summed to be n_level + 1 groups)
    Returns:
        x_dilated: [B, C, max_h, max_w]
    """
    assert x_1d.shape[1] == 2 ** n_level, f"Invalid x_1d shape: {x_1d.shape}"
    if n_level == 1:
        last_level_features = x_1d[:, 0:2, :].mean(dim=1, keepdim=True)
    else:
        last_level_features = x_1d[:, -2**(n_level - 1):, :]
    if first_grow == "row":
        # first split in the row dimension (first increase w_block_num)
        w_block_num = 2 ** (n_level // 2)
        h_block_num = 2 ** ((n_level - 1)// 2)
        w_block_size = max_w // w_block_num
        h_block_size = max_h // h_block_num
    elif first_grow == "col":
        # first split in the col dimension (first increase h_block_num)
        h_block_num = 2 ** (n_level // 2)
        w_block_num = 2 ** ((n_level - 1)// 2)
        h_block_size = max_h // h_block_num
        w_block_size = max_w // w_block_num
    
    last_level_features = rearrange(last_level_features, "b (h w) c -> b c h w", h=h_block_num, w=w_block_num)
    # dilate the last level features by w_block_size and h_block_size in a differentiable way
    x_dilated = F.interpolate(last_level_features, size=(max_h, max_w), mode='nearest')

    return x_dilated


def multi_level_1d_features_to_2d_maps_avg(x_1d, n_level, max_h, max_w, first_grow="row"):
    """
    Args:
        x_1d: [B, 2^n_level, C]
        n_level: token_length = 2^n_level = 1 + 2^0 + ... + 2^(n_level - 1)
            corresponding to token_group_idx: 0, 1, ..., n_level (summed to be n_level + 1 groups)
    Returns:
        x_dilated: [B, C, max_h, max_w]
    """
    assert x_1d.shape[1] == 2 ** n_level, f"Invalid x_1d shape: {x_1d.shape}"
    # print("this multi_level_1d_features_to_2d_maps_avg function is called ")

    for level_idx in range(n_level):
        start_idx = 0
        end_idx = 0
        if level_idx == 0:
            start_idx = 0
            end_idx = 2
            cur_level_features = x_1d[:, start_idx:end_idx, :].mean(dim=1, keepdim=True)
        else:
            start_idx = end_idx
            end_idx = start_idx + 2**level_idx
            # 1 + 2^(level_idx - 1) : 2 + 2^(level)
            cur_level_features = x_1d[:, start_idx:end_idx, :]

        if first_grow == "row":
            w_block_num = 2 ** ((level_idx + 1) // 2)
            h_block_num = 2 ** (level_idx // 2)

        if first_grow == "col":
            h_block_num = 2 ** ((level_idx + 1) // 2)
            w_block_num = 2 ** (level_idx // 2)


        cur_level_features = rearrange(cur_level_features, "b (h w) c -> b c h w", h=h_block_num, w=w_block_num)
        x_dilated = F.interpolate(cur_level_features, size=(max_h, max_w), mode='nearest')
        if level_idx == 0:
            x_result = x_dilated
        else:
            x_result += x_dilated
    
    x_result = x_result / (n_level)

    return x_result



def get_attn_mask(x_1d, causal_type="per-token"):
    """
    Generates an attention mask based on the given causal type.
    
    Args:
        x_1d: Tensor of shape [2^n_level, B, C], where 2^n_level represents the number of tokens.
        causal_type: A string indicating the type of causal mask to generate. 
                     Options are "per-token" or "per-level".
    
    Returns:
        A binary attention mask of shape [2^n_level, 2^n_level] with dtype torch.bool.
    """
    
    seq_len = x_1d.shape[0]  # This is 2^n_level
    n_level = int(np.round(np.log2(seq_len)))  # Calculate n_level from the sequence length

    if causal_type == "per-token":
        # Generate a token-wise (lower triangular) causal mask
        attn_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    
    elif causal_type == "per-level":
        # Generate a level-wise causal mask
        attn_mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
        
        # Define the starting index for each level
        for level_idx in range(n_level + 1):
            level_start =  0 if level_idx == 0 else 2 ** level_idx
            level_end = 2 ** (level_idx + 1)
            
            # Allow attention within the same level and all previous levels
            attn_mask[level_start:level_end, :level_end] = False
    
    else:
        raise ValueError(f"Unknown causal_type: {causal_type}. Choose 'per-token' or 'per-level'.")
    
    return attn_mask.to(x_1d.device)


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """ Depth-to-Space DCR mode (depth-column-row) core implementation.

        Args:
            x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
            block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions"
        )
    c, h, w = x.shape[-3:]

    s = block_size**2
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )

    outer_dims = x.shape[:-3]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, block_size, block_size, c // s, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 3, 4, 1, 5, 2)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, h * block_size,
                            w * block_size)

    return x



############################################################################
## Layer Component Module
############################################################################

def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class D2SUpsampler(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        dim_out = dim * 4
        self.conv1 = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        """
        input_image: [B C H W]
        """
        out = self.conv1(x)
        out = self.depth2space(out, block_size=2)
        return out
 


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, z_channel, in_filters, num_groups=32, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_filters, eps=eps, affine=False)
        # self.lin = nn.Linear(z_channels, in_filters * 2)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps
    
    def forward(self, x, quantizer):
        B, C, _, _ = x.shape
        # quantizer = F.adaptive_avg_pool2d(quantizer, (1, 1))
        ### calcuate var for scale
        scale = rearrange(quantizer, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps #not unbias
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1)

        ### calculate mean for bias
        bias = rearrange(quantizer, "b c h w -> b c (h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(B, C, 1, 1)
       
        x = self.gn(x)
        x = scale * x + bias

        return x


class AttentionCustom(nn.Module):
    def __init__(
            self, 
            dim,
            n_head,
            resid_dropout_p,
            use_rope=False,
            use_qk_norm=False,
            use_flash_attn=False,
            no_bias=False,
            attn_dropout_p=0
        ):
        """
        This custom attention block supports the following modifications:
        - ROPE
        - QK Norm
        - Flash attention
        Currently, the dimension of the key and value is the same as the query.
        """
        super().__init__()

        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn
        if self.use_flash_attn:
            print("Using flash attention!")
        # flash attention can be switched to normal attention for inference
        # rasie error only when training and use_flash_attn is True and 
        # flash attention is not installed
        assert HAS_FLASH_ATTENTION_V2 or (not self.use_flash_attn) or (not self.training), \
            "Flash attention is not installed and cannot be used when training"
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        total_qkv_dim =  3 * self.n_head * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.q_proj = nn.Linear(dim, dim, bias=not no_bias)
        self.k_proj = nn.Linear(dim, dim, bias=not no_bias)
        self.v_proj = nn.Linear(dim, dim, bias=not no_bias)
        self.wo = nn.Linear(dim, dim, bias=not no_bias)

        # regularization
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor,
        value: torch.Tensor,
        freqs_cis: torch.Tensor = None, 
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ):
        """
        The q, k, v will be projected into multiple heads.
        """
        seqlen, bsz, _ = query.shape

        # rearrange, (L, B, D) -> (B, L, D)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        xq = self.q_proj(query)
        xk = self.k_proj(key)
        xv = self.v_proj(value)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_head, self.head_dim)
    
        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        if self.use_rope:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)
        else:
            assert freqs_cis is None, "Attention Module is not using ROPE but freqs_cis is not None. Check your setting!"
        

        # (B, L, H, D) -> (B, H, L, D)
        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.use_flash_attn:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
                output = F.scaled_dot_product_attention(
                    xq, xk, xv, 
                    attn_mask=attn_mask, 
                    is_causal=is_causal, # is_causal=False is for KV cache
                    dropout_p=self.attn_dropout_p if self.training else 0)
        else:
            output = F.scaled_dot_product_attention(
                xq, xk, xv, 
                attn_mask=attn_mask, 
                is_causal=is_causal, # is_causal=False is for KV cache
                dropout_p=self.attn_dropout_p if self.training else 0)
            
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        output = self.resid_dropout(self.wo(output))

        # rearrange, (B, L, D) -> (L, B, D)
        return output.transpose(0, 1)


class SelfAttentionCustom(nn.Module):
    def __init__(
            self, 
            dim,
            n_head,
            resid_dropout_p,
            use_rope=False,
            use_qk_norm=False,
            no_bias=False,
            use_flash_attn=False,
            attn_dropout_p=0
        ):
        """
        This custom attention block supports the following modifications:
        - ROPE
        - QK Norm
        - Flash attention
        """
        super().__init__()

        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn
        # flash attention can be switched to normal attention for inference
        # rasie error only when training and use_flash_attn is True and 
        # flash attention is not installed
        assert HAS_FLASH_ATTENTION_V2 or (not use_flash_attn) or (not self.training), \
            "Flash attention is not installed and cannot be used when training"
        assert dim % n_head == 0

        if self.use_flash_attn:
            print("Using flash attention!")

        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        total_qkv_dim =  3 * self.n_head * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(dim, total_qkv_dim, bias=not no_bias)
        self.wo = nn.Linear(dim, dim, bias=not no_bias)

        # regularization
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor = None, 
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ):
        seqlen, bsz, _ = x.shape

        # rearrange, (L, B, D) -> (B, L, D)
        x = x.transpose(0, 1)

        xq, xk, xv = self.wqkv(x).split([self.dim, self.dim, self.dim], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_head, self.head_dim)
    
        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        if self.use_rope:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)
        else:
            assert freqs_cis is None, "Attention Module is not using ROPE but freqs_cis is not None. Check your setting!"
        
        # (B, L, H, D) -> (B, H, L, D)
        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.use_flash_attn:
            # Shape: (batch_size, num_heads, seq_length, head_dim)
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
                output = F.scaled_dot_product_attention(
                    xq, xk, xv, 
                    attn_mask=mask, 
                    is_causal=is_causal, # is_causal=False is for KV cache
                    dropout_p=self.attn_dropout_p if self.training else 0)            
        else:
            output = F.scaled_dot_product_attention(
                xq, xk, xv, 
                attn_mask=mask, 
                is_causal=is_causal, # is_causal=False is for KV cache
                dropout_p=self.attn_dropout_p if self.training else 0)            
            
        # (B, H, L, D) -> (B, L, H*D)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        output = self.resid_dropout(self.wo(output))

        # rearrange back, (B, L, D) -> (L, B, D)
        return output.transpose(0, 1)





############################################################################
## Layer Module
############################################################################

class TransformerDecoderLayer(nn.Module):
    """
    This is the Q-former layer from DETR.
    """
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.1,
                 activation=nn.GELU, normalize_before=True, query_rope=False,
                 use_qk_norm=False, use_flash_attn=False
                 ):
        super().__init__()
        self.query_rope = query_rope
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn
        if self.query_rope:
            self.self_attn = SelfAttentionCustom(
                                d_model, 
                                nhead, 
                                resid_dropout_p=dropout, 
                                use_rope=True,
                            )
            self.multihead_attn = AttentionCustom(
                                d_model,
                                nhead,
                                resid_dropout_p=dropout,
                                use_rope=True,
                            )
        elif self.use_qk_norm or self.use_flash_attn:
            self.self_attn = AttentionCustom(
                                d_model, 
                                nhead, 
                                resid_dropout_p=dropout, 
                                use_qk_norm=self.use_qk_norm,
                                use_flash_attn=self.use_flash_attn,
                            )
            self.multihead_attn = AttentionCustom(
                                d_model,
                                nhead,
                                resid_dropout_p=dropout,
                                use_qk_norm=self.use_qk_norm,
                                use_flash_attn=self.use_flash_attn,
                            )
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        dim_feedforward = int(d_model * mlp_ratio)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before



    def with_pos_embed(self, tensor, pos: Optional[Tensor], rope=False):
        if rope:
            return apply_rotary_emb(tensor, pos, bs_first=False)
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        
        if self.query_rope:
            tgt2 = self.self_attn(
                tgt2,
                freqs_cis=query_pos,
                mask=tgt_mask,
            )
        elif self.use_qk_norm or self.use_flash_attn:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        else:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.use_qk_norm or self.use_flash_attn:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos, rope=self.query_rope),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask)
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos, rope=self.query_rope),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        if self.query_rope:
            tgt2 = self.self_attn(
                tgt2,
                query_pos,
                tgt_mask
            )
        elif self.use_qk_norm or self.use_flash_attn:
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask)
        else:
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        if torch.isnan(tgt2).any():
            print("tgt2 is nan")
            print("q shape and values:", q.shape, q.min().item(), q.max().item())
            print("k shape and values:", k.shape, k.min().item(), k.max().item())
            print("tgt shape and values:", tgt.shape, tgt.min().item(), tgt.max().item())
            tmp = self.norm1(tgt)
            print("after norm tgt shape and values:", tmp.shape, tmp.min().item(), tmp.max().item())

            print("q is nan:", torch.isnan(q).any())
            print("k is nan:", torch.isnan(k).any())
            print("tgt is nan:", torch.isnan(tgt).any())

            # check any nans in the weights
            for name, param in self.self_attn.named_parameters():
                if torch.isnan(param).any():
                    print(f"{name} has nan values")
            torch.set_printoptions(threshold=10_000)
            print(q)
            print(tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        if self.use_qk_norm or self.use_flash_attn:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos, rope=self.query_rope),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask)
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos, rope=self.query_rope),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
 
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerLayer(nn.Module):
    """
    This is the standard Transformer layer. Currently only supports absolute positional embedding.
    # TODO: support RoPE 2d
    """
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.1,
                 activation=nn.GELU, normalize_before=True, use_rope=False,
                 use_qk_norm=False,
                 ):
        super().__init__()
        self.use_rope = use_rope
        assert not self.use_rope, "QformerLayerVar does not support RoPE yet"
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.self_attn = AttentionCustom(
                                d_model, 
                                nhead, 
                                resid_dropout_p=dropout, 
                                use_qk_norm=self.use_qk_norm,
                                use_flash_attn=self.use_flash_attn,
                            )
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        dim_feedforward = int(d_model * mlp_ratio)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before



    def with_pos_embed(self, tensor, pos: Optional[Tensor], rope=False):
        if rope:
            return apply_rotary_emb(tensor, pos, bs_first=False)
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
    ):
        if self.use_qk_norm:
            q = k = tgt
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        else:
            q = k = tgt
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        if self.use_qk_norm:
            tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask)
        else:
            tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        assert pos is None, "TransformerLayer does not support injected positional embedding"
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, pos)

class QformerLayerVar(nn.Module):
    """
    This is the variant of Q-former layer from DETR. It is merely used for comparison.
    This layer has 2 multi-head attention layers, and the positional embedding follows the original DETR.
    But there is not reference feature map any more. All attentions are self-attention.
    """
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.1,
                 activation=nn.GELU, normalize_before=True, use_rope=False,
                 use_qk_norm=False,
                 ):
        super().__init__()
        self.use_rope = use_rope
        assert not self.use_rope, "QformerLayerVar does not support RoPE yet"
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.self_attn = AttentionCustom(
                                d_model, 
                                nhead, 
                                resid_dropout_p=dropout, 
                                use_qk_norm=self.use_qk_norm,
                                use_flash_attn=self.use_flash_attn,
                            )
            self.multihead_attn = AttentionCustom(
                                d_model,
                                nhead,
                                resid_dropout_p=dropout,
                                use_qk_norm=self.use_qk_norm,
                                use_flash_attn=self.use_flash_attn,
                            )
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        dim_feedforward = int(d_model * mlp_ratio)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before



    def with_pos_embed(self, tensor, pos: Optional[Tensor], rope=False):
        if rope:
            return apply_rotary_emb(tensor, pos, bs_first=False)
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
    ):
        
        if self.use_qk_norm:
            q = k = self.with_pos_embed(tgt, pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        else:
            q = k = self.with_pos_embed(tgt, pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.use_qk_norm :
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos, rope=self.use_rope),
                                    key=self.with_pos_embed(tgt, pos),
                                    value=tgt, attn_mask=tgt_mask)
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos, rope=self.use_rope),
                                    key=self.with_pos_embed(tgt, pos),
                                    value=tgt, attn_mask=tgt_mask,
                                    key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        if self.use_qk_norm:
            q = k = self.with_pos_embed(tgt, pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        else:
            q = k = self.with_pos_embed(tgt2, pos)
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        if self.use_qk_norm:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, pos, rope=self.use_rope),
                                    key=self.with_pos_embed(tgt2, pos),
                                    value=tgt2, attn_mask=tgt_mask)
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, pos, rope=self.use_rope),
                                    key=self.with_pos_embed(tgt2, pos),
                                    value=tgt2, attn_mask=tgt_mask,
                                    key_padding_mask=tgt_key_padding_mask)[0]
 
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, pos)




class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x

############################################################################
## Part Module
############################################################################

class ViTEncoder(nn.Module):
    def __init__(
            self, 
            image_size=256, 
            patch_size=16, 
            model_size='small', 
            num_latent_tokens=32, 
            token_size=256, 
            dropout=0.0,
            multi_level_query_init=False,
            learnable_1d_query_init=False,
            rope_1d=False,
            downsample_improve=False,
            use_qk_norm=False,
            use_flash_attn=False
            ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = model_size
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn

        # TODO: if verify the downsample_improve, then hard code it as a must
        self.downsample_improve = downsample_improve

        self.multi_level_query_init = multi_level_query_init
        self.learnable_1d_query_init = learnable_1d_query_init

        self.rope_1d = rope_1d

        self.width = {
                "tiny": 256,
                "small": 512,
                "base": 768,
                "large": 1024,
                "xl": 1280,
                "xxl": 1536,
                "xxxl": 2560
            }[self.model_size]
        self.num_layers = {
                "tiny": 4,
                "small": 6,
                "base": 12,
                "large": 24,
                "xl": 36,
                "xxl": 48,
                "xxxl": 48
            }[self.model_size]
        self.num_heads = {
                "tiny": 4,
                "small": 8,
                "base": 12,
                "large": 16,
                "xl": 20,
                "xxl": 24,
                "xxxl": 40
            }[self.model_size]
        
        self.encoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        self.ln_pre = nn.LayerNorm(self.token_size)
        scale = self.width ** -0.5

        assert not (self.multi_level_query_init and self.learnable_1d_query_init)

        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2, 1, self.width))
        
        if self.rope_1d:
            self.freqs_cis_1d = precompute_freqs_cis(
                                    self.grid_size ** 2, 
                                    self.width // self.num_heads, 
                                    base=10000, 
                                    cls_token_num=0)
        else:
            self.latent_token_positional_embedding = nn.Parameter(
                scale * torch.randn(self.num_latent_tokens, 1, self.width))

        if self.learnable_1d_query_init:
            self.query_1d = nn.Parameter(
                scale * torch.randn(self.num_latent_tokens, 1, self.width)
            )
        else:
            self.global_proj = nn.Linear(self.width, self.width, bias=True)

        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(TransformerDecoderLayer(
                self.width, self.num_heads, mlp_ratio=4.0, dropout=dropout,
                query_rope=self.rope_1d, use_qk_norm=self.use_qk_norm, use_flash_attn=self.use_flash_attn,
            ))
        
        if downsample_improve:
            self.ln_post = nn.Identity()
            self.conv_out = nn.Identity()
        else:
            self.ln_post = nn.LayerNorm(self.width)
            self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)

    def forward(self, x, num_q_level=None, causal_type=None, return_feat=False):
        bs = x.shape[0]
        if self.multi_level_query_init:
            x = x.reshape(x.shape[0], x.shape[1], -1)   # B, C, H, W -> B, C, H*W
            x = x.permute(2, 0, 1) # shape = [grid ** 2, B, width]
            x = self.encoder_embed(self.ln_pre(x))
            n_levels = round(np.log2(self.num_latent_tokens))
            latent_tokens = sample_multi_level_1d_tokens(
                                rearrange(x, '(h w) b c -> b c h w', h=self.grid_size), 
                                n_levels, 
                                first_grow="row")
            # if os.environ.get("LOCAL_RANK", 0) == 0:
            #     print(latent_tokens.shape)
            #     print(x.shape)
            latent_tokens = latent_tokens.permute(1, 0, 2)  # num_latent_tokens, B, width
            latent_tokens = self.global_proj(latent_tokens)
        elif self.learnable_1d_query_init:
            x = x.reshape(x.shape[0], x.shape[1], -1)   # B, C, H, W -> B, C, H*W
            x = x.permute(2, 0, 1) # shape = [grid ** 2, B, width]
            x = self.encoder_embed(self.ln_pre(x))
            # n_levels = round(np.log2(self.num_latent_tokens))
            latent_tokens = self.query_1d.repeat(1, bs, 1).to(x.dtype)
        else:
            x = x.reshape(x.shape[0], x.shape[1], -1)   # B, C, H, W -> B, C, H*W
            x = x.permute(2, 0, 1) # shape = [grid ** 2, B, width]
            x = self.encoder_embed(self.ln_pre(x))
            # class embeddings and positional embeddings

            # shape: (num_tokens, B, c), 
            # note: using nn.MultiheadAttention with default batch_first=True, means the input should also be 
            #       sequence length first
            latent_tokens = self.global_proj(x.mean(dim=0, keepdim=True)).repeat(self.num_latent_tokens, 1, 1)
        

        # select a certain number of latent_tokens
        if num_q_level is None:
            selected_token_num = self.num_latent_tokens
        else:
            if causal_type == "per-level":
                selected_token_num = 2**num_q_level
            elif causal_type == "per-token":
                selected_token_num = num_q_level
            else:
                selected_token_num = num_q_level
                # raise ValueError(f"Unknown causal_type: {causal_type}. Choose 'per-token' or 'per-level'.")

        latent_tokens = latent_tokens[:selected_token_num]

        pos_embed = self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]

        if self.rope_1d:
            query_pos = self.freqs_cis_1d.to(x.dtype)[:selected_token_num].to(x.device)
        else:
            query_pos = self.latent_token_positional_embedding.to(x.dtype)[:selected_token_num]

        # get the attn_mask
        if causal_type is None:
            attn_mask = None
        elif causal_type == "per-level":
            attn_mask = get_attn_mask(latent_tokens, causal_type=causal_type)
        elif causal_type == "per-token":
            attn_mask = get_attn_mask(latent_tokens, causal_type=causal_type)
        else:
            attn_mask = None


        # x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            latent_tokens = self.transformer[i](latent_tokens, x, pos=pos_embed, query_pos=query_pos, tgt_mask=attn_mask)
        
        latent_tokens = self.ln_post(latent_tokens)
        # fake 2D shape
        latent_tokens = latent_tokens.permute(1, 2, 0).reshape(bs, self.width, 1, selected_token_num)  # LND -> NDL
        if return_feat:
            return latent_tokens[:, :, :, :selected_token_num]
        latent_tokens = self.conv_out(latent_tokens)

        return latent_tokens


class ViTEncoder2D(nn.Module):
    def __init__(
            self, 
            image_size=256, 
            patch_size=16, 
            model_size='small', 
            token_size=256, 
            dropout=0.0,
            transformer_layer_type="QformerLayerVar"
            ):
        super().__init__()
        self.transformer_layer_type = transformer_layer_type
        assert transformer_layer_type in \
            ["QformerLayerVar", "TransformerLayer", "TransformerDecoderLayer"]
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = model_size
        self.token_size = token_size

        self.width = {
                "tiny": 256,
                "small": 512,
                "base": 768,
                "large": 1024,
                "xl": 1280,
                "xxl": 1536,
                "xxxl": 2560
            }[self.model_size]
        self.num_layers = {
                "tiny": 4,
                "small": 6,
                "base": 12,
                "large": 24,
                "xl": 36,
                "xxl": 48,
                "xxxl": 48
            }[self.model_size]
        self.num_heads = {
                "tiny": 4,
                "small": 8,
                "base": 12,
                "large": 16,
                "xl": 20,
                "xxl": 24,
                "xxxl": 40
            }[self.model_size]
        
        self.encoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        self.ln_pre = nn.LayerNorm(self.token_size)
        scale = self.width ** -0.5

        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2, 1, self.width))

        self.transformer = nn.ModuleList()

        layer_cls = eval(self.transformer_layer_type)
        for i in range(self.num_layers):
            self.transformer.append(layer_cls(
                self.width, self.num_heads, mlp_ratio=4.0, dropout=dropout,
            ))
        self.ln_post = nn.LayerNorm(self.width)
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)

    def forward(self, x, return_feat=False):
        bs = x.shape[0]

        x = x.reshape(x.shape[0], x.shape[1], -1)   # B, C, H, W -> B, C, H*W
        x = x.permute(2, 0, 1) # shape = [grid ** 2, B, width] or LND
        x = self.encoder_embed(self.ln_pre(x))

        # shape: (num_tokens, B, c), 
        # note: using nn.MultiheadAttention with default batch_first=True, means the input should also be 
        #       sequence length first
        pos_embed = self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]

        latent_tokens = x
        if self.transformer_layer_type == "TransformerLayer":
            # this is the original transformer layer, use absolute position embedding
            latent_tokens = latent_tokens + pos_embed
        for i in range(self.num_layers):
            if self.transformer_layer_type == "TransformerDecoderLayer":
                latent_tokens = self.transformer[i](
                                    latent_tokens, 
                                    latent_tokens, 
                                    pos=pos_embed, 
                                    query_pos=pos_embed, 
                                    tgt_mask=None
                                    )
            elif self.transformer_layer_type == "TransformerLayer":
                latent_tokens = self.transformer[i](
                                    latent_tokens
                                    )
            elif self.transformer_layer_type == "QformerLayerVar":
                latent_tokens = self.transformer[i](
                                    latent_tokens, 
                                    pos=pos_embed
                                    )
            # latent_tokens = self.transformer[i](x, pos=pos_embed)
        
        latent_tokens = self.ln_post(latent_tokens)
        # 2D shape; L N D -> N D H W
        latent_tokens = rearrange(latent_tokens, '(h w) b c -> b c h w', h=self.grid_size, w=self.grid_size)
        if return_feat:
            return latent_tokens
        latent_tokens = self.conv_out(latent_tokens)

        return latent_tokens


    

class ViTDecoder_V2(nn.Module):
    def __init__(
            self, 
            image_size=256, 
            patch_size=16, 
            model_size='small', 
            num_latent_tokens=32, 
            token_size=256, 
            dropout=0.0,
            last_level_2d_query_init=False,
            multi_level_2d_query_init=False,
            learnable_2d_query_init=False,
            rope_2d=True
            ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = model_size
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size
        self.last_level_2d_query_init = last_level_2d_query_init
        self.multi_level_2d_query_init = multi_level_2d_query_init
        self.learnable_2d_query_init = learnable_2d_query_init
        # rope_2d will be effective only for 2d query embedding
        self.rope_2d = rope_2d
        assert not (self.last_level_2d_query_init and self.multi_level_2d_query_init)
        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 6,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        self.ln_pre = nn.LayerNorm(self.token_size)
        scale = self.width ** -0.5

        if self.learnable_2d_query_init:
            self.q_2d = nn.Parameter(scale * torch.randn(self.grid_size ** 2, 1, self.width))
        
        if self.rope_2d:
            self.freqs_cis = precompute_freqs_cis_2d(self.grid_size, self.width // self.num_heads, base=10000, cls_token_num=0)
            # self.freqs_cis = precompute_freqs_cis_2d(self.grid_size, self.width, cls_token_num=0)
        else:
            self.positional_embedding = nn.Parameter(
                    scale * torch.randn(self.grid_size ** 2, 1, self.width))
        # add mask token and query pos embed
        # self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, 1, self.width))
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(TransformerDecoderLayer(
                self.width, self.num_heads, mlp_ratio=4.0, dropout=dropout,
                query_rope=self.rope_2d,
            ))
        self.ln_post = nn.LayerNorm(self.width)

        self.conv_out = nn.Conv2d(self.width, token_size, kernel_size=3, stride=1, padding=1)
    
    def forward(self, z_quantized):
        N, C, H, W = z_quantized.shape
        # assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        selected_latent_tokens = W
        x = z_quantized.reshape(N, C*H, W).permute(2, 0, 1) # LND
        x = self.decoder_embed(self.ln_pre(x))

        seq_len, bs, _ = x.shape    # shape: (num_latent_tokens, B, c)

        if self.last_level_2d_query_init:
            n_level = round(np.log2(selected_latent_tokens))
            # (B, C, grid_size, grid_size)
            latent_tokens = last_level_1d_features_to_2d_maps(
                x_1d=rearrange(x, 'l b c -> b l c', b=bs),
                n_level=n_level,
                max_h=self.grid_size,
                max_w=self.grid_size,
                first_grow="row",
            )
            latent_tokens = rearrange(latent_tokens, 'b c h w -> (h w) b c').to(x.dtype)
            # (grid_size**2, B, c)
        elif self.multi_level_2d_query_init:
            n_level = round(np.log2(selected_latent_tokens))
            # (B, C, grid_size, grid_size)
            latent_tokens = multi_level_1d_features_to_2d_maps_avg(
                x_1d=rearrange(x, 'l b c -> b l c', b=bs),
                n_level=n_level,
                max_h=self.grid_size,
                max_w=self.grid_size,
                first_grow="row",
            )
            latent_tokens = rearrange(latent_tokens, 'b c h w -> (h w) b c').to(x.dtype)
        elif self.learnable_2d_query_init:
            latent_tokens = self.q_2d.repeat(1, bs, 1).to(x.dtype)
        else:
            latent_tokens = x[:1, :, :].repeat(self.grid_size**2, 1, 1).to(x.dtype)

        # mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        # x = x + self.latent_token_positional_embedding[:seq_len]
        # x = torch.cat([mask_tokens, x], dim=1)

        if self.rope_2d:
            query_pos = self.freqs_cis.to(x.dtype).to(x.device)
        else:
            query_pos = self.positional_embedding.repeat(1, bs, 1).to(x.dtype) # shape = [*, grid ** 2 + 1, width]
        pos_embed = self.latent_token_positional_embedding[:selected_latent_tokens].repeat(1, bs, 1).to(x.dtype)

        for i in range(self.num_layers):
            latent_tokens = self.transformer[i](latent_tokens, x, pos=pos_embed, query_pos=query_pos)

        latent_tokens = self.ln_post(latent_tokens)
        # L N D -> N D H W
        latent_tokens = latent_tokens.permute(1, 2, 0).reshape(bs, self.width, self.grid_size, self.grid_size)
        latent_tokens = self.conv_out(latent_tokens.contiguous())
        return latent_tokens


class ViTDecoder(nn.Module):
    def __init__(
            self, 
            image_size=256, 
            patch_size=16, 
            model_size='small', 
            num_latent_tokens=32, 
            token_size=256, 
            dropout=0.0,
            last_level_2d_query_init=False,
            multi_level_2d_query_init=False,
            learnable_2d_query_init=False,
            q_upsample=False,
            out_inner_feat=False,
            out_inner_dim=768,   # for dino-v2
            out_inner_depth=None,
            use_qk_norm=False,
            use_flash_attn=False,
            ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = model_size
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size
        self.last_level_2d_query_init = last_level_2d_query_init
        self.multi_level_2d_query_init = multi_level_2d_query_init
        self.learnable_2d_query_init = learnable_2d_query_init
        self.out_inner_feat = out_inner_feat
        self.out_inner_depth = out_inner_depth

        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn

        assert not (self.last_level_2d_query_init and self.multi_level_2d_query_init)
        self.width = {
                "tiny": 256,
                "small": 512,
                "base": 768,
                "large": 1024,
                "xl": 1280,
                "xxl": 1536,
            }[self.model_size]
        self.num_layers = {
                "tiny": 4,
                "small": 6,
                "base": 12,
                "large": 24,
                "xl": 36,
                "xxl": 48,
            }[self.model_size]
        self.num_heads = {
                "tiny": 4,
                "small": 8,
                "base": 12,
                "large": 16,
                "xl": 20,
                "xxl": 24,
            }[self.model_size]
 

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        self.ln_pre = nn.LayerNorm(self.token_size)
        scale = self.width ** -0.5

        if self.learnable_2d_query_init:
            self.q_2d = nn.Parameter(scale * torch.randn(self.grid_size ** 2, 1, self.width))

        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2, 1, self.width))
        # add mask token and query pos embed
        # self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, 1, self.width))
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(TransformerDecoderLayer(
                self.width, self.num_heads, mlp_ratio=4.0, dropout=dropout,
                use_qk_norm=self.use_qk_norm, use_flash_attn=self.use_flash_attn
            ))
        self.ln_post = nn.LayerNorm(self.width)

        self.conv_out = nn.Conv2d(self.width, token_size, kernel_size=3, stride=1, padding=1)

        self.q_upsample = q_upsample
        if self.q_upsample:
            self.upsampler = QFormerUpsample(
                                width=self.width, 
                                nhead=self.num_heads, 
                                dropout=dropout, 
                                num_queries=num_latent_tokens
                            )
        
        if out_inner_feat:
            self.distill_mlp = nn.Sequential(
                    nn.Linear(self.width, self.width * 4),
                    nn.SiLU(),
                    nn.Linear(self.width * 4, self.width * 4),
                    nn.SiLU(),
                    nn.Linear(self.width * 4, out_inner_dim),
                    )

    
    def forward(
            self, 
            z_quantized, 
            ret_inner_feat=False,   # return inner feature(through mlp) for distillation
            return_feat=False,      # return feature for linear probe
            ):
        N, C, H, W = z_quantized.shape
        # assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        selected_latent_tokens = W
        x = z_quantized.reshape(N, C*H, W).permute(2, 0, 1) # LND
        x = self.decoder_embed(self.ln_pre(x))
        # upsample
        if self.q_upsample:
            x = self.upsampler(x)
            selected_latent_tokens = self.num_latent_tokens

        seq_len, bs, _ = x.shape    # shape: (num_latent_tokens, B, c)

        if self.last_level_2d_query_init:
            n_level = round(np.log2(selected_latent_tokens))
            # (B, C, grid_size, grid_size)
            latent_tokens = last_level_1d_features_to_2d_maps(
                x_1d=rearrange(x, 'l b c -> b l c', b=bs),
                n_level=n_level,
                max_h=self.grid_size,
                max_w=self.grid_size,
                first_grow="row",
            )
            latent_tokens = rearrange(latent_tokens, 'b c h w -> (h w) b c').to(x.dtype)
            # (grid_size**2, B, c)
        elif self.multi_level_2d_query_init:
            n_level = round(np.log2(selected_latent_tokens))
            # (B, C, grid_size, grid_size)
            latent_tokens = multi_level_1d_features_to_2d_maps_avg(
                x_1d=rearrange(x, 'l b c -> b l c', b=bs),
                n_level=n_level,
                max_h=self.grid_size,
                max_w=self.grid_size,
                first_grow="row",
            )
            latent_tokens = rearrange(latent_tokens, 'b c h w -> (h w) b c').to(x.dtype)
        elif self.learnable_2d_query_init:
            latent_tokens = self.q_2d.repeat(1, bs, 1).to(x.dtype)
        else:
            latent_tokens = x[:1, :, :].repeat(self.grid_size**2, 1, 1).to(x.dtype)

        # mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype
        # x = x + self.latent_token_positional_embedding[:seq_len]
        # x = torch.cat([mask_tokens, x], dim=1)

        query_pos = self.positional_embedding.repeat(1, bs, 1).to(x.dtype) # shape = [*, grid ** 2 + 1, width]
        pos_embed = self.latent_token_positional_embedding[:selected_latent_tokens].repeat(1, bs, 1).to(x.dtype)

        for i in range(self.num_layers):
            latent_tokens = self.transformer[i](latent_tokens, x, pos=pos_embed, query_pos=query_pos)
            if self.out_inner_feat and ret_inner_feat and (i + 1) == self.out_inner_depth:
                inner_feat = self.distill_mlp(latent_tokens)

            elif self.out_inner_feat and return_feat and (i + 1) == self.out_inner_depth:
                # return the feature without distill_mlp, used for linear probe
                assert not (ret_inner_feat and return_feat), \
                    "ret_inner_feat and return_feat cannot be True at the same time"
                inner_feat = latent_tokens
                # L N D -> N L D
                inner_feat = inner_feat.permute(1, 0, 2)
                # assert inner_feat.shape[1] == self.width, \
                #     f"inner_feat.shape[1]={inner_feat.shape[1]} != self.width={self.width},"\
                #     f"current latent_tokens.shape={latent_tokens.shape}"\
                #     f"current x.shape={x.shape}"
                return None, inner_feat

        latent_tokens = self.ln_post(latent_tokens)
        # L N D -> N D H W
        latent_tokens = latent_tokens.permute(1, 2, 0).reshape(bs, self.width, self.grid_size, self.grid_size)
        latent_tokens = self.conv_out(latent_tokens.contiguous())
        if self.out_inner_feat and (ret_inner_feat or return_feat):
            # L N D -> N L D
            inner_feat = inner_feat.permute(1, 0, 2)
            return latent_tokens, inner_feat
        else:
            return latent_tokens


class ViTDecoder2D(nn.Module):
    def __init__(
            self, 
            image_size=256, 
            patch_size=16, 
            model_size='small', 
            token_size=256, 
            dropout=0.0,
            out_inner_feat=False,
            out_inner_dim=768,   # for dino-v2
            out_inner_depth=None,
            transformer_layer_type="QformerLayerVar"
            ):
        super().__init__()
        self.transformer_layer_type = transformer_layer_type
        assert self.transformer_layer_type in \
            ["QformerLayerVar", "TransformerLayer","TransformerDecoderLayer" ]

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = model_size
        self.token_size = token_size
        self.out_inner_feat = out_inner_feat
        self.out_inner_depth = out_inner_depth
        self.width = {
                "tiny": 256,
                "small": 512,
                "base": 768,
                "large": 1024,
                "xl": 1280,
                "xxl": 1536,
                "xxxl": 2560
            }[self.model_size]
        self.num_layers = {
                "tiny": 4,
                "small": 6,
                "base": 12,
                "large": 24,
                "xl": 36,
                "xxl": 48,
                "xxxl": 48
            }[self.model_size]
        self.num_heads = {
                "tiny": 4,
                "small": 8,
                "base": 12,
                "large": 16,
                "xl": 20,
                "xxl": 24,
                "xxxl": 40
            }[self.model_size]

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        self.ln_pre = nn.LayerNorm(self.token_size)
        scale = self.width ** -0.5

        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2, 1, self.width))
        self.transformer = nn.ModuleList()
        layer_cls = eval(self.transformer_layer_type)
        for i in range(self.num_layers):
            self.transformer.append(layer_cls(
                self.width, self.num_heads, mlp_ratio=4.0, dropout=dropout,
            ))
        self.ln_post = nn.LayerNorm(self.width)

        self.conv_out = nn.Conv2d(self.width, token_size, kernel_size=3, stride=1, padding=1)
       
        if out_inner_feat:
            self.distill_mlp = nn.Sequential(
                    nn.Linear(self.width, self.width * 4),
                    nn.SiLU(),
                    nn.Linear(self.width * 4, self.width * 4),
                    nn.SiLU(),
                    nn.Linear(self.width * 4, out_inner_dim),
                    )

    
    def forward(self, z_quantized, ret_inner_feat=False):
        N, C, H, W = z_quantized.shape
        selected_latent_tokens = W
        x = z_quantized.reshape(N, C, H*W).permute(2, 0, 1) # LND
        x = self.decoder_embed(self.ln_pre(x))

        seq_len, bs, _ = x.shape    # shape: (num_latent_tokens, B, c)
        latent_tokens = x

        pos_embed = self.positional_embedding.repeat(1, bs, 1).to(x.dtype) # shape = [*, grid ** 2 + 1, width]
        if self.transformer_layer_type == "TransformerLayer":
                    # this is the original transformer layer, use absolute position embedding
                    latent_tokens = latent_tokens + pos_embed

        for i in range(self.num_layers):
            if self.transformer_layer_type == "TransformerDecoderLayer":
                latent_tokens = self.transformer[i](
                                    latent_tokens, 
                                    latent_tokens, 
                                    pos=pos_embed, 
                                    query_pos=pos_embed, 
                                    tgt_mask=None
                                    )
            elif self.transformer_layer_type == "TransformerLayer":
                latent_tokens = self.transformer[i](
                                    latent_tokens
                                    )
            elif self.transformer_layer_type == "QformerLayerVar":
                latent_tokens = self.transformer[i](
                                    latent_tokens, 
                                    pos=pos_embed
                                    )
            if self.out_inner_feat and ret_inner_feat and (i + 1) == self.out_inner_depth:
                inner_feat = self.distill_mlp(latent_tokens)

        latent_tokens = self.ln_post(latent_tokens)
        # L N D -> N D H W
        latent_tokens = latent_tokens.permute(1, 2, 0).reshape(bs, self.width, self.grid_size, self.grid_size)
        latent_tokens = self.conv_out(latent_tokens.contiguous())
        if self.out_inner_feat and ret_inner_feat:
            # L N D -> N L D
            inner_feat = inner_feat.permute(1, 0, 2)
            return latent_tokens, inner_feat
        else:
            return latent_tokens



class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, 
                 norm_type='group', dropout=0.0, resamp_with_conv=True, z_channels=256,
                 use_attn=True, res_down_sample=False, downsample_match_channel=False
                 ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.use_attn = use_attn
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        prev_out_dim = ch

        # downsampling
        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # res
            res_block = nn.ModuleList()
            if self.use_attn:
                attn_block = nn.ModuleList()
            # block_in = ch*in_ch_mult[i_level]
            block_in = prev_out_dim
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if self.use_attn:
                    if i_level == self.num_resolutions - 1:
                        attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            if self.use_attn:
                conv_block.attn = attn_block
            # downsample
            if i_level != self.num_resolutions-1:
                if res_down_sample:
                    assert resamp_with_conv, 'res_down_sample only support resamp_with_conv'
                    if downsample_match_channel:
                        conv_block.downsample = DownsamplerWithPixunshuffleResidual(
                                                    block_in, 
                                                    ch*ch_mult[i_level+1],
                                                )
                        prev_out_dim = ch*ch_mult[i_level+1]
                    else:
                        conv_block.downsample = DownsamplerWithPixunshuffleResidual(block_in)
                        prev_out_dim = block_out
                else:
                    conv_block.downsample = Downsample(block_in, resamp_with_conv)
                    prev_out_dim = block_out

            self.conv_blocks.append(conv_block)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        if self.use_attn:
            self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if self.use_attn and len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
        
        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h



class Decoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, norm_type="group",
                 dropout=0.0, resamp_with_conv=True, out_channels=3, 
                 adaptive_gn=False, d2s_up=False, use_attn=True,
                 res_up_sample=False, upsample_match_channel=False
                 ):
        """
        adaptive_gn: whether to use adaptive group normalization as in MAGVIT-v2
        d2s_up: whether to use depth_to_space for up sampling
        res_up: whether to use residual non-parametric depth-to-space when upsampling
        """
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch*ch_mult[self.num_resolutions-1]
        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.adaptive_gn = adaptive_gn
        self.d2s_up = d2s_up

        self.use_attn = use_attn

       # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        if self.use_attn:
            self.mid.append(AttnBlock(block_in, norm_type=norm_type))

        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # upsampling
        prev_out_dim = block_in

        self.conv_blocks = nn.ModuleList()
        if self.adaptive_gn:
            self.adaptive = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            if self.use_attn:
                attn_block = nn.ModuleList()
            block_in = prev_out_dim
            block_out = ch*ch_mult[i_level]
            if self.adaptive_gn:
                self.adaptive.append(AdaptiveGroupNorm(z_channels, block_in))
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if self.use_attn and i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            if self.use_attn:
                conv_block.attn = attn_block
            # upsample
            if i_level != 0:
                if d2s_up:
                    assert not res_up_sample,'d2s_up or res_up_sample can not be True at the same time'
                    conv_block.upsample = D2SUpsampler(block_in)
                    prev_out_dim = block_in
                elif res_up_sample:
                    if upsample_match_channel:
                        conv_block.upsample = UpsamplerWithPixshuffleDupResidual(
                                                    block_in,
                                                    ch*ch_mult[i_level-1],
                                                )
                        prev_out_dim = ch*ch_mult[i_level-1]
                    else:
                        conv_block.upsample = UpsamplerWithPixshuffleDupResidual(block_in)
                        prev_out_dim = block_in
                else:
                    conv_block.upsample = Upsample(block_in, resamp_with_conv)
                    prev_out_dim = block_in

            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight
    
    def forward(self, z):

        if self.adaptive_gn:
            style = z.clone()

        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            if self.adaptive_gn:
                ### pass in each resblock first adaGN
                try:
                    h = self.adaptive[i_level](h, style)
                except Exception as e:
                    error_info = str(e) + f"Showing the h shape: {h.shape}, {style.shape}"
                    raise ValueError(error_info)
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if self.use_attn and len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h


   

############################################################################
## Deprecated
############################################################################


def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


class QFormerUpsample(nn.Module):
    """
    Cross attn module as an Upsampler
    """
    def __init__(
        self,
        width,
        nhead,
        dropout,
        num_queries=256,
        mlp_ratio=4.0,
        activation=nn.GELU,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, nhead, dropout=dropout)
        self.cross_attn= nn.MultiheadAttention(width, nhead, dropout=dropout)
        scale = width ** -0.5
        self.query_1d = nn.Parameter(
            scale * torch.randn(num_queries, 1, width)
        )

        dim_feedforward = int(width * mlp_ratio)
        self.linear1 = nn.Linear(width, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, width)

        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation()

    def forward(self, x):
        x_2 = self.norm1(x) # N, B, C
        x_2 = self.self_attn(x_2, x_2, value=x_2)[0]
        x = x + self.dropout1(x_2)
        x = self.norm2(x)

        x = self.cross_attn(query=self.query_1d, key=x, value=x)[0]
        x_2 = self.norm3(x)
        x_2 = self.linear2(self.dropout(self.activation(self.linear1(x_2))))
        x = x + self.dropout3(x_2)
        return x




class UpsamplerWithPixshuffleDupResidual(nn.Module):
    """
    Using pixshuffle for upsampling, and with a residual connction.
    The residual is the depth to space + channel duplicated value
    """
    def __init__(
        self,
        dim,
        dim_out=None,
        factor=2,    # the spatial upsampling factor
    ):
        super().__init__()

        # for the main upsampler
        self.dim_out = dim if dim_out is None else dim_out
        self.factor = factor
        self.conv1 = nn.Conv2d(dim, self.dim_out * factor**2, (3, 3), padding=1)

        # for residual non-parameteric connections
        # note we
        assert self.dim_out * factor**2 % dim == 0
        self.repeats = self.dim_out * factor**2 // dim


    def forward(self, x: torch.Tensor):
        """
        input_image: [B C H W]
        """
        # we use the implementation from efficientvit/models/nn/ops: first duplicate then shuffle
        # but this is not exactly the same as the LARP paper presents(first shuffle then duplicate).
        residual = x.repeat_interleave(self.repeats, dim=1)
        residual = F.pixel_shuffle(residual, self.factor)
 
        out = self.conv1(x)
        out = F.pixel_shuffle(out, self.factor)
        return out + residual

class DownsamplerWithPixunshuffleResidual(nn.Module):
    """
    Using pixshuffle for upsampling, and with a residual connction.
    The residual is the depth to space + channel duplicated value
    """
    def __init__(
        self,
        dim,
        dim_out=None,
        factor=2,    # the spatial downsampling factor
    ):
        super().__init__()

        # for the main downsampler
        self.dim_out = dim if dim_out is None else dim_out
        self.factor = factor
        self.conv1 = nn.Conv2d(dim, self.dim_out // factor**2, (3, 3), padding=1)

        # for residual non-parameteric connections
        # note we
        self.factor = factor
        assert dim * factor**2 % self.dim_out == 0
        self.group_size = dim * factor**2 // self.dim_out


    def forward(self, x: torch.Tensor):
        """
        input_image: [B C H W]
        """
        residual = F.pixel_unshuffle(x, self.factor)
        B, C, H, W = residual.shape
        residual = residual.view(B, self.dim_out, self.group_size, H, W)
        residual = residual.mean(dim=2)
 
        out = self.conv1(x)
        out = F.pixel_unshuffle(out, self.factor)
        return out + residual


class ChannelDownsampleResidual(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
    ):
        """
        Down sample the width by a conv and a shortcut with channel averaging
        """
        super().__init__()
        # for the main downsampler
        self.dim_out = dim if dim_out is None else dim_out
        self.norm = Normalize(dim)
        self.nonlinear = nn.SiLU()
        self.conv1 = nn.Conv2d(dim, self.dim_out,  (3, 3), padding=1)

        self.in_channels = dim
        self.out_channels = dim_out
        assert self.in_channels % self.out_channels == 0
        self.group_size = self.in_channels // self.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x.view(B, self.out_channels, self.group_size, H, W)
        residual = residual.mean(dim=2)

        x = self.norm(x)
        x = self.nonlinear(x)
        x = self.conv1(x)

        return x + residual


class ChannelUpsampleResidual(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
    ):
        """
        Up sample the width by a conv and a shortcut with channel duplication
        """
        super().__init__()
        # for the main downsampler
        self.dim_out = dim if dim_out is None else dim_out
        self.conv1 = nn.Conv2d(dim, self.dim_out,  (3, 3), padding=1)

        self.in_channels = dim
        self.out_channels = dim_out
        assert self.out_channels % self.in_channels == 0
        self.repeats = self.out_channels // self.in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.repeat_interleave(self.repeats, dim=1)
        x = self.conv1(x)

        return x + residual



class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class QFormerUpsample(nn.Module):
    """
    Cross attn module as an Upsampler
    """
    def __init__(
        self,
        width,
        nhead,
        dropout,
        num_queries=256,
        mlp_ratio=4.0,
        activation=nn.GELU,
    ):
        super().__init__()
        self.cross_attn= nn.MultiheadAttention(width, nhead, dropout=dropout)
        scale = width ** -0.5
        self.num_queries = num_queries
        self.query_1d = nn.Parameter(
            scale * torch.randn(num_queries, 1, width)
        )

        self.pos_emb = nn.Parameter(
            scale * torch.randn(num_queries, 1, width)
        )

        dim_feedforward = int(width * mlp_ratio)
        self.linear1 = nn.Linear(width, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, width)

        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        # self.dropout1 = nn.Dropout(dropout)

        self.activation = activation()

    def forward(self, x):
        if x.shape[0] == self.num_queries:
            return x
        else:
            x_2 = self.norm1(x)
            x_2 = self.pos_emb[:x.shape[0]].repeat(1, x.shape[1], 1).to(x.dtype) + x_2
            query = self.query_1d[x.shape[0]:].repeat(1, x.shape[1], 1).to(x.dtype)
            x_2 = self.cross_attn(query=query, key=x_2, value=x_2)[0]
            x_2 = self.norm2(x_2)
            x_2 = self.linear2(self.dropout(self.activation(self.linear1(x_2))))
            # concat at the first dimension
            x = torch.cat([x, x_2], dim=0)
            return x


# ------------------------------------------------------------------------------
# tokenizer/tokenizer_image/vq/lfq.py
# ------------------------------------------------------------------------------
"""
Modified from 
https://github.com/TencentARC/Open-MAGVIT2/blob//taming/modules/vqvae/lookup_free_quantize.py
"""






def entropy(prob):
    return (-prob * torch.log(prob + 1e-5)).sum(dim=-1)

# class

def mult_along_first_dims(x, y):
    """
    returns x * y elementwise along the leading dimensions of y
    """
    ndim_to_expand = x.ndim - y.ndim
    for _ in range(ndim_to_expand):
        y = y.unsqueeze(-1)
    return x * y


def masked_mean(x, m):
    """
    takes the mean of the elements of x that are not masked
    the mean is taken along the shared leading dims of m
    equivalent to: x[m].mean(tuple(range(m.ndim)))

    The benefit of using masked_mean rather than using
    tensor indexing is that masked_mean is much faster
    for torch-compile on batches.

    The drawback is larger floating point errors
    """
    x = mult_along_first_dims(x, m)
    x = x / m.sum()
    return x.sum(tuple(range(m.ndim)))

def entropy_loss(
    logits,
    mask=None,
    temperature=0.01,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
    eps=1e-5,
):
    """
    Entropy loss of unnormalized logits

    logits: Affinities are over the last dimension

    https://github.com/google-research/magvit/blob/05e8cfd6559c47955793d70602d62a2f9b0bdef5/videogvt/train_lib/losses.py#L279
    LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)
    """
    probs = F.softmax(logits / temperature, -1)
    log_probs = F.log_softmax(logits / temperature + eps, -1)

    if mask is not None:
        # avg_probs = probs[mask].mean(tuple(range(probs.ndim - 1)))
        # avg_probs = einx.mean("... D -> D", probs[mask])

        avg_probs = masked_mean(probs, mask)
        # avg_probs = einx.mean("... D -> D", avg_probs)
    else:
        avg_probs = reduce(probs, "... D -> D", "mean")

    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

    sample_entropy = -torch.sum(probs * log_probs, -1)
    if mask is not None:
        # sample_entropy = sample_entropy[mask].mean()
        sample_entropy = masked_mean(sample_entropy, mask).mean()
    else:
        sample_entropy = torch.mean(sample_entropy)

    loss = (sample_minimization_weight * sample_entropy) - (
        batch_maximization_weight * avg_entropy
    )

    return sample_entropy, avg_entropy, loss


def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


class LFQ(nn.Module):
    """
    Modified from 
    https://github.com/TencentARC/Open-MAGVIT2/blob/taming/modules/vqvae/lookup_free_quantize.py
    """
    def __init__(
        self,
        dim,
        beta, 
        entropy_loss_ratio, 
        n_e=None,   # codebook size
        num_codebooks = 1,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
        token_factorization = False,
        factorized_bits = [9, 9],
        show_usage = True,
    ):
        super().__init__()

        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.show_usage = show_usage

        # some assert validations
        assert dim is not None or n_e is not None, \
            'either dim or codebook_size must be specified for LFQ'

        assert n_e is None or np.log2(n_e).is_integer(), \
            f'your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** np.ceil(np.log2(n_e))})'

        self.n_e = default(n_e, lambda: 2 ** dim)
        self.e_dim = int(np.log2(n_e))

        codebook_dims = self.e_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = dim != codebook_dims
        self.has_projections = has_projections

        self.dim = dim
        self.e_dim = self.e_dim
        self.num_codebooks = num_codebooks
        
        # for entropy loss
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight

        # for no auxiliary loss, during inference
        self.token_factorization = token_factorization
        if not self.token_factorization: #for first stage model
            # used for bits to indices
            self.register_buffer('mask', 2 ** torch.arange(self.e_dim), persistent=False)
        else:
            self.factorized_bits = factorized_bits
            self.register_buffer("pre_mask", 2** torch.arange(factorized_bits[0]), persistent=False)
            self.register_buffer("post_mask", 2**torch.arange(factorized_bits[1]), persistent=False)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # codes
        all_codes = torch.arange(n_e)
        bits = self.indices_to_bits(all_codes)
        codebook = bits * 2.0 - 1.0

        self.register_buffer('codebook', codebook, persistent = False)


        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

    @property
    def dtype(self):
        return self.codebook.dtype
    
    def indices_to_bits(self, x):
        """
        x: long tensor of indices

        returns big endian bits (bool, True for 1, False for -1)
        """
        mask = 2 ** torch.arange(self.e_dim, device=x.device, dtype=torch.long)
        # x is now big endian bits, the last dimension being the bits
        x = (x.unsqueeze(-1) & mask) != 0
        return x

    def get_codebook_entry(self, x, shape, order): #0610
        if self.token_factorization:
            if order == "pre":
                mask = 2 ** torch.arange(self.factorized_bits[0], device=x.device, dtype=torch.long)
            else:
                mask = 2 ** torch.arange(self.factorized_bits[1], device=x.device, dtype=torch.long)
        else:
            mask = 2 ** torch.arange(self.e_dim, device=x.device, dtype=torch.long)
        
        # indices_to_bits
        x = (x.unsqueeze(-1) & mask) != 0
        x = x * 2.0 - 1.0 #back to the float
        ## scale back to the 
        b, c, h, w = shape
        x = rearrange(x, "(b h w) c -> b h w c", h=h, w=w, b=b)
        x = rearrange(x, "b h w c -> b c h w")
        return x

    def bits_to_indices(self, bits):
        """
        bits: bool tensor of big endian bits, where the last dimension is the bit dimension

        returns indices, which are long integers from 0 to self.codebook_size
        """
        assert bits.shape[-1] == self.e_dim
        indices = 2 ** torch.arange(
            0,
            self.e_dim,
            1,
            dtype=torch.long,
            device=bits.device,
        )
        return (bits * indices).sum(-1)
    
    def decode(self, x):
        """
        x: ... NH
            where NH is number of codebook heads
            A longtensor of codebook indices, containing values from
            0 to self.codebook_size
        """
        x = self.indices_to_bits(x)
        # to some sort of float
        x = x.to(self.dtype)
        # -1 or 1
        x = x * 2 - 1
        x = rearrange(x, "... NC Z-> ... (NC Z)")
        return x

    def forward(
        self,
        z,
        mask = None,
        return_loss = True,
        stochastic = False,
        **kwargs
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """
        for k, v in kwargs.items():
            assert v is None or not v, f'unsupported parameter {k}'

        z = rearrange(z, 'b d ... -> b ... d')
        z, ps = pack_one(z, 'b * d')
        # split out number of codebooks

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)


        codebook_value = torch.Tensor([1.0]).to(device=z.device, dtype=z.dtype)

        if stochastic:
            z = torch.sigmoid(z)
            quantized = torch.bernoulli(z)
            quantized = (quantized - 0.5) * 2.0 * codebook_value # -1 or 1
        else:
            quantized = torch.where(z > 0, codebook_value, -codebook_value) # higher than 0 filled 

        # calculate indices
        if self.token_factorization:
            indices_pre = reduce((quantized[..., :self.factorized_bits[0]] > 0).int() * self.pre_mask.int(), "b n c d -> b n c", "sum")
            indices_post = reduce((quantized[..., self.factorized_bits[0]:] > 0).int() * self.post_mask.int(), "b n c d -> b n c", "sum")
        else:
            indices = reduce((quantized > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')

        # entropy aux loss

        if self.training and return_loss:
            logits = 2 * einsum('... i d, j d -> ... i j', z, self.codebook)
            # the same as euclidean distance up to a constant
            per_sample_entropy, codebook_entropy, entropy_aux_loss = entropy_loss(
                logits = logits,
                sample_minimization_weight = self.sample_minimization_weight,
                batch_maximization_weight = self.batch_maximization_weight
            )
            avg_probs = self.zero
            if self.entropy_loss_ratio > 0:
                entropy_aux_loss = self.entropy_loss_ratio * entropy_aux_loss
            else:
                entropy_aux_loss = 0
        else:
            # logits = 2 * einsum('... i d, j d -> ... i j', x, self.codebook)
            # probs = F.softmax(logits / 0.01, -1)
            # avg_probs = reduce(probs, "b n c d -> b d", "mean")
            # avg_probs = torch.sum(avg_probs, 0) #batch dimension
            # if not training, just return dummy 0
            per_sample_entropy = codebook_entropy = self.zero
            ## calculate the codebook_entropy needed for one batch evaluation
            entropy_aux_loss = self.zero
            avg_probs = self.zero

        # commit loss

        if self.training:
            commit_loss = F.mse_loss(z, quantized.detach(), reduction = 'none')

            if exists(mask):
                commit_loss = commit_loss[mask]

            commit_loss = commit_loss.mean()
            commit_loss = self.beta * commit_loss
        else:
            commit_loss = self.zero


        # use straight-through gradients (optionally with custom activation fn) if training

        quantized = z + (quantized - z).detach() #transfer to quantized

        # merge back codebook dim

        quantized = rearrange(quantized, 'b n c d -> b n (c d)')

        # reconstitute image or video dimensions, i.e. b c h w or b c t h w
        quantized = unpack_one(quantized, ps, 'b * d')
        quantized = rearrange(quantized, 'b ... d -> b d ...')

        
        if self.token_factorization:
            indices_pre = unpack_one(indices_pre, ps, "b * c")
            indices_post = unpack_one(indices_post, ps, "b * c")
            indices_pre = indices_pre.flatten()
            indices_post = indices_post.flatten()
            indices = (indices_pre, indices_post)
        else:
            indices = unpack_one(indices, ps, 'b * c')
            indices = indices.flatten() # b * h * w

        codebook_usage = 0
        if self.show_usage and self.training:
            cur_len = indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = indices 
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        perplexity = None
        min_encodings = None

        return quantized, (commit_loss, entropy_aux_loss, codebook_usage), (perplexity, min_encodings, indices)


# ------------------------------------------------------------------------------
# tokenizer/tokenizer_image/vq/gptc.py
# ------------------------------------------------------------------------------
"""
Codes adapted from https://github.com/hywang66/LARP/

This is the continuous gpt model. It will take token embedding as input, and output continuous features
The features will be used to calculate the logits using similarity.
Currently, this gptc is only for regularization when training tokenizers.
"""






@dataclass
class GPTCConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    max_seq_len: int = 1024
    n_ind: int = 16 # number of input dim
    n_embd: int = 1024
    n_head: int = 16
    n_layer: int = 24
    detach_x: bool = False
    detach_target: bool = True
    l2_normalized: bool = True
    n_classes: int = -1
    fully_separated: bool = False



class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head

        self.p_attn_drop = config.attn_pdrop

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        present = torch.stack((k, v)) # (2, B, nh, T, hs)
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        if hasattr(F, "scaled_dot_product_attention") and torch.__version__ >= "2.1.0":
            is_causal = layer_past is None 
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p_attn_drop, is_causal=is_causal)  
        else:
            raise NotImplementedError("scaled_dot_product_attention not available in this version of PyTorch")
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present  


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        # TODO: check that training still works
        if return_present: assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)

        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x


class GPTC(nn.Module):
    """  the continuous GPT model"""
    def __init__(self, config: GPTCConfig) -> None:
        super().__init__()
        # input embedding stem
        self.input_proj = nn.Linear(config.n_ind, config.n_embd)
        self.pos_emb = nn.Parameter(torch.randn(1, config.max_seq_len, config.n_embd) * 0.02) 
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        self.apply(self._init_weights)
        self.config = config
        self.max_seq_length = config.max_seq_len
        self.detach_x = config.detach_x
        self.detach_target = config.detach_target
        self.l2_normalized = config.l2_normalized

        self.n_classes = config.n_classes
        self.fully_separated = config.fully_separated
        assert not (self.detach_x and self.detach_target), 'Cannot detach both x and target'
        self.head = nn.Linear(config.n_embd, config.n_ind)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, targets=None):
        # forward the GPTC model
        # x: (b, n, n_ind)
        token_embeddings = self.input_proj(x) # (b, n, d)

        x = self.drop(token_embeddings + self.pos_emb[:, :token_embeddings.shape[1], :])

        x = self.blocks(x)

        x = self.ln_f(x)
        pred = self.head(x) # (b, n, n_ind)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            if self.diff_loss:
                loss = self.diff_loss_module.loss(pred, targets)
            else:
                loss = F.mse_loss(pred, targets)

        return pred, loss
    
    def compute_prior_loss(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, n, n_ind) 
        if self.l2_normalized:
            x = F.normalize(x, p=2, dim=-1)
            
        target = x[:, 1:]
        if self.detach_target:
            target = target.detach()

        x = x[:, :-1]
        if self.detach_x:
            x = x.detach()

        _, loss = self.forward(x, targets=target)

        return loss

    def ar_predict(self, x: torch.Tensor) -> torch.Tensor:
        # make ar prediction using teacher forcing
        # x: (b, n, n_ind)
        x = x[:, :-1] # (b, n-1, n_ind)
        pred, _ = self.forward(x) # (b, n-1, n_ind)
        full_pred = torch.cat([x[:, :1], pred], dim=1) # (b, n, n_ind)

        if self.l2_normalized:
            full_pred = F.normalize(full_pred, p=2, dim=-1)
        return full_pred
    


#################################################################################
#                                 GPTC Configs                                  #
#################################################################################   

def GPTC_L(**kwargs):
    return GPTC(GPTCConfig(n_layer=24, n_head=16, n_embd=1024, **kwargs)) # 316.4M?

def GPTC_B(**kwargs):
    return GPTC(GPTCConfig(n_layer=12, n_head=12, n_embd=768, **kwargs)) # 85.9M

def GPTC_M(**kwargs):
    return GPTC(GPTCConfig(n_layer=12, n_head=8, n_embd=512, **kwargs)) # 38.4M

def GPTC_S(**kwargs):
    return GPTC(GPTCConfig(n_layer=12, n_head=6, n_embd=384, **kwargs)) # 21.7M

def GPTC_XS(**kwargs):
    return GPTC(GPTCConfig(n_layer=6, n_head=6, n_embd=384, **kwargs)) # 11.1M

def GPTC_XXS(**kwargs):
    return GPTC(GPTCConfig(n_layer=6, n_head=4, n_embd=256, **kwargs)) # 5.0M

GPTC_models = {
    'gptc-L': GPTC_L,
    'gptc-B': GPTC_B,
    'gptc-M': GPTC_M,
    'gptc-S': GPTC_S,
    'gptc-XS': GPTC_XS,
    'gptc-XXS': GPTC_XXS
}

'''
# Count number of parameters

import torch
import models
from utils import compute_num_params
compute_num_params(models.make({'name': 'gptc-S', 'args': {}}))

'''


# ------------------------------------------------------------------------------
# tokenizer/tokenizer_image/vq/vq_vit_model.py
# ------------------------------------------------------------------------------
# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
#   REPA: https://github.com/sihyun-yu/REPA
#   DETR: https://github.com/facebookresearch/detr








def set_requires_grad(requires_grad, *models):
    """
    Sets requires_grad true or false for all parameters within the
    models passed.
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)


@dataclass
class VQVitModelPlusArgs:

    # for quantization
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0

    # tricks for lfq. deprecated
    use_lfq: bool = False
    bernoulli_sample: bool = False
    eval_deterministic: bool = False

    # SimVQ trick, deprecated
    simvq: bool = False
    codebook_transform: str = None
    freeze_codebook: bool = False
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    model_size: str = 'small'
    encoder_size: str = None 
    decoder_size: str = None
    num_latent_tokens: int = 256
    z_channels: int = 256   # the dimension of the intermediate downsample towards codebook dimension
    dropout_p: float = 0.0

    # use rope for the decoder Q-former attention. 
    use_rope: bool = False
    use_qk_norm: bool = False

    # TODO: remove option. flash attention is automatically used when calling scaled_dot_product_attention
    use_flash_attn: bool = False

    # the setting for initializing the 1d queries for the 2dto1d encoder
    multi_level_query_init: bool = False
    learnable_1d_query_init: bool = False
    rope_1d: bool = False

    # the initialization for the 2d queries. the "level" corresponds to the 
    # "level" division for "multi_level_query_init". It assumes 1d tokens have levels
    # all false means simply using the global average of the 1d tokens to initialize
    # the 2d queries.
    last_level_2d_query_init: bool = False
    multi_level_2d_query_init: bool = False
    learnable_2d_query_init: bool = False

    # tricks for the CNN 2d decoder
    adaptive_gn: bool = False
    d2s_up: bool = False
    res_up_down_sample: bool = False
    downsample_match_channel: bool = False
    upsample_match_channel: bool = False
    res_codebook_updown_sample: bool = False
    downsample_improve: bool = False
    # whether to use attention in the 2d encoder or decoder
    # suggested not to. May be unstable and slower
    use_attn: bool = True

    # rope 2d only supports the 1dto2d decoder queries (since Q-former)
    rope_2d: bool = False

    # the rotation trick for quantizer. The influence is limited
    rot: bool = False

    # for stochastic quantization. Closed by default
    stochastic: bool = False
    stochastic_temperature: float = 0.03

    # distillation setting
    distill_depth: int = None
    # whether to distill from encoder. Not tested yet.
    # (to be deleted)
    encoder_2d_distill: bool = False

    # for semantic distillation regularization
    # the default 768 is for dino-v2 base
    out_inner_dim: int = 768

    fea_rec_loss_type: str = "cosine"
    fea_rec_loss_weight: float = 1.0

    # for gptc model, which tries to utilize AR prior for 
    # training tokenizers. The effect is limited and this feature
    # is deprecated.
    # for ar prior model
    with_prior_model: bool = False
    prior_model_config: dict = None




class VQVitModelPlus(nn.Module):
    def __init__(self, config: VQVitModelPlusArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
                        ch_mult=config.encoder_ch_mult, 
                        z_channels=config.z_channels, 
                        dropout=config.dropout_p, 
                        use_attn=config.use_attn,
                        res_down_sample=config.res_up_down_sample,
                        downsample_match_channel=config.downsample_match_channel,
                        )

        if config.encoder_2d_distill:
            # setting is from REPA
            self.distill_mlp = nn.Sequential(
                    nn.Linear(config.z_channels, config.z_channels * 4),
                    nn.SiLU(),
                    nn.Linear(config.z_channels * 4, config.z_channels * 4),
                    nn.SiLU(),
                    nn.Linear(config.z_channels * 4, config.out_inner_dim),
                    )


        # set the size of the transformer encoder/decoder size
        encoder_size = config.model_size if config.encoder_size is None else config.encoder_size
        decoder_size = config.model_size if config.decoder_size is None else config.decoder_size
       
        # when encoder size or decoder size is given, model size should be none
        if config.encoder_size is not None or config.decoder_size is not None:
            assert config.model_size is None
        
        if config.encoder_2d_distill:
            assert config.distill_depth is None



        self.s2to1encoder = ViTEncoder(model_size=encoder_size, num_latent_tokens=config.num_latent_tokens, 
                               token_size=config.z_channels, dropout=config.dropout_p, 
                               patch_size=2**(len(config.encoder_ch_mult) - 1),
                               multi_level_query_init=config.multi_level_query_init,
                               learnable_1d_query_init=config.learnable_1d_query_init,
                               rope_1d=config.rope_1d,
                               downsample_improve=config.downsample_improve,
                               use_qk_norm=config.use_qk_norm,
                               use_flash_attn=config.use_flash_attn,
                               )

        if config.use_rope:
            # V2 model is specifically designed for rope2d
            self.s1to2decoder = ViTDecoder_V2(model_size=decoder_size, num_latent_tokens=config.num_latent_tokens, 
                                token_size=config.z_channels, dropout=config.dropout_p,
                                patch_size=2**(len(config.decoder_ch_mult) - 1),
                                last_level_2d_query_init=config.last_level_2d_query_init,
                                multi_level_2d_query_init=config.multi_level_2d_query_init,
                                learnable_2d_query_init=config.learnable_2d_query_init,
                                rope_2d=True,
                                use_qk_norm=config.use_qk_norm,
                                use_flash_attn=config.use_flash_attn,
                                )
        
        else:
            self.s1to2decoder = ViTDecoder(model_size=decoder_size, num_latent_tokens=config.num_latent_tokens, 
                                token_size=config.z_channels, dropout=config.dropout_p,
                                patch_size=2**(len(config.decoder_ch_mult) - 1),
                                last_level_2d_query_init=config.last_level_2d_query_init,
                                multi_level_2d_query_init=config.multi_level_2d_query_init,
                                learnable_2d_query_init=config.learnable_2d_query_init,
                                out_inner_feat=config.distill_depth is not None,
                                out_inner_depth=config.distill_depth,
                                out_inner_dim=config.out_inner_dim,
                                use_qk_norm=config.use_qk_norm,
                                use_flash_attn=config.use_flash_attn,
                                )

        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, 
                               z_channels=config.z_channels, 
                               dropout=config.dropout_p,
                               adaptive_gn=config.adaptive_gn,
                               d2s_up=config.d2s_up,
                               use_attn=config.use_attn,
                               res_up_sample=config.res_up_down_sample,
                               upsample_match_channel=config.upsample_match_channel,
                               )

        self.num_latent_tokens = config.num_latent_tokens
        # scale = self.s2to1encoder.width ** -0.5
        # self.latent_tokens = nn.Parameter(
        #     scale * torch.randn(self.num_latent_tokens, self.s2to1encoder.width))

        # the weight initialization seems to have ignored post_quant_conv and pre_quant_conv
        # and it potentially affects the prior model(deprecated) training
        # but weight initialization is also ignored(?) in llamagen implementation
        # currently this setting can just work.
        self.apply(self._init_weights)

        if self.config.with_prior_model:
            if self.config.use_lfq:
                raise NotImplementedError("LFQ is not implemented yet")
            else:
                self.quantize = VectorQuantizerWithPM(
                                                config.codebook_size, config.codebook_embed_dim, 
                                                config.commit_loss_beta, config.entropy_loss_ratio,
                                                config.codebook_l2_norm, config.codebook_show_usage,
                                                rot=config.rot, stochastic=config.stochastic,
                                                stochastic_temperature=config.stochastic_temperature,
                                                prior_model_config=config.prior_model_config,
                                                simvq=config.simvq,
                                                codebook_transform=config.codebook_transform,
                                                freeze_codebook=config.freeze_codebook,
                                        )

            
        else:
            if self.config.use_lfq:
                self.quantize = LFQ(
                    dim=config.codebook_embed_dim,
                    beta=config.commit_loss_beta,
                    entropy_loss_ratio=config.entropy_loss_ratio,
                    n_e=config.codebook_size,
                )
            else:
                self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                                config.commit_loss_beta, config.entropy_loss_ratio,
                                                config.codebook_l2_norm, config.codebook_show_usage,
                                                rot=config.rot, stochastic=config.stochastic,
                                                stochastic_temperature=config.stochastic_temperature,
                                                eval_deterministic=config.eval_deterministic,
                                                simvq=config.simvq,
                                                codebook_transform=config.codebook_transform,
                                                freeze_codebook=config.freeze_codebook,
                                                )

        if self.config.res_codebook_updown_sample:

            if self.config.downsample_improve:
                self.quant_conv = ChannelDownsampleResidual(self.s2to1encoder.width, config.codebook_embed_dim)
            else:
                self.quant_conv = ChannelDownsampleResidual(config.z_channels, config.codebook_embed_dim)

            self.post_quant_conv = ChannelUpsampleResidual(config.codebook_embed_dim, self.config.z_channels)
        else:
            self.quant_conv = nn.Conv2d(self.config.z_channels, config.codebook_embed_dim, 1)
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, self.config.z_channels, 1)
        
        self.freeze_but_2d_decoder_flag = False

        def nan_hook(self, inp, output):
            if not isinstance(output, torch.Tensor):
                return
            if torch.isnan(output).any():
                print(f"NaN detected in {self}")
                raise RuntimeError("NaN detected")

        # for name, module in self.named_modules():
        #     module.register_forward_hook(nan_hook)

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        wrap_modules = []
        # Add encoder layers
        for layer in self.s2to1encoder.transformer:
            wrap_modules.append(layer)
        # Add decoder layers
        for layer in self.s1to2decoder.transformer:
            wrap_modules.append(layer)

        return wrap_modules
        

    # def eval(self):
        # delete unused modules for inferencing
        # - semantic distillation mlp
        # - ar prior model

        # if self.config.encoder_2d_distill:
        #     del self.distill_mlp
        
        # if self.config.distill_depth is not None:
        #     del self.s1to2decoder.distill_mlp
        
        # if self.config.with_prior_model:
        #     del self.quantize.prior_model
        # super().eval()
    

    def freeze_but_2d_decoder(self):
        """deprecated"""
        for param in self.parameters():
            param.requires_grad = False

        set_requires_grad(True, self.decoder)
        self.freeze_but_2d_decoder_flag = True
    
    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x, 
               return_code=True, 
               return_feat=False, 
               return_cont_feat=False,      # return the feature before quantization
               return_fix_dim_feat=False,   # the fix dimension is the conv out before to codebook dim
               num_en_q_level=None, 
               causal_type=None,        # deprecated
               random_mix_reg=False,    # deprecated
               replace_ratio=None,      # deprecated
               global_step=None,        # prior model related, deprecated
               max_steps=None,          # prior model related, deprecated
               ):
        # causal_type = causal_type if causal_type is not None else self.config.causal_type
        if return_feat:
            assert (not return_code) and (not return_fix_dim_feat)
            s = self.encoder(x)
            h = self.s2to1encoder(
                s, num_q_level=num_en_q_level, 
                causal_type=causal_type, 
                return_feat=True)
            # return the feature of exactly the same width as the vit encoder
            return h, None, None
        
        if return_cont_feat:
            assert (not return_code) and (not return_fix_dim_feat)
            s = self.encoder(x)
            h = self.s2to1encoder(
                s, num_q_level=num_en_q_level,
                causal_type=causal_type,
                return_feat=False)
            # return the feature before quantization
            h = self.quant_conv(h)
            return h, None, None
        
        if return_fix_dim_feat:
            s = self.encoder(x)
            h = self.s2to1encoder(
                s, num_q_level=num_en_q_level,
                causal_type=causal_type,
                return_feat=False)
            # return the feature of the fix width (e.g. 256) before further downsampled to codebook dim
            return h, None, None


        s = self.encoder(x)
        h = self.s2to1encoder(s, num_q_level=num_en_q_level, causal_type=causal_type)
        # print("s shape:", s.shape)

        h = self.quant_conv(h)
        if self.training and self.config.with_prior_model:
            quant, emb_loss, info = self.quantize(h, random_replace=random_mix_reg, replace_ratio=replace_ratio,
                                                   global_step=global_step, max_steps=max_steps)
        else:
            quant, emb_loss, info = self.quantize(h, random_replace=random_mix_reg, replace_ratio=replace_ratio)

        if return_code:
            return quant, emb_loss, info
       
        return quant, emb_loss, s

    def decode(
            self, quant, 
            ret_inner_feat=False, # the feature passed through a MLP for alignment loss
            return_feat=False,    # the feature for linear probe
            ):
        quant = self.post_quant_conv(quant)
        if ret_inner_feat:
            rec_spatial, inner_feat = self.s1to2decoder(quant, ret_inner_feat=True)
            pixel_dec = self.decoder(rec_spatial)
            return pixel_dec, rec_spatial, inner_feat
        elif return_feat:
            # specifically for linear probe
            _, inner_feat = self.s1to2decoder(quant, return_feat=True)
            # pixel_dec = self.decoder(rec_spatial)
            return None, None, inner_feat
        else:
            rec_spatial = self.s1to2decoder(quant)
            pixel_dec = self.decoder(rec_spatial)
            return pixel_dec, rec_spatial

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec, rec_spatial = self.decode(quant_b)
        return dec

    def forward(
            self, 
            input, 
            num_en_q_level=None, 
            causal_type=None, 
            rec_loss=True, 
            ret_inner_feat=False,
            random_mix_reg=False,
            replace_ratio=None,
            global_step=None,
            max_steps=None,
            ):
        quant, diff, spatial = self.encode(
                                    input, 
                                    return_code=False, 
                                    num_en_q_level=num_en_q_level, 
                                    causal_type=causal_type,
                                    random_mix_reg=random_mix_reg,
                                    replace_ratio=replace_ratio,
                                    global_step=global_step,
                                    max_steps=max_steps
                                    )
        if ret_inner_feat:
            if self.config.encoder_2d_distill:
                inner_feat = rearrange(spatial, 'b c h w -> b (h w) c')
                inner_feat = self.distill_mlp(inner_feat)
                dec, rec_spatial = self.decode(quant)
            else:
                dec, rec_spatial, inner_feat = self.decode(quant, ret_inner_feat=True)
        else:
            dec, rec_spatial = self.decode(quant)

        if self.training:
            if rec_loss:
                if self.config.fea_rec_loss_type == "cosine":
                    fea_rec_loss = self.config.fea_rec_loss_weight * compute_cosinesim_loss(spatial.detach(), rec_spatial, 1)
                elif self.config.fea_rec_loss_type == "mse":
                    fea_rec_loss = self.config.fea_rec_loss_weight * F.mse_loss(spatial.detach(), rec_spatial)
            else:
                fea_rec_loss = 0

        if self.training:
            if rec_loss:
                dir_dec = self.decoder(spatial)
            else:
                dir_dec = None
            
            if ret_inner_feat:
                return [dec, dir_dec], [diff, fea_rec_loss], inner_feat
            return [dec, dir_dec], [diff, fea_rec_loss]

        return dec, diff



@dataclass
class VQVitModel2DPlusArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    model_size: str = 'small'
    num_latent_tokens: int = 256
    encoder_size: str = None 
    decoder_size: str = None
    transformer_layer_type: str = "TransformerDecoderLayer"
    z_channels: int = 256
    dropout_p: float = 0.0

    adaptive_gn: bool = False
    d2s_up: bool = False

    rot: bool = False
    distill_depth: int = None

    encoder_2d_distill: bool = False

    # for semantic distillation regularization
    # the default 768 is for dino-v2 base
    out_inner_dim: int = 768

    fea_rec_loss_type: str = "cosine"
    fea_rec_loss_weight: float = 1.0
    use_attn: bool = True


class VQVitModel2DPlus(nn.Module):
    def __init__(self, config: VQVitModelPlusArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
                        ch_mult=config.encoder_ch_mult, 
                        z_channels=config.z_channels, 
                        dropout=config.dropout_p, 
                        use_attn=config.use_attn,
                        )

        if config.encoder_2d_distill:
            self.distill_mlp = nn.Sequential(
                    nn.Linear(config.z_channels, config.z_channels * 4),
                    nn.SiLU(),
                    nn.Linear(config.z_channels * 4, config.z_channels * 4),
                    nn.SiLU(),
                    nn.Linear(config.z_channels * 4, config.out_inner_dim),
                    )


        if config.encoder_size is not None:
            encoder_size = config.encoder_size
        else:
            encoder_size = config.model_size
        
        if config.decoder_size is not None:
            decoder_size = config.decoder_size
        else:
            decoder_size = config.model_size
        
        # when encoder size or decoder size is given, model size should be none
        if config.encoder_size is not None or config.decoder_size is not None:
            assert config.model_size is None
        
        if config.encoder_2d_distill:
            assert config.distill_depth is None



        self.s2dencoder = ViTEncoder2D(
            model_size=encoder_size, 
            token_size=config.z_channels, dropout=config.dropout_p, 
            patch_size=2**(len(config.encoder_ch_mult) - 1),
            transformer_layer_type=config.transformer_layer_type,
            )

        self.s2ddecoder = ViTDecoder2D(
            model_size=decoder_size,
            token_size=config.z_channels, dropout=config.dropout_p,
            patch_size=2**(len(config.decoder_ch_mult) - 1),
            out_inner_feat=config.distill_depth is not None,
            out_inner_depth=config.distill_depth,
            out_inner_dim=config.out_inner_dim,
            transformer_layer_type=config.transformer_layer_type,
            )

        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, 
            z_channels=config.z_channels, 
            dropout=config.dropout_p,
            adaptive_gn=config.adaptive_gn,
            d2s_up=config.d2s_up,
            use_attn=config.use_attn
            )

        self.apply(self._init_weights)

        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage,
                                        rot=config.rot
                                        )
        self.quant_conv = nn.Conv2d(self.config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, self.config.z_channels, 1)

        def nan_hook(self, inp, output):
            if not isinstance(output, torch.Tensor):
                return
            if torch.isnan(output).any():
                print(f"NaN detected in {self}")
                raise RuntimeError("NaN detected")

        for name, module in self.named_modules():
            module.register_forward_hook(nan_hook)
    
    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x, 
               return_code=True, 
               return_feat=False, 
               random_mix_reg=False,
               replace_ratio=0.1,
               **kwargs
               ):
        # causal_type = causal_type if causal_type is not None else self.config.causal_type
        if return_feat:
            s = self.encoder(x)
            h = self.s2dencoder(s, return_feat=True)
            # return the feature before quantization
            return h, None, None

        s = self.encoder(x)
        h = self.s2dencoder(s)
        # print("s shape:", s.shape)

        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, random_replace=random_mix_reg, replace_ratio=replace_ratio)

        if return_code:
            return quant, emb_loss, info
       
        return quant, emb_loss, s

    def decode(self, quant, ret_inner_feat=False, return_feat=False):
        quant = self.post_quant_conv(quant)
        if ret_inner_feat:
            rec_spatial, inner_feat = self.s2ddecoder(quant, ret_inner_feat=True)
            pixel_dec = self.decoder(rec_spatial)
            return pixel_dec, rec_spatial, inner_feat
        elif return_feat:
            # specifically for linear probe or visualization (don not go through mlp)
            _, feat = self.s2ddecoder(quant, return_feat=True)
            # pixel_dec = self.decoder(rec_spatial)
            return _, feat
        else:
            rec_spatial = self.s2ddecoder(quant)
            pixel_dec = self.decoder(rec_spatial)
            return pixel_dec, rec_spatial

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec, rec_spatial = self.decode(quant_b)
        return dec

    def forward(
            self, 
            input, 
            num_en_q_level=None, 
            causal_type=None, 
            rec_loss=True, 
            ret_inner_feat=False,
            random_mix_reg=False,
            replace_ratio=None,
            global_step=None,
            max_steps=None,
            ):
        quant, diff, spatial = self.encode(
                                    input, 
                                    return_code=False, 
                                    random_mix_reg=random_mix_reg,
                                    replace_ratio=replace_ratio
                                    )
        if ret_inner_feat:
            if self.config.encoder_2d_distill:
                inner_feat = rearrange(spatial, 'b c h w -> b (h w) c')
                inner_feat = self.distill_mlp(inner_feat)
                dec, rec_spatial = self.decode(quant)
            else:
                dec, rec_spatial, inner_feat = self.decode(quant, ret_inner_feat=True)
        else:
            dec, rec_spatial = self.decode(quant)
        # if torch.isnan(dec).any():
        #     print("nan in dec")
        # if torch.isnan(rec_spatial).any():
        #     print("nan in rec_spatial")

        if self.training:
            if rec_loss:
                if self.config.fea_rec_loss_type == "cosine":
                    fea_rec_loss = self.config.fea_rec_loss_weight * compute_cosinesim_loss(spatial.detach(), rec_spatial, 1)
                elif self.config.fea_rec_loss_type == "mse":
                    fea_rec_loss = self.config.fea_rec_loss_weight * F.mse_loss(spatial.detach(), rec_spatial)
            else:
                fea_rec_loss = 0

        if self.training:
            if rec_loss:
                dir_dec = self.decoder(spatial)
            else:
                dir_dec = None
            
            if ret_inner_feat:
                return [dec, dir_dec], [diff, fea_rec_loss], inner_feat
            return [dec, dir_dec], [diff, fea_rec_loss]

        return dec, diff




class VectorQuantizer(nn.Module):
    def __init__(
            self, 
            n_e, 
            e_dim, 
            beta, 
            entropy_loss_ratio, 
            l2_norm, 
            show_usage, 
            rot=False,
            stochastic=False,
            stochastic_temperature=1.0,
            eval_deterministic=False,
            simvq=False,
            codebook_transform=None,
            freeze_codebook=False,
            ):
        """
        Args:
            n_e: the size of the codebook
            e_dim: the dimension of the codebook vectors
            beta: the commitment loss weight
            entropy_loss_ratio: the ratio of the entropy loss to the commitment loss
            l2_norm: whether to normalize the codebook vectors
            show_usage: whether to show the usage of the codebook vectors
            rot: whether to use rotation trick
            stochastic: whether to use stochastic quantization
            stochastic_temperature: the temperature of the stochastic quantization
            eval_deterministic: whether to use deterministic quantization in evaluation mode
            simvq: whether to use simvq https://arxiv.org/abs/2411.02038
            codebook_transform: the transform to apply to the codebook vectors,
                choices from [ None, "linear", "mlp"]
            freeze_codebook: whether to freeze the codebook vectors
        """

        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.rot = rot
        self.stochastic = stochastic
        self.eval_deterministic = eval_deterministic

        self.simvq = simvq
        self.codebook_transform = codebook_transform
        self.freeze_codebook = freeze_codebook


        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

        if self.stochastic:
            if stochastic_temperature > 0: # fixed temperature
                self.stochastic_temperature_inv = 1 / stochastic_temperature
            else: # set stochastic_temperature < 0 to use learnable temperature
                self.stochastic_temperature_inv = nn.Parameter(torch.tensor(10.0))
        
        if self.simvq:
            if codebook_transform == "linear":
                codebook_transform = nn.Linear(self.e_dim, self.e_dim, bias=False)
            elif codebook_transform == "mlp":
                codebook_transform = nn.Sequential(
                    nn.Linear(self.e_dim, self.e_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.e_dim * 4, self.e_dim),
                )
            else:
                raise ValueError("codebook_transform: {} Not Acceptable".format(codebook_transform))
            self.codebook_transform = codebook_transform

            if self.freeze_codebook:
                self.embedding.weight.requires_grad = False

    def get_emb(self):
        if self.simvq:
            return self.codebook_transform(self.embedding.weight)
        else:
            return self.embedding.weight

    @staticmethod
    def get_very_efficient_rotation(u, q, e):
        # from https://github.com/cfifty/rotation_trick/blob/main/src/models/vq_vae.py
        w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()
        e = e - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1)) + 2 * torch.bmm(
        torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
        return e

    
    def forward(self, z, random_replace=False, replace_ratio=0.1):
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.get_emb(), p=2, dim=-1)
        else:
            embedding = self.get_emb()

        if self.stochastic:
            # sample the softmaxed cosine similarity
            # reference: LARP
            assert self.l2_norm, "Stochastic sampling requires l2 normalization"
            cos_sim = torch.einsum("bd,nd->bn", z_flattened, embedding)
            probs = F.softmax(cos_sim * self.stochastic_temperature_inv, dim=-1)
            if self.eval_deterministic and not self.training:
                min_encoding_indices = torch.argmax(probs, dim=-1)

            else:
                min_encoding_indices = torch.multinomial(probs, 1)
                min_encoding_indices = min_encoding_indices.squeeze(-1)
        else:
            # look up by l2 distance, argmin
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(embedding**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

            min_encoding_indices = torch.argmin(d, dim=1)   # (b*h*w)

        z_q = embedding[min_encoding_indices].view(z.shape)

        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2) 
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
            if self.entropy_loss_ratio > 0:
                entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)
            else:
                entropy_loss = 0

        b, h, w, c = z.shape
        if self.rot:
            # adapted from https://github.com/cfifty/rotation_trick/blob/main/src/models/vq_vae.py
            b, h, w, c = z.shape
            z = z / torch.norm(z, dim=-1, keepdim=True)
            # assert self.l2_norm, "Rot requires l2 normalization"
            z = rearrange(z, 'b h w c-> (b h w) c')
            z_q= rearrange(z_q, 'b h w c -> (b h w) c')
            pre_norm_q = self.get_very_efficient_rotation(z / (torch.norm(z, dim=1, keepdim=True) + 1e-6),
                                                            z_q / (torch.norm(z_q, dim=1, keepdim=True) + 1e-6),
                                                            z.unsqueeze(1)).squeeze()
            z_q = pre_norm_q * (
                    torch.norm(z_q, dim=1, keepdim=True) / (torch.norm(z, dim=1, keepdim=True) + 1e-6)).detach()
            z_q = rearrange(z_q, '(b h w) c -> b h w c', b=b, h=h, w=w)
        else:
            # preserve gradients
            z_q = z + (z_q - z).detach()

        if random_replace and self.training:
            # randomly replace the quantized vectors with the continuous input
            z = rearrange(z, '(b h w) c -> b h w c', b=b, h=h, w=w)
            mask = torch.bernoulli(torch.full(z.shape[:-1], replace_ratio)).unsqueeze(-1).to(z.device)  # replace_ratio chance of replacement
            z_q = torch.where(mask.bool(), z, z_q)

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, [vq_loss, commit_loss, entropy_loss, codebook_usage], (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.get_emb(), p=2, dim=-1)
        else:
            embedding = self.get_emb()

        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q



class VectorQuantizerWithPM(nn.Module):
    def __init__(
            self, 
            n_e, 
            e_dim, 
            beta, 
            entropy_loss_ratio, 
            l2_norm, 
            show_usage, 
            rot=False,
            stochastic=False,
            stochastic_temperature=1.0,
            eval_deterministic=False,
            simvq=False,
            codebook_transform=None,
            freeze_codebook=False,
            prior_model_config=None
            ):
        """
        Args:
            n_e: the size of the codebook
            e_dim: the dimension of the codebook vectors
            beta: the commitment loss weight
            entropy_loss_ratio: the ratio of the entropy loss to the commitment loss
            l2_norm: whether to normalize the codebook vectors
            show_usage: whether to show the usage of the codebook vectors
            rot: whether to use rotation trick
            stochastic: whether to use stochastic quantization
            stochastic_temperature: the temperature of the stochastic quantization
            eval_deterministic: whether to use deterministic quantization in evaluation mode
            simvq: whether to use simvq https://arxiv.org/abs/2411.02038
            codebook_transform: the transform to apply to the codebook vectors,
                choices from [ None, "linear", "mlp"]
            freeze_codebook: whether to freeze the codebook vectors
            prior_model_config: the config for the prior model

        - prior ar model for ntp regularization
            - returns the prior model loss along with other codebook loss
        """
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.rot = rot
        self.stochastic = stochastic
        self.prior_model_config = prior_model_config
        self.eval_deterministic = eval_deterministic

        self.simvq = simvq
        self.codebook_transform = codebook_transform
        self.freeze_codebook = freeze_codebook

        # prior model training config 
        if prior_model_config is not None:
            self.prior_n_rounds = prior_model_config["train_args"]["n_rounds"]
            self.prior_no_grad_before_last_round = prior_model_config["train_args"]["no_grad_before_last_round"]
            self.prior_avg_loss_over_rounds = prior_model_config["train_args"]["avg_loss_over_rounds"]
            self.use_mix_ss = prior_model_config["train_args"]["use_mix_ss"]
            self.mix_ss_max_ratio = prior_model_config["train_args"]["mix_ss_max_ratio"]
            self.mix_ss_peak_steps_ratio = prior_model_config["train_args"]["mix_ss_peak_steps_ratio"]
            self.prior_latent_ce_temperature = prior_model_config["train_args"].get("latent_ce_temperature", 1.0)
 

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

        if self.stochastic:
            if stochastic_temperature > 0: # fixed temperature
                self.stochastic_temperature_inv = 1 / stochastic_temperature
            else: # set stochastic_temperature < 0 to use learnable temperature
                self.stochastic_temperature_inv = nn.Parameter(torch.tensor(10.0))

        if prior_model_config is None:
            self.prior_model = None
        else:
            prior_model_additional_args = {
                'n_ind': self.e_dim, 
                'n_classes': self.n_e
            }

            self.ar_prior_loss_weight = prior_model_config["train_args"].get('prior_loss_weight', 0.06)
            if prior_model_config["train_args"].get('no_dropout', False):
                prior_model_additional_args['embd_pdrop'] = 0.0
                prior_model_additional_args['resid_pdrop'] = 0.0
                prior_model_additional_args['attn_pdrop'] = 0.0
                print(f"Warning: prior_loss is using no dropout")
            
            # initialize
            self.prior_model = GPTC_models[self.prior_model_config['name']](
                    **prior_model_config['init_args'], 
                    **prior_model_additional_args
                )

        if self.simvq:
            if codebook_transform == "linear":
                codebook_transform = nn.Linear(self.e_dim, self.e_dim, bias=False)
            elif codebook_transform == "mlp":
                codebook_transform = nn.Sequential(
                    nn.Linear(self.e_dim, self.e_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.e_dim * 4, self.e_dim),
                )
            else:
                raise ValueError("codebook_transform: {} Not Acceptable".format(codebook_transform))
            self.codebook_transform = codebook_transform

            if self.freeze_codebook:
                self.embedding.weight.requires_grad = False

        
    def get_emb(self):
        if self.simvq:
            return self.codebook_transform(self.embedding.weight)
        else:
            return self.embedding.weight


    @staticmethod
    def get_very_efficient_rotation(u, q, e):
        # from https://github.com/cfifty/rotation_trick/blob/main/src/models/vq_vae.py
        w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()
        e = e - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1)) + 2 * torch.bmm(
        torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
        return e

    def logits_to_token_embedding_with_ss(
            self, 
            logits, 
            ar_input_staring_from_idx_1, 
            global_step,
            max_steps,
            mask=None):
        """
        adapted from https://github.com/hywang66/LARP/
        """
        # logits: (b, n - 1, codebook_size), sequence index from 1 to n-1 (inclusive)
        # ar_input_staring_from_idx_1: (b, n - 1, d=16), requires_grad=True
        if mask is None:
            b, n_minus_1, _ = logits.size()
            if self.use_mix_ss:
                ss_ratio = (global_step / (max_steps * self.mix_ss_peak_steps_ratio )) * self.mix_ss_max_ratio
                ss_ratio = min(ss_ratio, self.mix_ss_max_ratio)
            else:
                ss_ratio = 1.0

            mask = torch.rand(b, n_minus_1, 1, device=logits.device) < ss_ratio
            mask = mask.expand(-1, -1, self.e_dim) # (b, n - 1, d=16)

        with torch.autocast(device_type='cuda', enabled=False):
            logits = logits.float()
            probs = F.softmax(logits, dim=-1) # (b, n - 1, codebook_size)
            indices = torch.multinomial(probs.view(-1, self.n_e), 1).view(*probs.size()[:-1]) # (b, n - 1)
        token_embedding = F.embedding(indices, self.get_emb()) # (b, n - 1, d=16)
        token_embedding = torch.where(mask, token_embedding, ar_input_staring_from_idx_1)

        return token_embedding

    def calculate_logits_and_ar_pred_cont(self, prior_model_output):
        ar_pred_cont = prior_model_output # (b, n, d=16)
        # the prior_model_output and the embedding should have been normalized (-1 dim)
        logits = F.linear(prior_model_output, self.get_emb())[:, 1:]
        logits = logits.mul_(1 / self.prior_latent_ce_temperature)
        logits = logits.contiguous() # (b, n - 1, codebook_size)
        return logits, ar_pred_cont


    def prior_ar_predict_n_rounds_ss(
            self, 
            ar_input, 
            global_step,
            max_steps,
        ):
        """
        adapted from https://github.com/hywang66/LARP/
        """
        prior_model = self.prior_model
        n_rounds = self.prior_n_rounds
        no_grad_before_last_round = self.prior_no_grad_before_last_round

        b, n, _ = ar_input.size()
        n_minus_1 = n - 1
        if self.use_mix_ss:
            peak_steps_ratio = torch.tensor(self.mix_ss_peak_steps_ratio, dtype=torch.float32)
            max_ratio = torch.tensor(self.mix_ss_max_ratio, dtype=torch.float32)

            ss_ratio = (global_step / (max_steps * peak_steps_ratio)) * max_ratio
            ss_ratio = torch.min(ss_ratio, max_ratio)
        else:
            ss_ratio = torch.tensor(1.0, dtype=torch.float32)

        mask_ss = torch.rand(b, n_minus_1, 1, device=ar_input.device) < ss_ratio
        mask_ss = mask_ss.expand(-1, -1, self.e_dim) # (b, n - 1, d=16)

        logits_all_rounds = []
        next_ar_input = ar_input # (b, n, d=16)
        for i in range(n_rounds):
            if no_grad_before_last_round and i < n_rounds - 1:
                # we can not use "with torch.no_grad()" here due to a pytorch's bug!
                # https://github.com/pytorch/pytorch/issues/112583
                prior_model.requires_grad_(False)
                prior_model_output = prior_model.ar_predict(next_ar_input.detach()) # (b, n - 1, codebook_size)
                logits, ar_pred_cont = self.calculate_logits_and_ar_pred_cont(prior_model_output)
                prior_model.requires_grad_(True)
            else:
                prior_model_output = prior_model.ar_predict(next_ar_input) # (b, n, d=16)(1 orig + n - 1 pred)
                logits, ar_pred_cont = self.calculate_logits_and_ar_pred_cont(prior_model_output)   # (b, n - 1, codebook_size)
                logits_all_rounds.append(logits)


            if i < n_rounds - 1:
                token_embedding = self.logits_to_token_embedding_with_ss(
                                            logits, 
                                            ar_input[:, 1:], 
                                            global_step=global_step,
                                            max_steps=max_steps,
                                            mask=mask_ss) # (b, n - 1, d=16)
                next_ar_input = torch.cat([ar_input[:, :1], token_embedding], dim=1) # (b, n, d=16)

        if self.prior_avg_loss_over_rounds:
            logits_all_rounds = torch.stack(logits_all_rounds, dim=0) # (n_rounds, b, n - 1, codebook_size)

        else:
            logits_all_rounds = torch.stack([logits_all_rounds[-1]], dim=0) # (1, b, n - 1, codebook_size)

        return logits_all_rounds, ar_pred_cont, next_ar_input # here the next_ar_input is actually the last round's ar_input

    def calculate_prior_loss_with_pred(
            self, 
            encode_output, 
            indices,
            global_step, 
            max_steps,
            return_sampled_indices=False,
            sample_temperature=1.0,
        ):
        """
        adapted from https://github.com/hywang66/LARP/
        """
        B = encode_output.size(0)
        ar_input = encode_output # (b, n, d) normalized
        labels = indices[:, 1:].contiguous() # (b, n - 1)
        logits_all_rounds, ar_pred_cont, regularized_z_ss = self.prior_ar_predict_n_rounds_ss(
                                                                    ar_input, 
                                                                    global_step=global_step, 
                                                                    max_steps=max_steps,
                                                                ) # regularized_z_ss: (b, n, d=16)
        labels_all_rounds = labels.unsqueeze(0).expand(logits_all_rounds.size(0), -1, -1).contiguous() # (n_rounds or 1, b, n - 1)
        
        loss_latent_ce = F.cross_entropy(logits_all_rounds.view(-1, self.n_e), labels_all_rounds.view(-1))
        # return_dict['loss_latent_ce'] = loss_latent_ce
        # topk_accuracies = utils.calculate_topk_accuracy(logits_all_rounds[0], labels, topk=(1, 5), prepend='prior_')
        # return_dict.update(topk_accuracies)

        if return_sampled_indices:
            # sample the indices from the last round prediction because it is closer
            # to the downstream gpt prediction error pattern
            sampled_indices = torch.multinomial(F.softmax(logits_all_rounds[-1] / sample_temperature, dim=-1), 1).squeeze(-1)


        return loss_latent_ce 


    
    def forward(
            self, 
            z, 
            max_steps=None, # for pm training
            global_step=None,
            random_replace=False, 
            replace_ratio=0.1,
            ):
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        b, h, w, c = z.shape
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.get_emb(), p=2, dim=-1)
        else:
            embedding = self.get_emb()


        if self.stochastic:
            # sample the softmaxed cosine similarity
            # reference: LARP
            assert self.l2_norm, "Stochastic sampling requires l2 normalization"
            cos_sim = torch.einsum("bd,nd->bn", z_flattened, embedding)
            probs = F.softmax(cos_sim * self.stochastic_temperature_inv, dim=-1)
            if self.eval_deterministic and not self.training:
                min_encoding_indices = torch.argmax(probs, dim=-1)
            else:
                min_encoding_indices = torch.multinomial(probs, 1)
                min_encoding_indices = min_encoding_indices.squeeze(-1)
        else:
            # look up by l2 distance, argmin
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(embedding**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

            min_encoding_indices = torch.argmin(d, dim=1)

        z_q = embedding[min_encoding_indices].view(z.shape)

        perplexity = None
        min_encodings = None
        vq_loss = None
        ar_prior_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2) 
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
            if self.entropy_loss_ratio > 0:
                entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)
            else:
                entropy_loss = 0

        if self.rot:
            # adapted from https://github.com/cfifty/rotation_trick/blob/main/src/models/vq_vae.py
            b, h, w, c = z.shape
            z = z / torch.norm(z, dim=-1, keepdim=True)
            # assert self.l2_norm, "Rot requires l2 normalization"
            z = rearrange(z, 'b h w c-> (b h w) c')
            z_q= rearrange(z_q, 'b h w c -> (b h w) c')
            pre_norm_q = self.get_very_efficient_rotation(z / (torch.norm(z, dim=1, keepdim=True) + 1e-6),
                                                            z_q / (torch.norm(z_q, dim=1, keepdim=True) + 1e-6),
                                                            z.unsqueeze(1)).squeeze()
            z_q = pre_norm_q * (
                    torch.norm(z_q, dim=1, keepdim=True) / (torch.norm(z, dim=1, keepdim=True) + 1e-6)).detach()
            z_q = rearrange(z_q, '(b h w) c -> b h w c', b=b, h=h, w=w)
        else:
            # preserve gradients
            z_q = z + (z_q - z).detach()
        
        if self.prior_model is not None and self.training:
            # ar prior training must be put after straight-through estimator, so that the gradients can be backpropagated
            assert global_step is not None and max_steps is not None, \
                "global_step and max_steps must be provided when using prior model"
            # when quantizing, there are only 2 dimensions, now change back to 
            # B N C for AR trianing
            min_indices = rearrange(min_encoding_indices, '(b n) -> b n', b=b)
            ar_prior_loss = self.ar_prior_loss_weight * self.calculate_prior_loss_with_pred(
                                rearrange(z_q, 'b h w c -> b (h w) c'),
                                indices=min_indices,
                                global_step=global_step, 
                                max_steps=max_steps
                                )
        else:
            ar_prior_loss = None


        if random_replace and self.training:
            # randomly replace the quantized vectors with the continuous input
            z = rearrange(z, '(b h w) c -> b h w c', b=b, h=h, w=w)
            mask = torch.bernoulli(torch.full(z.shape[:-1], replace_ratio)).unsqueeze(-1).to(z.device)  # replace_ratio chance of replacement
            z_q = torch.where(mask.bool(), z, z_q)
        

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, (vq_loss, commit_loss, entropy_loss, ar_prior_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.get_emb(), p=2, dim=-1)
        else:
            embedding = self.get_emb()

        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q



def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    """
    modified from llamagen and magvit
    Args:
        affinity: (b, n, n), the affinity matrix, where affinity[i, j] is the affinity 
                between encoed vector i and codebook vector j
        loss_type: how to turn the affinity into probability distribution
    """
    # shape: (b n) n
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    # target_probs.shape: (b, n, n), and sum(target_probs, dim=-1) = 1
    avg_probs = torch.mean(target_probs, dim=0) # (,n)
    # average entropy corresponeds (negatively) to the diversity of indices for a single position
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    # sample entropy is the confidence for the quantization process
    # (bn, n) -> (bn) -> avg 
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss

def compute_cosinesim_loss(feat1, feat2, dim):
    cos_sim = F.cosine_similarity(feat1, feat2, dim=dim)
    loss = 1 - cos_sim
    return torch.mean(loss)  
