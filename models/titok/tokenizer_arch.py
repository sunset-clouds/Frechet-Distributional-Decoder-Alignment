"""TiTok tokenizer, flattened from the upstream release.

Concatenated in dependency order; intra-package imports removed, bodies otherwise
unchanged.
"""

import os
from typing import Union, Callable, Dict, Optional
import torch
import math
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
import einops
from einops.layers.torch import Rearrange
from typing import Mapping, Text, Tuple
from einops import rearrange
from accelerate.utils.operations import gather
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch import nn
import json
from omegaconf import OmegaConf
from pathlib import Path
from huggingface_hub import PyTorchModelHubMixin


# ------------------------------------------------------------------------------
# modeling/modules/base_model.py
# ------------------------------------------------------------------------------
"""This file contains some base class implementation for models.

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference:
    https://github.com/huggingface/open-muse/blob/main/muse/modeling_utils.py
"""



class BaseModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def save_pretrained_weight(
        self,
        save_directory: Union[str, os.PathLike],
        save_function: Callable = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Saves a model and its configuration file to a directory.

        Args:
            save_directory: A string or os.PathLike, directory to which to save. 
                Will be created if it doesn't exist.
            save_function: A Callable function, the function to use to save the state dictionary.
                Useful on distributed training like TPUs when one need to replace `torch.save` by
                another method. Can be configured with the environment variable `DIFFUSERS_SAVE_MODE`.
            state_dict: A dictionary from str to torch.Tensor, the state dictionary to save.
                If `None`, the model's state dictionary will be saved.
        """
        if os.path.isfile(save_directory):
            print(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        model_to_save = self

        if state_dict is None:
            state_dict = model_to_save.state_dict()
        weights_name = "pytorch_model.bin"

        save_function(state_dict, os.path.join(save_directory, weights_name))

        print(f"Model weights saved in {os.path.join(save_directory, weights_name)}")

    def load_pretrained_weight(
        self,
        pretrained_model_path: Union[str, os.PathLike],
        strict_loading: bool = True,
        torch_dtype: Optional[torch.dtype] = None
    ):
        r"""Instantiates a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        Args:
            pretrained_model_path: A string or os.PathLike, a path to a *directory* or *file* containing model weights.

        Raises:
            ValueError: If pretrained_model_path does not exist.
        """
        # If pretrained_model_path is a file, set model_file to this file.
        if os.path.isfile(pretrained_model_path):
            model_file = pretrained_model_path
        # If pretrained_model_path is a directory, set model_file to the path of the 
        # file "pytorch_model.bin" in this directory.
        elif os.path.isdir(pretrained_model_path):
            pretrained_model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
            if os.path.isfile(pretrained_model_path):
                model_file = pretrained_model_path
            else:
                raise ValueError(f"{pretrained_model_path} does not exist")
        else:
            raise ValueError(f"{pretrained_model_path} does not exist")

        # Load model state from checkpoint.
        checkpoint = torch.load(model_file, map_location="cpu")
        # Load state dictionary into self.
        msg = self.load_state_dict(checkpoint, strict=strict_loading)
        # Print information about loading weights.
        print(f"loading weight from {model_file}, msg: {msg}")
        # If torch_dtype is specified and is a valid torch.dtype, convert self to this dtype.
        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        elif torch_dtype is not None:
            self.to(torch_dtype)

        # Set model in evaluation mode to deactivate DropOut modules by default.
        self.eval()

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """Gets the number of parameters in the module.

        Args:
            only_trainable: A boolean, whether to only include trainable parameters.
            exclude_embeddings: A boolean, whether to exclude parameters associated with embeddings.

        Returns:
            An integer, the number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight"
                for name, module_type in self.named_modules()
                if isinstance(module_type, torch.nn.Embedding)
            ]
            non_embedding_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
            return sum(p.numel() for p in non_embedding_parameters if p.requires_grad or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)


# ------------------------------------------------------------------------------
# modeling/modules/blocks.py
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
    https://github.com/baofff/U-ViT/blob/main/libs/timm.py
"""



def modulate(x, shift, scale):
    return x * (1 + scale) + shift


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

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
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
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UViTBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


class TiTokEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size 
        self.patch_size = config.model.vq_model.vit_enc_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_enc_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size

        if config.model.vq_model.get("quantize_mode", "vq") == "vae":
            self.token_size = self.token_size * 2 # needs to split into mean and std

        self.is_legacy = config.model.vq_model.get("is_legacy", True)

        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]
        
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=self.width,
              kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)

    def forward(self, pixel_values, latent_tokens):
        batch_size = pixel_values.shape[0]
        x = pixel_values
        x = self.patch_embed(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]
        

        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        latent_tokens = x[:, 1+self.grid_size**2:]
        latent_tokens = self.ln_post(latent_tokens)
        # fake 2D shape
        if self.is_legacy:
            latent_tokens = latent_tokens.reshape(batch_size, self.width, self.num_latent_tokens, 1)
        else:
            # Fix legacy problem.
            latent_tokens = latent_tokens.reshape(batch_size, self.num_latent_tokens, self.width, 1).permute(0, 2, 1, 3)
        latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size, 1, self.num_latent_tokens)
        return latent_tokens
    

class TiTokDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_dec_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_dec_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size
        self.is_legacy = config.model.vq_model.get("is_legacy", True)
        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
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
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        # add mask token and query pos embed
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)

        if self.is_legacy:
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, 2 * self.width, 1, padding=0, bias=True),
                nn.Tanh(),
                nn.Conv2d(2 * self.width, 1024, 1, padding=0, bias=True),
            )
            self.conv_out = nn.Identity()
        else:
            # Directly predicting RGB pixels
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
                Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
                    p1 = self.patch_size, p2 = self.patch_size),)
            self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)
    
    def forward(self, z_quantized):
        N, C, H, W = z_quantized.shape
        assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.grid_size**2] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x


class TATiTokDecoder(TiTokDecoder):
    def __init__(self, config):
        super().__init__(config)
        scale = self.width ** -0.5
        self.text_context_length = config.model.vq_model.get("text_context_length", 77)
        self.text_embed_dim = config.model.vq_model.get("text_embed_dim", 768)
        self.text_guidance_proj = nn.Linear(self.text_embed_dim, self.width)
        self.text_guidance_positional_embedding = nn.Parameter(scale * torch.randn(self.text_context_length, self.width))

    def forward(self, z_quantized, text_guidance):
        N, C, H, W = z_quantized.shape
        assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)

        text_guidance = self.text_guidance_proj(text_guidance)
        text_guidance = text_guidance + self.text_guidance_positional_embedding
        x = torch.cat([x, text_guidance], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.grid_size**2] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x
    

class WeightTiedLMHead(nn.Module):
    def __init__(self, embeddings, target_codebook_size):
        super().__init__()
        self.weight = embeddings.weight
        self.target_codebook_size = target_codebook_size

    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        # Get the weights for the target codebook size
        weight = self.weight[:self.target_codebook_size]  # Shape: [target_codebook_size, embed_dim]
        # Compute the logits by matrix multiplication
        logits = torch.matmul(x, weight.t())  # Shape: [batch_size, seq_len, target_codebook_size]
        return logits


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# ------------------------------------------------------------------------------
# modeling/quantizer/quantizer.py
# ------------------------------------------------------------------------------
"""Vector quantizer.

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
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
    https://github.com/google-research/magvit/blob/main/videogvt/models/vqvae.py
    https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/distributions/distributions.py
    https://github.com/lyndonzheng/CVQ-VAE/blob/main/quantise.py
"""


class VectorQuantizer(torch.nn.Module):
    def __init__(self,
                 codebook_size: int = 1024,
                 token_size: int = 256,
                 commitment_cost: float = 0.25,
                 use_l2_norm: bool = False,
                 clustering_vq: bool = False
                 ):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_size = token_size
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.use_l2_norm = use_l2_norm

        self.clustering_vq = clustering_vq
        if clustering_vq:
            self.decay = 0.99
            self.register_buffer("embed_prob", torch.zeros(self.codebook_size))

    # Ensure quantization is performed using f32
    @autocast(enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z = z.float()
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 'b h w c -> (b h w) c')
        unnormed_z_flattened = z_flattened

        if self.use_l2_norm:
            z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
            embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, embedding.T)

        min_encoding_indices = torch.argmin(d, dim=1) # num_ele
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        if self.use_l2_norm:
            z = torch.nn.functional.normalize(z, dim=-1)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        codebook_loss = torch.mean((z_quantized - z.detach()) **2)

        if self.clustering_vq and self.training:
            with torch.no_grad():
                # Gather distance matrix from all GPUs.
                encoding_indices = gather(min_encoding_indices)
                if len(min_encoding_indices.shape) != 1:
                    raise ValueError(f"min_encoding_indices in a wrong shape, {min_encoding_indices.shape}")
                # Compute and update the usage of each entry in the codebook.
                encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=z.device)
                encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
                avg_probs = torch.mean(encodings, dim=0)
                self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1-self.decay)
                # Closest sampling to update the codebook.
                all_d = gather(d)
                all_unnormed_z_flattened = gather(unnormed_z_flattened).detach()
                if all_d.shape[0] != all_unnormed_z_flattened.shape[0]:
                    raise ValueError(
                        "all_d and all_unnormed_z_flattened have different length" + 
                        f"{all_d.shape}, {all_unnormed_z_flattened.shape}")
                indices = torch.argmin(all_d, dim=0)
                random_feat = all_unnormed_z_flattened[indices]
                # Decay parameter based on the average usage.
                decay = torch.exp(-(self.embed_prob * self.codebook_size * 10) /
                                   (1 - self.decay) - 1e-3).unsqueeze(1).repeat(1, self.token_size)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay

        loss = commitment_loss + codebook_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=min_encoding_indices.view(z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3])
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        if len(indices.shape) == 1:
            z_quantized = self.embedding(indices)
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum('bd,dn->bn', indices, self.embedding.weight)
        else:
            raise NotImplementedError
        if self.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        return z_quantized
    

class DiagonalGaussianDistribution(object):
    @autocast(enabled=False)
    def __init__(self, parameters, deterministic=False):
        """Initializes a Gaussian distribution instance given the parameters.

        Args:
            parameters (torch.Tensor): The parameters for the Gaussian distribution. It is expected
                to be in shape [B, 2 * C, *], where B is batch size, and C is the embedding dimension.
                First C channels are used for mean and last C are used for logvar in the Gaussian distribution.
            deterministic (bool): Whether to use deterministic sampling. When it is true, the sampling results
                is purely based on mean (i.e., std = 0).
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters.float(), 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    @autocast(enabled=False)
    def sample(self):
        x = self.mean.float() + self.std.float() * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    @autocast(enabled=False)
    def mode(self):
        return self.mean

    @autocast(enabled=False)
    def kl(self):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            return 0.5 * torch.sum(torch.pow(self.mean.float(), 2)
                                    + self.var.float() - 1.0 - self.logvar.float(),
                                    dim=[1, 2])


# ------------------------------------------------------------------------------
# modeling/modules/maskgit_vqgan.py
# ------------------------------------------------------------------------------
"""This file contains code for MaskGIT-VQGAN.

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference:
    https://github.com/huggingface/open-muse/blob/main/muse/modeling_maskgit_vqgan.py
"""
# Copyright 2023 Google LLC and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""MaskGIT Tokenizer based on VQGAN.

This tokenizer is a reimplementation of VQGAN [https://arxiv.org/abs/2012.09841]
with several modifications. The non-local layers are removed from VQGAN for
faster speed.
"""




# Conv2D with same padding
class Conv2dSame(nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return super().forward(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels_ = self.in_channels if self.out_channels is None else self.out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = Conv2dSame(self.in_channels, self.out_channels_, kernel_size=3, bias=False)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels_, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.conv2 = Conv2dSame(self.out_channels_, self.out_channels_, kernel_size=3, bias=False)

        if self.in_channels != self.out_channels_:
            self.nin_shortcut = Conv2dSame(self.out_channels_, self.out_channels_, kernel_size=1, bias=False)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels_:
            residual = self.nin_shortcut(hidden_states)

        return hidden_states + residual


class DownsamplingBlock(nn.Module):
    def __init__(self, config, block_idx: int):
        super().__init__()

        self.config = config
        self.block_idx = block_idx

        in_channel_mult = (1,) + tuple(self.config.channel_mult)
        block_in = self.config.hidden_channels * in_channel_mult[self.block_idx]
        block_out = self.config.hidden_channels * self.config.channel_mult[self.block_idx]

        res_blocks = nn.ModuleList()
        for _ in range(self.config.num_res_blocks):
            res_blocks.append(ResnetBlock(block_in, block_out, dropout_prob=self.config.dropout))
            block_in = block_out
        self.block = res_blocks

        self.downsample = self.block_idx != self.config.num_resolutions - 1

    def forward(self, hidden_states):
        for res_block in self.block:
            hidden_states = res_block(hidden_states)

        if self.downsample:
            hidden_states = F.avg_pool2d(hidden_states, kernel_size=2, stride=2)

        return hidden_states


class UpsamplingBlock(nn.Module):
    def __init__(self, config, block_idx: int):
        super().__init__()

        self.config = config
        self.block_idx = block_idx

        if self.block_idx == self.config.num_resolutions - 1:
            block_in = self.config.hidden_channels * self.config.channel_mult[-1]
        else:
            block_in = self.config.hidden_channels * self.config.channel_mult[self.block_idx + 1]

        block_out = self.config.hidden_channels * self.config.channel_mult[self.block_idx]

        res_blocks = []
        for _ in range(self.config.num_res_blocks):
            res_blocks.append(ResnetBlock(block_in, block_out, dropout_prob=self.config.dropout))
            block_in = block_out
        self.block = nn.ModuleList(res_blocks)

        self.add_upsample = self.block_idx != 0
        if self.add_upsample:
            self.upsample_conv = Conv2dSame(block_out, block_out, kernel_size=3)

    def forward(self, hidden_states):
        for res_block in self.block:
            hidden_states = res_block(hidden_states)

        if self.add_upsample:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            hidden_states = self.upsample_conv(hidden_states)

        return hidden_states


class Pixel_Eecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # downsampling
        self.conv_in = Conv2dSame(self.config.num_channels, self.config.hidden_channels, kernel_size=3, bias=False)

        downsample_blocks = []
        for i_level in range(self.config.num_resolutions):
            downsample_blocks.append(DownsamplingBlock(self.config, block_idx=i_level))
        self.down = nn.ModuleList(downsample_blocks)

        # middle
        mid_channels = self.config.hidden_channels * self.config.channel_mult[-1]
        res_blocks = nn.ModuleList()
        for _ in range(self.config.num_res_blocks):
            res_blocks.append(ResnetBlock(mid_channels, mid_channels, dropout_prob=self.config.dropout))
        self.mid = res_blocks

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=mid_channels, eps=1e-6, affine=True)
        self.conv_out = Conv2dSame(mid_channels, self.config.z_channels, kernel_size=1)

    def forward(self, pixel_values):
        # downsampling
        hidden_states = self.conv_in(pixel_values)
        for block in self.down:
            hidden_states = block(hidden_states)

        # middle
        for block in self.mid:
            hidden_states = block(hidden_states)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class Pixel_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # compute in_channel_mult, block_in and curr_res at lowest res
        block_in = self.config.hidden_channels * self.config.channel_mult[self.config.num_resolutions - 1]
        curr_res = self.config.resolution // 2 ** (self.config.num_resolutions - 1)
        self.z_shape = (1, self.config.z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = Conv2dSame(self.config.z_channels, block_in, kernel_size=3)

        # middle
        res_blocks = nn.ModuleList()
        for _ in range(self.config.num_res_blocks):
            res_blocks.append(ResnetBlock(block_in, block_in, dropout_prob=self.config.dropout))
        self.mid = res_blocks

        # upsampling
        upsample_blocks = []
        for i_level in reversed(range(self.config.num_resolutions)):
            upsample_blocks.append(UpsamplingBlock(self.config, block_idx=i_level))
        self.up = nn.ModuleList(list(reversed(upsample_blocks)))  # reverse to get consistent order

        # end
        block_out = self.config.hidden_channels * self.config.channel_mult[0]
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_out, eps=1e-6, affine=True)
        self.conv_out = Conv2dSame(block_out, self.config.num_channels, kernel_size=3)

    def forward(self, hidden_states):
        # z to block_in
        hidden_states = self.conv_in(hidden_states)

        # middle
        for block in self.mid:
            hidden_states = block(hidden_states)

        # upsampling
        for block in reversed(self.up):
            hidden_states = block(hidden_states)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class Pixel_Quantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    Discretization bottleneck part of the VQ-VAE.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        r"""
        Args:
            num_embeddings: number of vectors in the quantized space.
            embedding_dim: dimensionality of the tensors in the quantized space.
                Inputs to the modules must be in this format as well.
            commitment_cost: scalar which controls the weighting of the loss terms
                (see equation 4 in the paper https://arxiv.org/abs/1711.00937 - this variable is Beta).
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, hidden_states, return_loss=False):
        """
        Inputs the output of the encoder network z and maps it to a discrete one-hot vector that is the index of the
        closest embedding vector e_j z (continuous) -> z_q (discrete) z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()

        distances = self.compute_distances(hidden_states)
        min_encoding_indices = torch.argmin(distances, axis=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings).to(hidden_states)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(hidden_states.shape)

        # reshape to (batch, num_tokens)
        min_encoding_indices = min_encoding_indices.reshape(hidden_states.shape[0], -1)

        # compute loss for embedding
        loss = None
        if return_loss:
            loss = torch.mean((z_q.detach() - hidden_states) ** 2) + self.commitment_cost * torch.mean(
                (z_q - hidden_states.detach()) ** 2
            )
            # preserve gradients
            z_q = hidden_states + (z_q - hidden_states).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, min_encoding_indices, loss

    def compute_distances(self, hidden_states):
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        hidden_states_flattended = hidden_states.reshape((-1, self.embedding_dim))
        emb_weights = self.embedding.weight.t()

        inputs_norm_sq = hidden_states_flattended.pow(2.0).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = emb_weights.pow(2.0).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            hidden_states_flattended,
            emb_weights,
            alpha=-2.0,
        )
        return distances

    def get_codebook_entry(self, indices):
        # indices are expected to be of shape (batch, num_tokens)
        # get quantized latent vectors
        if len(indices.shape) == 2:
            batch, num_tokens = indices.shape
            z_q = self.embedding(indices)
            z_q = z_q.reshape(batch, int(math.sqrt(num_tokens)), int(math.sqrt(num_tokens)), -1).permute(0, 3, 1, 2)
        elif len(indices.shape) == 3:
            batch, height, width = indices.shape
            indices = indices.view(batch, -1)
            z_q = self.embedding(indices)
            z_q = z_q.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
        else:
            print(indices.shape)
            raise NotImplementedError
        return z_q

    # adapted from https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/models/rqvae/quantizations.py#L372
    def get_soft_code(self, hidden_states, temp=1.0, stochastic=False):
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()  # (batch, height, width, channel)
        distances = self.compute_distances(hidden_states)  # (batch * height * width, num_embeddings)

        soft_code = F.softmax(-distances / temp, dim=-1)  # (batch * height * width, num_embeddings)
        if stochastic:
            code = torch.multinomial(soft_code, 1)  # (batch * height * width, 1)
        else:
            code = distances.argmin(dim=-1)  # (batch * height * width)

        code = code.reshape(hidden_states.shape[0], -1)  # (batch, height * width)
        batch, num_tokens = code.shape
        soft_code = soft_code.reshape(batch, num_tokens, -1)  # (batch, height * width, num_embeddings)
        return soft_code, code

    def get_code(self, hidden_states):
        # reshape z -> (batch, height, width, channel)
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()
        distances = self.compute_distances(hidden_states)
        indices = torch.argmin(distances, axis=1).unsqueeze(1)
        indices = indices.reshape(hidden_states.shape[0], -1)
        return indices


# ------------------------------------------------------------------------------
# modeling/titok.py
# ------------------------------------------------------------------------------
"""This file contains the model definition of TiTok.

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
"""





class PretrainedTokenizer(nn.Module):
    def __init__(self, pretrained_weight):
        super().__init__()
        conf = OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 256})
        self.encoder = Pixel_Eecoder(conf)
        self.decoder = Pixel_Decoder(conf)
        self.quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        # Load pretrained weights
        self.load_state_dict(torch.load(pretrained_weight, map_location=torch.device("cpu")), strict=True)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode(self, x):
        hidden_states = self.encoder(x)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states)
        return codebook_indices.detach()
    
    @torch.no_grad()
    def decode(self, codes):
        quantized_states = self.quantize.get_codebook_entry(codes)
        rec_images = self.decoder(quantized_states)
        rec_images = torch.clamp(rec_images, 0.0, 1.0)
        return rec_images.detach()
    
    @torch.no_grad()
    def decode_tokens(self, codes):
        return self.decode(codes)


class TiTok(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2406.07550", "image-tokenization"], repo_url="https://github.com/bytedance/1d-tokenizer", license="apache-2.0"):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        # This should be False for stage1 and True for stage2.
        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", True)

        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        if self.quantize_mode not in ["vq", "vae"]:
            raise ValueError(f"Unsupported quantize mode {self.quantize_mode}.")
        
        if self.finetune_decoder and self.quantize_mode not in ["vq"]:
            raise ValueError("Only supprot finetune_decoder with vq quantization for now.")

        self.encoder = TiTokEncoder(config)
        self.decoder = TiTokDecoder(config)
        
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size,
                token_size=config.model.vq_model.token_size,
                commitment_cost=config.model.vq_model.commitment_cost,
                use_l2_norm=config.model.vq_model.use_l2_norm,)
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError
        
        if self.finetune_decoder:
            # Freeze encoder/quantizer/latent tokens
            self.latent_tokens.requires_grad_(False)
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)

            # Include MaskGiT-VQGAN's quantizer and decoder
            self.pixel_quantize = Pixel_Quantizer(
                num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
            self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
                {"channel_mult": [1, 1, 2, 2, 4],
                "num_resolutions": 5,
                "dropout": 0.0,
                "hidden_channels": 128,
                "num_channels": 3,
                "num_res_blocks": 2,
                "resolution": 256,
                "z_channels": 256}))
        
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

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

    def encode(self, x):
        if self.finetune_decoder:
            with torch.no_grad():
                self.encoder.eval()
                self.quantize.eval()
                z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
                z_quantized, result_dict = self.quantize(z)
                result_dict["quantizer_loss"] *= 0
                result_dict["commitment_loss"] *= 0
                result_dict["codebook_loss"] *= 0
        else:
            z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
            if self.quantize_mode == "vq":
                z_quantized, result_dict = self.quantize(z)
            elif self.quantize_mode == "vae":
                posteriors = self.quantize(z)
                z_quantized = posteriors.sample()
                result_dict = posteriors

        return z_quantized, result_dict
    
    def decode(self, z_quantized):
        decoded = self.decoder(z_quantized)
        if self.finetune_decoder:
            quantized_states = torch.einsum(
                'nchw,cd->ndhw', decoded.softmax(1),
                self.pixel_quantize.embedding.weight)
            decoded = self.pixel_decoder(quantized_states)
        return decoded
    
    def decode_tokens(self, tokens):
        if self.quantize_mode == "vq":
            tokens = tokens.squeeze(1)
            batch, seq_len = tokens.shape # B x N
            z_quantized = self.quantize.get_codebook_entry(
                tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
            z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        elif self.quantize_mode == "vae":
            z_quantized = tokens
        decoded = self.decode(z_quantized)
        return decoded
    
    def forward(self, x):
        z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized)
        return decoded, result_dict
