"""GigaTok GPT-1D generator and sampler, flattened from the upstream release.

Concatenated in dependency order; intra-package imports removed, bodies otherwise
unchanged.
"""

from models.gigatok.tokenizer_arch import DropPath
from dataclasses import dataclass
from typing import Optional, List
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo.config
import torch._inductor.config
import copy
import math
import torch.distributed as dist


# ------------------------------------------------------------------------------
# autoregressive/models/gpt_1d.py
# ------------------------------------------------------------------------------
# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
#   ELM:      https://github.com/Pepper-lll/LMforImageGeneration/blob/master/llama/ar_model.py






# check the version of pytorch, 
# if pytorch version >= 2.2.0, then flash_attention can be used 
if torch.__version__ >= "2.2.0":
    HAS_FLASH_ATTENTION_V2 = True
    # print("flash_attention can be used.")
else:
    HAS_FLASH_ATTENTION_V2 = False
    # print("flash_attention is not supported.")



def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0
    rope: bool = False
    use_qk_norm: bool = False
    use_flash_attn: bool = False
    use_adaLN: bool = False
    use_simple_adaLN: bool = False
    no_adaLN_before: int = None

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.rope = config.rope
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim
        self.use_qk_norm = config.use_qk_norm
        self.use_flash_attn = config.use_flash_attn
        # flash attention can be switched to normal attention for inference
        # rasie error only when training and use_flash_attn is True and 
        # flash attention is not installed
        assert HAS_FLASH_ATTENTION_V2 or (not self.use_flash_attn) or (not self.training), \
            "Flash attention is not installed and cannot be used when training"
        
        if self.use_flash_attn:
            print("Using flash attention!")

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)


    def forward(
        self, x: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)


        if self.rope:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        if self.use_flash_attn: 
            # Shape: (batch_size, num_heads, seq_length, head_dim)
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
                output = F.scaled_dot_product_attention(
                    xq, keys, values, 
                    attn_mask=mask, 
                    is_causal=True if mask is None else False, # is_causal=False is for KV cache
                    dropout_p=self.attn_dropout_p if self.training else 0)            
        else:
            output = F.scaled_dot_product_attention(
                xq, keys, values, 
                attn_mask=mask, 
                is_causal=True if mask is None else False, # is_causal=False is for KV cache
                dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_adaLN = config.use_adaLN
        self.use_simple_adaLN = config.use_simple_adaLN

        assert not (self.use_adaLN and self.use_simple_adaLN), "please choose one of them"
        
        if self.use_adaLN:
            # use adaptive layer normalization (zero)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.dim, 6 * config.dim, bias=True)
            )

        if self.use_simple_adaLN:
            self.ada_gss = nn.Parameter(torch.randn(1,1,6,config.dim) / config.dim**0.5)


    def forward(
        self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None,
        cond_adaln: Optional[torch.Tensor] = None
        ):
        if self.use_adaLN:
            assert cond is not None, "please provide cond for adaLN"
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=2)
            h = x + gate_msa * self.drop_path(self.attention(
                                                modulate(self.attention_norm(x), shift_msa, scale_msa), 
                                                start_pos, mask, freqs_cis=freqs_cis))
            out = h + gate_mlp * self.drop_path(self.feed_forward(modulate(self.ffn_norm(h), shift_mlp, scale_mlp)))
        elif self.use_simple_adaLN:
            assert cond_adaln is not None, "please provide cond_adaln for simple adaLN"
            # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=2)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (cond_adaln + self.ada_gss).unbind(2) # 116C + B16C
            h = x + gate_msa * self.drop_path(self.attention(
                                                modulate(self.attention_norm(x), shift_msa, scale_msa), 
                                                start_pos, mask, freqs_cis=freqs_cis))
            out = h + gate_mlp * self.drop_path(self.feed_forward(modulate(self.ffn_norm(h), shift_mlp, scale_mlp)))
        else:
            h = x + self.drop_path(self.attention(
                                        self.attention_norm(x), 
                                        start_pos, mask, freqs_cis=freqs_cis))

            out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.num_img_tokens = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.rope = config.rope
        assert self.cls_token_num == 1

        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            if config.no_adaLN_before is not None and layer_id <= config.no_adaLN_before \
                and (config.use_adaLN or config.use_simple_adaLN):
                new_config = deepcopy(config)
                new_config.use_adaLN = False
                new_config.use_simple_adaLN = False
                self.layers.append(TransformerBlock(new_config, dpr[layer_id]))

            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)
        # 1d absolute positional embedding
        scale = self.config.dim ** -0.5

        if self.rope:
            self.freqs_cis = precompute_freqs_cis(
                                self.num_img_tokens, 
                                self.config.dim // self.config.n_head, 
                                self.cls_token_num
                            )
        else:
            self.positional_embedding = nn.Parameter(
                    scale * torch.randn(self.cls_token_num + self.num_img_tokens, self.config.dim))
        
        if self.config.use_simple_adaLN:
            self.adaLN = nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(config.dim, 6 * config.dim, bias=True)
                    )
            # print("We are using adaLN!")

        if self.config.use_adaLN or self.config.use_simple_adaLN:
            self.final_adaLN = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.dim, 2 * config.dim, bias=True)
            )
        
        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out adaLN modulation layers in GPT blocks:
        if self.config.use_adaLN:
            for layer in self.layers:
                nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_adaLN[-1].weight, 0)
            nn.init.constant_(self.final_adaLN[-1].bias, 0)
        
        if self.config.use_simple_adaLN:
            nn.init.constant_(self.adaLN[-1].weight, 0)
            nn.init.constant_(self.adaLN[-1].bias, 0)
            nn.init.constant_(self.final_adaLN[-1].weight, 0)
            nn.init.constant_(self.final_adaLN[-1].bias, 0)
            # for layer in self.layers:
            #     nn.init.constant_(layer.ada_gss, 0)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        # grid_size = int(self.config.block_size ** 0.5)
        # assert grid_size * grid_size == self.block_size
        # self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)

    def forward(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        ret_inner_feat_layer: Optional[int] = None,
        using_kv_cache: Optional[bool] = False
    ):

        # self.positional_embedding = self.positional_embedding.to(h.device)
        if idx is not None and cond_idx is not None and not using_kv_cache: # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            token_embeddings = self.tok_embeddings(idx)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)

            if not self.rope:
                token_embeddings += self.positional_embedding[:token_embeddings.shape[1]].to(token_embeddings.dtype)
            
            # always use the cond_embeddings with positional embeddings for adaLN
            cond_embeddings = token_embeddings[:,:self.cls_token_num]
            h = self.tok_dropout(token_embeddings)
        else:
            if cond_idx is not None and idx is None: # prefill in inference
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
                cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
                if not self.rope:
                    token_embeddings += self.positional_embedding[:self.cls_token_num]
                cond_embeddings = token_embeddings
            else: # decode_n_tokens(kv cache) in inference
                token_embeddings = self.tok_embeddings(idx)
                if not self.rope:
                    token_embeddings += self.positional_embedding[input_pos].to(token_embeddings.dtype)
                
                # extra cond_embeddings for adaLN
                if cond_idx is not None:
                    cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num].to(token_embeddings.dtype)
                    cond_embeddings += self.positional_embedding[:self.cls_token_num].to(token_embeddings.dtype)
            
            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
            # self.freqs_cis = self.freqs_cis

        if self.rope:
            if self.training:
                freqs_cis = self.freqs_cis[:token_embeddings.shape[1]].to(h.device)
            else:
                self.freqs_cis = self.freqs_cis.to(h.device)
                freqs_cis = self.freqs_cis[input_pos]

        if self.config.use_simple_adaLN:
            B, L, C = cond_embeddings.shape
            cond_adaln = self.adaLN(cond_embeddings).reshape(B, L, 6, C) # shared_adaLN
        else:
            cond_adaln = None

        # transformer blocks
        for idx, layer in enumerate(self.layers):
            h = layer(h, input_pos, mask, 
                      freqs_cis=freqs_cis if self.rope else None,
                      cond=cond_embeddings if self.config.use_adaLN else None,
                      cond_adaln=cond_adaln,
                      )
            if ret_inner_feat_layer is not None and idx == ret_inner_feat_layer - 1:
                inner_feat = h.detach().clone()
                return inner_feat
        
        # output layers
        if self.config.use_adaLN or self.config.use_simple_adaLN:
            shift, scale = self.final_adaLN(cond_embeddings).chunk(2, dim=2)
            h = modulate(self.norm(h), shift, scale)
        else:
            h = self.norm(h)
 
        logits = self.output(h).float()
        
        if self.training:
            logits = logits[:, self.cls_token_num - 1:].contiguous()

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            # deprecated
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss


    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)



#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
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
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache 


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)



#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}


# ------------------------------------------------------------------------------
# autoregressive/models/generate.py
# ------------------------------------------------------------------------------
# Modified from:
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

# from autoregressive.models.gpt_ic import Transformer_IC


### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def logits_to_probs(logits, temperature: float = 1.0, top_p: float=1.0, top_k: int = None, **kwargs):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    if cfg_scale > 1.0:
        logits, _ = model(None, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(None, cond_idx, input_pos)

    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(
        model, 
        x: torch.Tensor, 
        input_pos: torch.Tensor, 
        cfg_scale: float, 
        cfg_flag: bool, 
        cond_idx: torch.Tensor,
        **sampling_kwargs,
    ):
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _ = model(x_combined, cond_idx=cond_idx, input_pos=input_pos, using_kv_cache=True)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _ = model(x, cond_idx=cond_idx, input_pos=input_pos, using_kv_cache=True)
    return sample(logits, **sampling_kwargs)


# a series of cfg scheduling functions
def constant_schedule(ratio, cfg_scale, **kwargs):
    return cfg_scale

def linear_schedule(ratio, cfg_scale, **kwargs):
    return 1.0 + (cfg_scale - 1.0) * ratio + 1e-4

def linear_re_schedule(ratio, cfg_scale, **kwargs):
    return 1.0 + (cfg_scale - 1.0) * (1.0 - ratio) + 1e-4

def triangular_schedule(ratio, cfg_scale, peak=0.5, **kwargs):
    if ratio < peak:
        return 1.0 + (cfg_scale - 1.0) * (ratio / peak) + 1e-4
    else:
        return 1.0 + (cfg_scale - 1.0) * ((1.0 - ratio) / (1.0 - peak)) + 1e-4

def rectangular_schedule(ratio, cfg_scale, window_start=0.05, window_end=1.0, **kwargs):
    if window_start <= ratio <= window_end:
        return cfg_scale
    else:
        return 1.0 + 1e-4

def step_schedule(ratio, cfg_scale, window_start, **kwargs):
    if cfg_scale == 1.0:
        return cfg_scale
    elif window_start <= ratio:
        return cfg_scale
    else:
        return 1.0 + 1e-4


def cosine_schedule_re(ratio, cfg_scale, **kwargs):
    return 1.0 + (cfg_scale - 1.0) * (0.5 * (1 + math.cos(math.pi * ratio)))


def decode_n_tokens(
    model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int, cfg_schedule: str = "constant", cfg_schedule_kwargs: dict = {},
    **sampling_kwargs):
    """
    Args:
        cfg_interval: -1 means always cfg, other value bigger than -1 means only first cfg_interval tokens are cfg
    """
    # reference: 
    # https://github.com/Pepper-lll/LMforImageGeneration/blob/9aed8d795ce2c79cbfaa75884c09fa2e1fc744ee/llama/ar_model.py#L510
    valid_schedules = ["constant", "linear", "linear_re", "triangular", "rectangular", "cosine", "sigmoid", "step"]
    assert cfg_schedule in valid_schedules, f"cfg_schedule must be one of {valid_schedules}"

    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(num_new_tokens):
        ratio = 1. * ( i + 1 ) / num_new_tokens
        if cfg_schedule == 'constant':
            cfg = constant_schedule(ratio, cfg_scale)
        elif cfg_schedule == 'linear':
            cfg = linear_schedule(ratio, cfg_scale)
        elif cfg_schedule == 'linear_re':
            cfg = linear_re_schedule(ratio, cfg_scale)
        elif cfg_schedule == 'triangular':
            cfg = triangular_schedule(ratio, cfg_scale)
        elif cfg_schedule == 'rectangular':
            cfg = rectangular_schedule(ratio, cfg_scale, **cfg_schedule_kwargs)
        elif cfg_schedule == 'step':
            cfg = step_schedule(ratio, cfg_scale, **cfg_schedule_kwargs)
        elif cfg_schedule == 'cosine':
            cfg = cosine_schedule_re(ratio, cfg_scale)

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, cfg, cfg_flag, **sampling_kwargs
            )
            input_pos += 1

            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(-1, 1)
    
    return new_tokens, new_probs


@torch.no_grad()
def generate(
        model, 
        cond, 
        max_new_tokens, 
        emb_masks=None, 
        cfg_scale=1.0, 
        cfg_interval=-1, 
        cfg_schedule="constant",
        cfg_schedule_kwargs={},
        prefilled_guidance=False,
        **sampling_kwargs):
    device = cond.device
    if model.model_type == 'c2i':
        if not prefilled_guidance:
            if cfg_scale > 1.0:
                cond_null = torch.ones_like(cond, device=device) * model.num_classes
                cond_combined = torch.cat([cond, cond_null])
            else:
                cond_combined = cond
            T = 1
        else:
            raise NotImplementedError
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond, device=device) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        # FD post-training keeps master parameters in fp32 and samples under
        # autocast.  The KV cache must follow the active compute dtype rather
        # than the embedding parameter dtype, otherwise SDPA receives mixed
        # bf16/fp32 query and key tensors.
        cache_dtype = model.tok_embeddings.weight.dtype
        try:
            autocast_enabled = torch.is_autocast_enabled("cuda")
            if autocast_enabled:
                cache_dtype = torch.get_autocast_dtype("cuda")
        except TypeError:  # compatibility with older PyTorch 2.x
            if torch.is_autocast_enabled():
                cache_dtype = torch.get_autocast_gpu_dtype()
        model.setup_caches(
            max_batch_size=max_batch_size_cfg,
            max_seq_length=max_seq_length,
            dtype=cache_dtype,
        )
    
    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    input_pos = torch.arange(0, T, device=device)
    # check device
    rank = int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else 0
    # if rank == 0:
    #     # print device
    #     print(next(model.parameters()).device)
    #     print(cond_combined.device)
    #     print(input_pos.device)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, T:T+1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(
                            model, 
                            next_token, 
                            input_pos, 
                            max_new_tokens-1, 
                            cfg_scale, 
                            cfg_interval, 
                            cfg_schedule=cfg_schedule,
                            cfg_schedule_kwargs=cfg_schedule_kwargs,
                            cond_idx=cond_combined,
                            **sampling_kwargs)
    seq[:, T+1:] = torch.cat(generated_tokens, dim=1)

    return seq[:, T:]
