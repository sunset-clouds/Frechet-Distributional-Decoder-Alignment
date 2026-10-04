"""LlamaGen class-conditional GPT, inference only.

Transformer.forward has three modes:
  (idx, cond_idx)             teacher forcing, logits[:, i] predicts idx[:, i]
  (None, cond_idx, input_pos) cached prefill of the class token
  (idx, None, input_pos)      cached decode of one image token

Sampling lives in pure_generation, which drives the model through those calls.
"""
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llamagen.registry import ModelArgs, build_args


def find_multiple(n, k):
    return n if n % k == 0 else n + k - (n % k)

# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden = int(2 * (4 * config.dim) / 3)
        if config.ffn_dim_multiplier is not None:
            hidden = int(config.ffn_dim_multiplier * hidden)
        hidden = find_multiple(hidden, config.multiple_of)
        self.w1 = nn.Linear(config.dim, hidden, bias=False)
        self.w3 = nn.Linear(config.dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))   # SwiGLU


class KVCache(nn.Module):
    def __init__(self, max_batch, max_len, n_kv_head, head_dim, dtype, device):
        super().__init__()
        shape = (max_batch, n_kv_head, max_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype, device=device))

    def update(self, input_pos, k, v):
        # k, v: (B, n_kv_head, len(input_pos), head_dim) -> write at positions, return full cache
        self.k_cache[:, :, input_pos] = k
        self.v_cache[:, :, input_pos] = v
        return self.k_cache, self.v_cache


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        self.head_dim = config.dim // config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

    def forward(self, x, freqs_cis, input_pos=None, mask=None):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        xq, xk, xv = (t.transpose(1, 2) for t in (xq, xk, xv))

        if self.kv_cache is not None:
            xk, xv = self.kv_cache.update(input_pos, xk, xv)
        rep = self.n_head // self.n_kv_head
        xk = xk.repeat_interleave(rep, dim=1)
        xv = xv.repeat_interleave(rep, dim=1)

        out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, is_causal=(mask is None))
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x, freqs_cis, input_pos=None, mask=None):
        h = x + self.attention(self.attention_norm(x), freqs_cis, input_pos, mask)
        return h + self.feed_forward(self.ffn_norm(h))


class LabelEmbedder(nn.Module):
    """Class label -> embedding. One extra null row is reserved for CFG."""
    def __init__(self, num_classes, dim):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, dim)
        self.num_classes = num_classes

    def forward(self, labels):
        return self.embedding_table(labels).unsqueeze(1)   # (B, 1, dim)


# ---------------------------------------------------------------------------
# 2D rotary positional embedding
# ---------------------------------------------------------------------------
def precompute_freqs_cis_2d(grid_size, n_elem, base=10000.0, cls_token_num=1):
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: half_dim // 2].float() / half_dim))
    t = torch.arange(grid_size)
    freqs = torch.outer(t, freqs)
    freqs_grid = torch.cat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)
    cache = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1).flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache])
    return cond_cache


def apply_rotary_emb(x, freqs_cis):
    # x: (B, seq, n_head, head_dim) ; freqs_cis: (seq, head_dim//2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out = torch.stack([
        xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
        xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1).flatten(3)
    return x_out.type_as(x)

# ---------------------------------------------------------------------------
# generation (KV-cache): one loop, generation is the prefix=None case
# ---------------------------------------------------------------------------
def _top_k_top_p(logits, top_k=0, top_p=1.0):
    if top_k > 0:
        thresh = torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(logits < thresh, -float("inf"))
    if top_p < 1.0:
        s, idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(s, dim=-1), dim=-1)
        rm = cum > top_p
        rm[..., 1:] = rm[..., :-1].clone()
        rm[..., 0] = False
        logits = logits.masked_fill(rm.scatter(1, idx, rm), -float("inf"))
    return logits


def _sample(logits, temperature=1.0, top_k=0, top_p=1.0):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = _top_k_top_p(logits, top_k, top_p)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


def _apply_cfg(logits, cfg_omega):
    if cfg_omega <= 1.0:
        return logits
    cond, uncond = logits.chunk(2, dim=0)
    return uncond + (cond - uncond) * cfg_omega


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.cls_token_num = config.cls_token_num
        self.block_size = config.block_size

        self.cls_embedding = LabelEmbedder(config.num_classes, config.dim)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        grid = int(self.block_size ** 0.5)
        assert grid * grid == self.block_size
        freqs_cis = precompute_freqs_cis_2d(grid, config.dim // config.n_head,
                                            config.rope_base, config.cls_token_num)
        # recomputed here, not loaded from checkpoint -> non-persistent buffer
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.causal_mask = None

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        """Allocate per-layer KV caches + causal mask for autoregressive generation."""
        device = self.freqs_cis.device
        head_dim = self.config.dim // self.config.n_head
        n_kv = self.config.n_kv_head if self.config.n_kv_head is not None else self.config.n_head
        for layer in self.layers:
            layer.attention.kv_cache = KVCache(max_batch_size, max_seq_length, n_kv, head_dim, dtype, device)
        self.causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length,
                                                  dtype=torch.bool, device=device))
        self.causal_mask = self.causal_mask.unsqueeze(0).repeat(max_batch_size, 1, 1)

    @torch.no_grad()
    def forward(self, idx, cond_idx, input_pos=None):
        """Three modes:
          (idx, cond_idx)            -> teacher forcing, logits[:, i] predicts idx[:, i]
          (None, cond_idx, input_pos) -> cached prefill (cond token)
          (idx, None, input_pos)      -> cached decode step (one image token)
        """
        if idx is not None and cond_idx is not None:           # teacher forcing (no cache)
            cond = self.cls_embedding(cond_idx)[:, :self.cls_token_num]
            tok = self.tok_embeddings(idx)
            h = torch.cat([cond, tok], dim=1)
            freqs_cis = self.freqs_cis[:h.shape[1]]
            mask = None
            slice_out = True
        else:                                                  # cached inference
            if cond_idx is not None:                           # prefill
                h = self.cls_embedding(cond_idx)[:, :self.cls_token_num]
            else:                                              # decode one token
                h = self.tok_embeddings(idx)
            freqs_cis = self.freqs_cis[input_pos]
            mask = self.causal_mask[:h.shape[0], None, input_pos]
            slice_out = False

        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)
        logits = self.output(self.norm(h)).float()
        return logits[:, self.cls_token_num - 1:] if slice_out else logits



def load_ar(ar_name, resolution, ckpt_path=None, device="cuda", dtype=torch.float32):
    if ckpt_path is None:                                  # resolve from registry by name
        from models.llamagen.registry import ckpt_path as _ckpt
        ckpt_path = _ckpt("ar", ar_name, resolution)
    model = Transformer(build_args(ar_name, resolution))
    if ckpt_path is not None:
        try:
            sd = torch.load(ckpt_path, map_location="cpu")
        except pickle.UnpicklingError:
            # torch>=2.6 defaults weights_only=True, which refuses anything but tensors. A
            # checkpoint written by someone else's training script carries its argparse.Namespace
            # under "args" and trips that. Only the tensors under "model" are read here, but
            # unpickling the rest executes code, so the fallback is for checkpoints whose source
            # is known -- the released FDPT-AR post-trained weights, not an arbitrary file.
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        weights = sd.get("model", sd.get("state_dict", sd))
        missing, unexpected = model.load_state_dict(weights, strict=False)
        unexpected = [k for k in unexpected if k != "freqs_cis"]   # recomputed, not loaded
        assert not missing, f"missing keys: {missing}"
        assert not unexpected, f"unexpected keys: {unexpected}"
    model = model.eval().to(device=device, dtype=dtype)
    model.freqs_cis = model.freqs_cis.float()   # keep rope table in fp32 for precision
    return model


def pure_generation(model, cond, max_token=None, cfg_omega=1.0,
                    temperature=1.0, top_k=0, top_p=1.0):
    """Class-conditional generation from scratch (no real prefix), with KV cache.

    cond     : (B,) class labels
    cfg_omega: classifier-free guidance scale (1.0 = off)
    returns  : (B, max_token) sampled image tokens

    Flow:
      1. prefill the class token into the cache (position 0); its logits predict token 0
      2. autoregressively sample token 0 .. max_token-1, feeding each prediction back in
    Every token is conditioned only on the class label and the model's own earlier
    predictions -- there is no real image content.
    """
    device = cond.device
    B = cond.shape[0]
    max_token = max_token or model.block_size
    T = model.cls_token_num
    cfg = cfg_omega > 1.0
    # CFG runs the batch twice (conditional + null class); the two are combined per step
    cond_in = torch.cat([cond, torch.full_like(cond, model.num_classes)]) if cfg else cond
    model.setup_caches(cond_in.shape[0], T + max_token, model.tok_embeddings.weight.dtype)

    seq = torch.empty(B, max_token, dtype=torch.long, device=device)

    # prefill the class token (position 0) -> last logit predicts the first image token
    logits = model(None, cond_in, torch.arange(0, T, device=device))
    nxt = _sample(_apply_cfg(logits[:, -1], cfg_omega), temperature, top_k, top_p)

    # decode tokens 0 .. max_token-1
    input_pos = torch.tensor([T], device=device)
    for t in range(max_token):
        seq[:, t:t + 1] = nxt
        if t + 1 >= max_token:
            break
        tok_in = torch.cat([nxt, nxt]) if cfg else nxt      # double for CFG
        logits = model(tok_in, None, input_pos)              # cached decode step
        nxt = _sample(_apply_cfg(logits[:, -1], cfg_omega), temperature, top_k, top_p)
        input_pos = input_pos + 1

    for layer in model.layers:   # drop cache refs
        layer.attention.kv_cache = None
    return seq
