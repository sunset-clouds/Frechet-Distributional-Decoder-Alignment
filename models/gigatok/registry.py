"""GigaTok registry: tokenizer VQ_SS256 and generator GPT-B."""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CKPT = os.path.join("checkpoints", "gigatok")
_FDPT = os.path.join("checkpoints", "fdptar")

EVAL_SIZE = 256
CODEBOOK_SIZE = 16384
NUM_LATENT_TOKENS = 256   # 1D tokens
CODEBOOK_EMBED_DIM = 8

# (H, W) of the quantised latent: (B, embed_dim, 1, T)
LATENT_SHAPE = (1, NUM_LATENT_TOKENS)

# model.init_args of upstream's VQ_SS256.yaml; out_inner_dim is what their loader fills in from
# trainer.distill_loss (dinov2-vit-b)
TOK_INIT_ARGS = {
    "codebook_size": CODEBOOK_SIZE, "codebook_embed_dim": CODEBOOK_EMBED_DIM,
    "codebook_l2_norm": True, "codebook_show_usage": True, "commit_loss_beta": 0.25,
    "entropy_loss_ratio": 0.0,
    "encoder_ch_mult": [1, 1, 2, 2, 4], "decoder_ch_mult": [1, 1, 2, 2, 4],
    "model_size": None, "encoder_size": "small", "decoder_size": "small",
    "num_latent_tokens": NUM_LATENT_TOKENS, "z_channels": 256, "dropout_p": 0.0,
    "multi_level_query_init": True,
    "last_level_2d_query_init": False, "multi_level_2d_query_init": False,
    "adaptive_gn": True, "d2s_up": True, "rot": True, "use_attn": False,
    "fea_rec_loss_type": "mse", "fea_rec_loss_weight": 1.0, "distill_depth": 3,
    "out_inner_dim": 768,
}

GENERATOR_DEFAULTS = {
    "use_cfg": True,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    "cfg_interval": -1,
}

MODELS = {
    "tokenizer": {
        "gigatok-vqss": {"ckpt": os.path.join(_CKPT, "VQ_SS256_e100.pt")},
    },
    "generator": {
        "gigatok-B_256": {"ckpt": os.path.join(_CKPT, "GPT_B256_e300_VQ_SS.pt"),
                          "gpt_model": "GPT-B", "native_resolution": 256, "cfg_omega": 2.00},
        # FDAR post-trained, same architecture
        "gigatok-B_256-fdpt": {"ckpt": os.path.join(_FDPT, "gigatok-ss.pt"),
                               "gpt_model": "GPT-B", "native_resolution": 256, "cfg_omega": 2.00},
    },
}

_NON_ARCH = ("ckpt", "native_resolution")


def _resolve(kind, name):
    kind = "generator" if kind == "ar" else kind
    if kind not in MODELS or name not in MODELS[kind]:
        raise ValueError(f"Unknown gigatok {kind}: {name}. Known: {', '.join(MODELS.get(kind, {}))}")
    return kind


def native_resolution(name):
    return MODELS[_resolve("generator", name)][name]["native_resolution"]


def _check_resolution(name, resolution):
    if resolution is not None and resolution != native_resolution(name):
        raise ValueError(f"{name} is a {native_resolution(name)}-native checkpoint but "
                         f"resolution={resolution} was requested.")


def model_params(kind, name):
    kind = _resolve(kind, name)
    if kind == "tokenizer":
        return {"init_args": dict(TOK_INIT_ARGS)}
    return {k: v for k, v in MODELS[kind][name].items() if k not in _NON_ARCH}


def ckpt_path(kind, name, resolution=None):
    kind = _resolve(kind, name)
    if kind == "generator":
        _check_resolution(name, resolution)
    return os.path.join(_ROOT, MODELS[kind][name]["ckpt"])


def eval_size():
    return EVAL_SIZE


def token_length(resolution=EVAL_SIZE, name=None):
    # 1D tokens: fixed count; resolution and name are ignored
    _ = resolution, name
    return NUM_LATENT_TOKENS
