"""VAR registry: one multi-scale VQVAE and next-scale transformers d16 / d20 / d24. A latent is the
token maps of all ten scales concatenated, 1+4+...+256 = 680 indices over a 4096 codebook.
Guidance is applied as (1 + cfg * ratio) with ratio ramping over the scales."""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CKPT = os.path.join("checkpoints", "var")
_FDPT = os.path.join("checkpoints", "fdptar")

EVAL_SIZE = 256
CODEBOOK_SIZE = 4096
PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
NUM_LATENT_TOKENS = sum(p * p for p in PATCH_NUMS)   # 680

VQVAE_ARGS = {"vocab_size": CODEBOOK_SIZE, "z_channels": 32, "ch": 160, "share_quant_resi": 4,
              "v_patch_nums": PATCH_NUMS, "test_mode": True}

# VAR's FID sampling settings
GENERATOR_DEFAULTS = {"top_k": 900, "top_p": 0.96}
# sequences per _sample call, independent of the caller's batch size
SAMPLE_CHUNK = 64

MODELS = {
    "tokenizer": {
        "var-vqvae": {"ckpt": os.path.join(_CKPT, "vae_ch160v4096z32.pth")},
    },
    "generator": {
        **{f"var-d{d}_256": {"ckpt": os.path.join(_CKPT, f"var_d{d}.pth"), "depth": d,
                             "native_resolution": 256, "cfg_omega": 1.5} for d in (16, 20, 24)},
        # FDAR post-trained, same architecture
        **{f"var-d{d}_256-fdpt": {"ckpt": os.path.join(_FDPT, f"var-d{d}.pt"), "depth": d,
                                  "native_resolution": 256, "cfg_omega": 1.5} for d in (16, 20, 24)},
    },
}

_NON_ARCH = ("ckpt", "native_resolution")


def _resolve(kind, name):
    kind = "generator" if kind == "ar" else kind
    if kind not in MODELS or name not in MODELS[kind]:
        raise ValueError(f"Unknown var {kind}: {name}. Known: {', '.join(MODELS.get(kind, {}))}")
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
        return dict(VQVAE_ARGS)
    return {k: v for k, v in MODELS[kind][name].items() if k not in _NON_ARCH}


def ckpt_path(kind, name, resolution=None):
    kind = _resolve(kind, name)
    if kind == "generator":
        _check_resolution(name, resolution)
    return os.path.join(_ROOT, MODELS[kind][name]["ckpt"])


def eval_size():
    return EVAL_SIZE


def token_length(resolution=EVAL_SIZE, name=None):
    # same count for every depth; resolution and name are ignored
    _ = resolution, name
    return NUM_LATENT_TOKENS
