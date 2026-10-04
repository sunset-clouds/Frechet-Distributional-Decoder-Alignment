"""iMF registry: one SD-VAE tokenizer and 1-NFE MeanFlow transformers B / L / XL, plus the
FD-Loss FD-SIM post-trained twins. A latent is the normalized SD-VAE posterior mean flattened
to 4 * 32 * 32 = 4096 floats. Guidance is (omega, t_min, t_max) inside the network."""

import os
from functools import partial

from models.imf.generator_arch import DiT_iMF

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CKPT = os.path.join("checkpoints", "imf")

EVAL_SIZE = 256
CODEBOOK_SIZE = None            # continuous latents; pools store float16
LATENT_CHANNELS = 4
LATENT_SIZE = 32
NUM_LATENT_DIMS = LATENT_CHANNELS * LATENT_SIZE * LATENT_SIZE   # 4096

# latent normalization of the FD-Loss iMF port
LATENT_MEAN = (0.86488, -0.27787343, 0.21616915, 0.3738409)
LATENT_STD = (4.85503674, 5.31922414, 3.93725398, 3.9870003)

GENERATOR_DEFAULTS = {"num_steps": 1}

_SIZES = {
    "B": {"model_str": "DiT_iMF_B_2", "cfg_omega": 8.0, "t_min": 0.4, "t_max": 0.65},
    "L": {"model_str": "DiT_iMF_L_2", "cfg_omega": 10.5, "t_min": 0.4, "t_max": 0.6},
    "XL": {"model_str": "DiT_iMF_XL_2", "cfg_omega": 8.0, "t_min": 0.42, "t_max": 0.62},
}

MODELS = {
    "tokenizer": {
        "imf-sdvae": {"ckpt": os.path.join(_CKPT, "sdvae")},
    },
    "generator": {
        **{f"imf-{s}_256": {"ckpt": os.path.join(_CKPT, f"iMF-{s}.pth"),
                            "native_resolution": 256, **p} for s, p in _SIZES.items()},
        # FD-Loss FD-SIM post-trained, same architecture
        **{f"imf-{s}_256-fdsim": {"ckpt": os.path.join(_CKPT, f"iMF-{s}_FD-SIM.pth"),
                                  "native_resolution": 256, **p} for s, p in _SIZES.items()},
    },
}

_NON_ARCH = ("ckpt", "native_resolution")


def _resolve(kind, name):
    kind = "generator" if kind == "ar" else kind
    if kind not in MODELS or name not in MODELS[kind]:
        raise ValueError(f"Unknown imf {kind}: {name}. Known: {', '.join(MODELS.get(kind, {}))}")
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
        return {}
    return {k: v for k, v in MODELS[kind][name].items() if k not in _NON_ARCH}


def ckpt_path(kind, name, resolution=None):
    kind = _resolve(kind, name)
    if kind == "generator":
        _check_resolution(name, resolution)
    return os.path.join(_ROOT, MODELS[kind][name]["ckpt"])


def eval_size():
    return EVAL_SIZE


def token_length(resolution=EVAL_SIZE, name=None):
    # flattened latent dims, same for every size; resolution and name are ignored
    _ = resolution, name
    return NUM_LATENT_DIMS


DiT_iMF_B_2 = partial(DiT_iMF, depth=12, hidden_size=768, patch_size=2, num_heads=12,
                      aux_head_depth=8)

DiT_iMF_L_2 = partial(DiT_iMF, depth=32, hidden_size=1024, patch_size=2, num_heads=16,
                      aux_head_depth=8)

DiT_iMF_XL_2 = partial(DiT_iMF, depth=48, hidden_size=1024, patch_size=2, num_heads=16,
                       aux_head_depth=8)
