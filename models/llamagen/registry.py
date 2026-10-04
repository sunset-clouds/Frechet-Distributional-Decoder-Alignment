"""LlamaGen registry. A bare name is the 384-native model (24x24 tokens); the ``_256`` suffix is
the 256-native one (16x16), e.g. ``--generator_name llamagen-B_256 --resolution 256``."""

import os
from dataclasses import dataclass
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_SIZE = 256
VQ_SCALE_FACTOR = 16
CODEBOOK_SIZE = 16384

TOKENIZER_DEFAULTS = {
    "codebook_size": CODEBOOK_SIZE,
    "codebook_embed_dim": 8,
    "codebook_l2_norm": True,
    "z_channels": 256,
    "encoder_ch_mult": [1, 1, 2, 2, 4],
    "decoder_ch_mult": [1, 1, 2, 2, 4],
    "dropout_p": 0.0,
}

GENERATOR_DEFAULTS = {
    "use_cfg": True,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
}

_GEN = os.path.join("checkpoints", "llamagen", "generator")
# FDAR post-trained ARs
_FDPT = os.path.join("checkpoints", "fdptar")

MODELS = {
    "tokenizer": {
        # one tokenizer for every generator (16x downsample)
        "llamagen-vq16": {"ckpt": os.path.join("checkpoints", "llamagen", "tokenizer",
                                               "vq_ds16_c2i.pt")},
    },
    "generator": {
        # --- 384-native, 24x24 tokens ---
        "llamagen-B":     {"ckpt": os.path.join(_GEN, "c2i_B_384.pt"),   "native_resolution": 384,
                           "dim": 768,  "n_layer": 12, "n_head": 12, "cfg_omega": 2.25},
        "llamagen-L":     {"ckpt": os.path.join(_GEN, "c2i_L_384.pt"),   "native_resolution": 384,
                           "dim": 1024, "n_layer": 24, "n_head": 16, "cfg_omega": 2.00},
        # --- 256-native, 16x16 tokens ---
        "llamagen-B_256": {"ckpt": os.path.join(_GEN, "c2i_B_256.pt"),   "native_resolution": 256,
                           "dim": 768,  "n_layer": 12, "n_head": 12, "cfg_omega": 2.00},
        "llamagen-L_256": {"ckpt": os.path.join(_GEN, "c2i_L_256.pt"),   "native_resolution": 256,
                           "dim": 1024, "n_layer": 24, "n_head": 16, "cfg_omega": 2.00},
        # FDAR post-trained, same architecture; cfg_omega 2.00 is their sampler's
        "llamagen-B_256-fdpt": {"ckpt": os.path.join(_FDPT, "llamagen-b.pt"),
                                "native_resolution": 256,
                                "dim": 768,  "n_layer": 12, "n_head": 12, "cfg_omega": 2.00},
        "llamagen-L_256-fdpt": {"ckpt": os.path.join(_FDPT, "llamagen-l.pt"),
                                "native_resolution": 256,
                                "dim": 1024, "n_layer": 24, "n_head": 16, "cfg_omega": 2.00},
    },
}

# entry keys that are not ModelArgs fields
_NON_ARCH = ("ckpt", "native_resolution")


def _resolve(kind, name):
    # "ar" is an alias of "generator"
    kind = "generator" if kind == "ar" else kind
    if kind not in MODELS or name not in MODELS[kind]:
        known = ", ".join(MODELS.get(kind, {}))
        raise ValueError(f"Unknown llamagen {kind}: {name}. Known: {known}")
    return kind


def native_resolution(name):
    return MODELS[_resolve("generator", name)][name]["native_resolution"]


def _check_resolution(name, resolution):
    if resolution is None:
        return
    native = native_resolution(name)
    if resolution != native:
        raise ValueError(
            f"{name} is a {native}-native checkpoint but resolution={resolution} was requested. "
            f"Use the matching name (e.g. llamagen-B for 384, llamagen-B_256 for 256) rather than "
            f"changing the resolution, or the model is built with {(resolution // 16) ** 2} token "
            f"positions and loaded with weights for {(native // 16) ** 2}."
        )


def model_params(kind, name):
    kind = _resolve(kind, name)
    if kind == "tokenizer":
        return dict(TOKENIZER_DEFAULTS)
    return {k: v for k, v in MODELS[kind][name].items() if k not in _NON_ARCH}


def ckpt_path(kind, name, resolution=None):
    kind = _resolve(kind, name)
    if kind == "generator":
        _check_resolution(name, resolution)
    rel_path = MODELS[kind][name]["ckpt"]
    return os.path.join(_ROOT, rel_path) if rel_path else ""


def eval_size():
    return EVAL_SIZE


def token_length(resolution=EVAL_SIZE, name=None):
    # name is ignored; the count follows from the image size
    _ = name
    grid = resolution // VQ_SCALE_FACTOR
    return grid * grid


def latent_resolution(resolution=EVAL_SIZE):
    grid = resolution // VQ_SCALE_FACTOR
    return (grid, grid)


# --- AR hyper-parameters: name -> (dim, n_layer, n_head), resolution -> block_size ---
@dataclass
class ModelArgs:
    # by model name
    dim: int = 1024
    n_layer: int = 24
    n_head: int = 16
    n_kv_head: Optional[int] = None
    # by resolution
    block_size: int = 576          # number of image tokens = grid ** 2
    cls_token_num: int = 1         # class-conditional: one prepended cond token
    # fixed
    vocab_size: int = 16384
    num_classes: int = 1000        # + 1 null slot reserved inside LabelEmbedder (CFG)
    cfg_omega: float = 2.0         # guidance scale used when CFG is enabled
    # architecture constants
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000.0
    norm_eps: float = 1e-5

def build_args(ar_name, resolution, downsample=16):
    """(ar_name, resolution) -> ModelArgs with the registry's architecture params."""
    grid = resolution // downsample
    return ModelArgs(**model_params("ar", ar_name), block_size=grid * grid, cls_token_num=1)
