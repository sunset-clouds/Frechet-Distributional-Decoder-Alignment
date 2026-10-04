"""TiTok registry: L-32 and B-64 tokenizer/generator pairs, ``pair_for`` naming each generator's
tokenizer. CONFIGS carries upstream's titok_l32.yaml / titok_b64.yaml, field names kept."""

import os

from omegaconf import OmegaConf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CKPT = os.path.join("checkpoints", "titok")
_FDPT = os.path.join("checkpoints", "fdptar")

EVAL_SIZE = 256
CODEBOOK_SIZE = 4096


def _config(size, tokens, randomize_temperature, guidance_scale, **generator_extra):
    return {
        "model": {
            "vq_model": {
                "codebook_size": CODEBOOK_SIZE, "token_size": 12, "use_l2_norm": True,
                "commitment_cost": 0.25,
                "vit_enc_model_size": size, "vit_dec_model_size": size,
                "vit_enc_patch_size": 16, "vit_dec_patch_size": 16,
                "num_latent_tokens": tokens, "finetune_decoder": True,
            },
            "generator": {
                "model_type": "ViT", "hidden_size": 768, "num_hidden_layers": 24,
                "num_attention_heads": 16, "intermediate_size": 3072,
                "dropout": 0.1, "attn_drop": 0.1, "num_steps": 8, "class_label_dropout": 0.1,
                "image_seq_len": tokens, "condition_num_classes": 1000,
                "randomize_temperature": randomize_temperature, "guidance_scale": guidance_scale,
                "guidance_decay": "linear", **generator_extra,
            },
        },
        "dataset": {"preprocessing": {"crop_size": EVAL_SIZE}},
    }


# B-64's yaml omits softmax_temperature_annealing; generator.py defaults it to True
CONFIGS = {
    "L32": _config("large", 32, 9.5, 4.5, softmax_temperature_annealing=True),
    "B64": _config("base", 64, 11.0, 3.0),
}

MODELS = {
    "tokenizer": {
        "titok-vql32": {"ckpt": os.path.join(_CKPT, "tokenizer_titok_l32.bin"), "config": "L32"},
        "titok-vqb64": {"ckpt": os.path.join(_CKPT, "tokenizer_titok_b64.bin"), "config": "B64"},
    },
    "generator": {
        "titok-L32": {"ckpt": os.path.join(_CKPT, "generator_titok_l32.bin"), "config": "L32",
                      "native_resolution": 256},
        "titok-B64": {"ckpt": os.path.join(_CKPT, "generator_titok_b64.bin"), "config": "B64",
                      "native_resolution": 256},
        # FDAR post-trained, same architecture
        "titok-L32-fdpt": {"ckpt": os.path.join(_FDPT, "titok-l32.pt"), "config": "L32",
                           "native_resolution": 256},
        "titok-B64-fdpt": {"ckpt": os.path.join(_FDPT, "titok-b64.pt"), "config": "B64",
                           "native_resolution": 256},
    },
}

_PAIRS = {"titok-L32": "titok-vql32", "titok-B64": "titok-vqb64",
          "titok-L32-fdpt": "titok-vql32", "titok-B64-fdpt": "titok-vqb64"}


def pair_for(generator_name):
    """The tokenizer name this generator's tokens decode with."""
    return _PAIRS[generator_name]


def _resolve(kind, name):
    kind = "generator" if kind == "ar" else kind
    if kind not in MODELS or name not in MODELS[kind]:
        raise ValueError(f"Unknown titok {kind}: {name}. Known: {', '.join(MODELS.get(kind, {}))}")
    return kind


def native_resolution(name):
    return MODELS[_resolve("generator", name)][name]["native_resolution"]


def _check_resolution(name, resolution):
    if resolution is not None and resolution != native_resolution(name):
        raise ValueError(f"{name} is a {native_resolution(name)}-native checkpoint but "
                         f"resolution={resolution} was requested.")


def config(kind, name):
    return OmegaConf.create(CONFIGS[MODELS[_resolve(kind, name)][name]["config"]])


def model_params(kind, name):
    return {"config": config(kind, name)}


def ckpt_path(kind, name, resolution=None):
    kind = _resolve(kind, name)
    if kind == "generator":
        _check_resolution(name, resolution)
    return os.path.join(_ROOT, MODELS[kind][name]["ckpt"])


def eval_size():
    return EVAL_SIZE


def token_length(resolution=EVAL_SIZE, name=None):
    """num_latent_tokens of the named model (32 or 64); ``name`` required, ``resolution`` ignored."""
    _ = resolution
    if name is None:
        raise ValueError("titok.token_length needs the model name: L-32 and B-64 differ")
    kind = "generator" if name in MODELS["generator"] else "tokenizer"
    return int(config(kind, name).model.vq_model.num_latent_tokens)
