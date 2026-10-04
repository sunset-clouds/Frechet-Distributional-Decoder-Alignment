"""Checkpoint unwrapping for the GigaTok, TiTok and VAR families: the state dict is the checkpoint
itself or its first entry among _CANDIDATES, and the key used is printed."""
import torch

_CANDIDATES = ("ema", "model", "state_dict", "module")


def _is_state_dict(value):
    return (isinstance(value, dict) and value
            and all(isinstance(k, str) and torch.is_tensor(v) for k, v in value.items()))


def extract_state_dict(checkpoint, description):
    if _is_state_dict(checkpoint):
        return checkpoint
    for key in _CANDIDATES:
        if _is_state_dict(checkpoint.get(key)):
            print(f"[ckpt] {description}: using checkpoint key '{key}'", flush=True)
            return checkpoint[key]
    raise KeyError(f"{description} checkpoint has no state dict under {_CANDIDATES}; "
                   f"top-level keys are {list(checkpoint)}")


def load_weights(module, ckpt_path, description, allow_missing=(), strip_prefix=()):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = extract_state_dict(checkpoint, description)
    for prefix in strip_prefix:
        if any(k.startswith(prefix) for k in weights):
            weights = {k[len(prefix):] if k.startswith(prefix) else k: v
                       for k, v in weights.items()}
    missing, unexpected = module.load_state_dict(weights, strict=False)
    missing = [k for k in missing if not any(p in k for p in allow_missing)]
    unexpected = [k for k in unexpected if not any(p in k for p in allow_missing)]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch for {description} at '{ckpt_path}': "
                           f"missing={missing[:8]}, unexpected={unexpected[:8]}")
