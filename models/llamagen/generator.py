import os
import torch
from torch import nn

from models.llamagen import registry
from models.llamagen.generator_arch import load_ar, pure_generation


class LlamaGenGenerator(nn.Module):
    """Frozen LlamaGen AR sampler; ``standard_generation`` -> (B, T) long token indices."""

    def __init__(self, generator_name, ckpt_path="", resolution=registry.EVAL_SIZE,
                 cfg_omega=None):
        super(LlamaGenGenerator, self).__init__()
        self.generator_name = generator_name
        self.params = registry.model_params("generator", generator_name)
        registry._check_resolution(generator_name, resolution)
        self.ckpt_path = ckpt_path or registry.ckpt_path("generator", generator_name, resolution)
        self.resolution = resolution
        self.token_length = registry.token_length(resolution, name=generator_name)
        self.cfg_omega = self.params["cfg_omega"] if cfg_omega is None else float(cfg_omega)
        self.model = self._build_model()
        self.eval_mode()

    def _build_model(self):
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f"LlamaGen checkpoint not found: {self.ckpt_path}. "
                "Pass --generator_ckpt_path or update models/llamagen/registry.py."
            )
        return load_ar(self.generator_name, self.resolution, self.ckpt_path,
                       device="cpu", dtype=torch.bfloat16)

    def eval_mode(self):
        self.eval()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def standard_generation(self, batch_size, labels=None, use_cfg=True, device=None, seeds=None):
        """-> (batch_size, T) long token indices; ``seeds``: one torch.manual_seed per sample."""
        if device is None:
            device = next(self.model.parameters()).device

        if labels is None:
            labels = torch.randint(0, self.model.num_classes, (batch_size,), device=device)
        else:
            labels = labels.to(device)

        cfg_omega = self.cfg_omega if use_cfg else 1.0
        defaults = registry.GENERATOR_DEFAULTS

        with torch.no_grad():
            if seeds is None:
                return pure_generation(
                    self.model, labels, max_token=self.token_length, cfg_omega=cfg_omega,
                    temperature=defaults["temperature"], top_k=defaults["top_k"],
                    top_p=defaults["top_p"],
                )
            out = torch.empty(batch_size, self.token_length, dtype=torch.long, device=device)
            for i, seed in enumerate(seeds):
                torch.manual_seed(int(seed))
                out[i] = pure_generation(
                    self.model, labels[i:i + 1], max_token=self.token_length, cfg_omega=cfg_omega,
                    temperature=defaults["temperature"], top_k=defaults["top_k"],
                    top_p=defaults["top_p"],
                )[0]
            return out
