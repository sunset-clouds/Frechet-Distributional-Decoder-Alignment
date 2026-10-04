import torch
from torch import nn

from models.titok import registry
from models.ckpt_load import load_weights
from models.titok.generator_arch import ImageBert, UViTBert


class TiTokGenerator(nn.Module):
    """Frozen TiTok MaskGIT generator: all-mask start, ``num_steps`` parallel decoding steps;
    ``standard_generation`` -> (B, T) long token indices."""

    def __init__(self, generator_name, ckpt_path="", resolution=registry.EVAL_SIZE,
                 cfg_omega=None):
        super().__init__()
        self.generator_name = generator_name
        config = registry.config("generator", generator_name)
        registry._check_resolution(generator_name, resolution)
        self.token_length = int(config.model.vq_model.num_latent_tokens)
        gen = config.model.generator
        self.sampling = {
            "guidance_scale": float(gen.guidance_scale),
            "guidance_decay": str(gen.get("guidance_decay", "constant")),
            "guidance_scale_pow": float(gen.get("guidance_scale_pow", 3.0)),
            "randomize_temperature": float(gen.randomize_temperature),
            # B-64's yaml omits the key
            "softmax_temperature_annealing": bool(gen.get("softmax_temperature_annealing", True)),
            "num_sample_steps": int(gen.get("num_steps", 8)),
        }
        self.cfg_omega = self.sampling["guidance_scale"] if cfg_omega is None else float(cfg_omega)
        model_cls = UViTBert if str(gen.model_type) == "UViT" else ImageBert
        self.model = model_cls(config)
        load_weights(self.model, ckpt_path or registry.ckpt_path("generator", generator_name),
                     f"generator {generator_name}")
        self.eval_mode()

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
            labels = torch.randint(0, 1000, (batch_size,), device=device)
        else:
            labels = labels.to(device)

        options = dict(self.sampling)
        options["guidance_scale"] = self.cfg_omega if use_cfg else 1.0

        with torch.no_grad():
            if seeds is None:
                return self.model.generate(labels, **options).reshape(batch_size, -1)
            out = torch.empty(batch_size, self.token_length, dtype=torch.long, device=device)
            for i, seed in enumerate(seeds):
                torch.manual_seed(int(seed))
                out[i] = self.model.generate(labels[i:i + 1], **options).reshape(-1)
            return out
