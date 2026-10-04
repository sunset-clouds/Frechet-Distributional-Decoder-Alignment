import torch
from torch import nn

from models.gigatok import registry
from models.ckpt_load import load_weights
from models.gigatok.generator_arch import GPT_models, generate


class GigaTokGenerator(nn.Module):
    """Frozen GigaTok GPT-B class-conditional AR; ``standard_generation`` -> (B, 256) long token
    indices."""

    def __init__(self, generator_name, ckpt_path="", resolution=registry.EVAL_SIZE,
                 cfg_omega=None):
        super().__init__()
        self.generator_name = generator_name
        self.params = registry.model_params("generator", generator_name)
        registry._check_resolution(generator_name, resolution)
        self.token_length = registry.token_length(resolution, name=generator_name)
        self.cfg_omega = self.params["cfg_omega"] if cfg_omega is None else float(cfg_omega)
        self.model = GPT_models[self.params["gpt_model"]](
            block_size=self.token_length, num_classes=1000, model_type="c2i")
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
            labels = torch.randint(0, self.model.num_classes, (batch_size,), device=device)
        else:
            labels = labels.to(device)

        defaults = registry.GENERATOR_DEFAULTS
        sample = lambda cond: generate(
            self.model, cond, self.token_length,
            cfg_scale=self.cfg_omega if use_cfg else 1.0,
            cfg_interval=defaults["cfg_interval"], temperature=defaults["temperature"],
            top_k=defaults["top_k"], top_p=defaults["top_p"], sample_logits=True,
        )

        with torch.no_grad():
            if seeds is None:
                return sample(labels)
            out = torch.empty(batch_size, self.token_length, dtype=torch.long, device=device)
            for i, seed in enumerate(seeds):
                torch.manual_seed(int(seed))
                out[i] = sample(labels[i:i + 1])[0]
            return out
