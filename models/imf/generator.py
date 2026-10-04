import torch
from torch import nn

from models.imf import registry
from models.imf.generator_arch import (
    BatchGenerator,
    GlobalGenerator,
    extract_official_imf_state,
    iMeanFlow,
    validate_official_imf_load,
)


class IMFGenerator(nn.Module):
    """Frozen 1-NFE MeanFlow transformer, B / L / XL; ``standard_generation`` -> (B, 4096) float,
    the normalized SD-VAE latent flattened. Guidance (omega, t_min, t_max) is evaluated inside
    the network; ``use_cfg=False`` samples with omega = 1 over the whole interval."""

    def __init__(self, generator_name, ckpt_path="", resolution=registry.EVAL_SIZE,
                 cfg_omega=None):
        super().__init__()
        self.generator_name = generator_name
        params = registry.model_params("generator", generator_name)
        registry._check_resolution(generator_name, resolution)
        self.token_length = registry.token_length(resolution, name=generator_name)
        self.cfg_omega = params["cfg_omega"] if cfg_omega is None else float(cfg_omega)
        self.t_min, self.t_max = params["t_min"], params["t_max"]
        self.num_classes = 1000

        self.model = iMeanFlow(params["model_str"], img_size=registry.LATENT_SIZE,
                               img_channels=registry.LATENT_CHANNELS,
                               num_classes=self.num_classes, eval=True)
        path = ckpt_path or registry.ckpt_path("generator", generator_name)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = extract_official_imf_state(checkpoint, checkpoint_key="model",
                                           target_keys=set(self.model.state_dict()))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        validate_official_imf_load(missing, unexpected)
        self.eval_mode()

    def eval_mode(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def standard_generation(self, batch_size, labels=None, use_cfg=True, device=None, seeds=None):
        """-> (batch_size, 4096) float latents; ``seeds``: one CPU torch.Generator per sample."""
        if device is None:
            device = next(self.model.parameters()).device
        omega = self.cfg_omega if use_cfg else 1.0
        t_min = self.t_min if use_cfg else 0.0
        t_max = self.t_max if use_cfg else 1.0
        rng = GlobalGenerator(device) if seeds is None else BatchGenerator(device, seeds)
        if labels is None:
            labels = rng.randint(0, self.num_classes, (batch_size,))
        labels = labels.to(device)
        latents = self.model.generate(
            n_sample=batch_size, rng=rng,
            num_steps=registry.GENERATOR_DEFAULTS["num_steps"],
            omega=omega, t_min=t_min, t_max=t_max, labels=labels)
        return latents.flatten(1)
