import torch
from torch import nn
from diffusers import AutoencoderKL

from models.imf import registry


class IMFTokenizer(nn.Module):
    """SD-VAE (sd-vae-ft-mse) with the FD-Loss latent normalization; latents are the normalized
    posterior mean flattened to (B, 4096) float. Only ``decoder`` trains; ``post_quant_conv`` and the
    encoder side stay frozen."""

    DECODER_MODULES = ("decoder",)

    def __init__(self, tokenizer_name, ckpt_path="", resolution=registry.EVAL_SIZE):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.model = AutoencoderKL.from_pretrained(
            ckpt_path or registry.ckpt_path("tokenizer", tokenizer_name))
        self.register_buffer("latent_mean", torch.tensor(registry.LATENT_MEAN).view(1, 4, 1, 1))
        self.register_buffer("latent_std", torch.tensor(registry.LATENT_STD).view(1, 4, 1, 1))
        self.train_mode()

    def _decoder_modules(self):
        return [getattr(self.model, name) for name in self.DECODER_MODULES]

    def train_mode(self):
        """Decoder trainable; encoder, quant_conv and post_quant_conv frozen."""
        self.model.train()
        trainable = set()
        for module in self._decoder_modules():
            for param in module.parameters():
                param.requires_grad = True
                trainable.add(id(param))
        for param in self.model.parameters():
            if id(param) not in trainable:
                param.requires_grad = False

    def eval_mode(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        return [p for module in self._decoder_modules() for p in module.parameters()]

    @property
    def last_layer(self):
        decoder = getattr(self.model.decoder, "module", self.model.decoder)
        return decoder.conv_out.weight

    def _unflatten(self, latents):
        c, s = registry.LATENT_CHANNELS, registry.LATENT_SIZE
        if latents.dim() == 2:
            return latents.view(latents.shape[0], c, s, s)
        return latents

    def extract_latents(self, x):
        """(B, 3, H, W) in [-1, 1] -> (B, 4096) float, normalized posterior mean."""
        with torch.no_grad():
            latents = self.model.encode(x).latent_dist.mean
            latents = (latents - self.latent_mean) / self.latent_std
        return latents.flatten(1)

    def latent_to_image(self, latents):
        """(B, 4096) or (B, 4, 32, 32) float -> (B, 3, H, W) in [-1, 1]."""
        device = next(self.model.parameters()).device
        latents = self._unflatten(latents.to(device).float())
        latents = latents * self.latent_std + self.latent_mean
        return self.model.decode(latents).sample.clamp(-1, 1)

    def standard_reconstruction(self, x):
        return self.latent_to_image(self.extract_latents(x))
