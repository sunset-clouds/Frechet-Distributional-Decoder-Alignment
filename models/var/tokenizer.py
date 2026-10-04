import torch
from torch import nn

from models.var import registry
from models.ckpt_load import load_weights
from models.var.tokenizer_arch import VQVAE


class VARTokenizer(nn.Module):
    """VAR multi-scale VQVAE; latents are (B, 680) long, the ten scale maps (1x1 ... 16x16) in scale
    order. ``post_quant_conv`` and ``decoder`` train; the quantiser's phi convolutions and codebook
    stay frozen."""

    DECODER_MODULES = ("post_quant_conv", "decoder")

    def __init__(self, tokenizer_name, ckpt_path="", resolution=registry.EVAL_SIZE):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.patch_nums = registry.PATCH_NUMS
        self.model = VQVAE(**registry.model_params("tokenizer", tokenizer_name))
        load_weights(self.model, ckpt_path or registry.ckpt_path("tokenizer", tokenizer_name),
                     f"tokenizer {tokenizer_name}")
        self.train_mode()

    def _decoder_modules(self):
        return [getattr(self.model, name) for name in self.DECODER_MODULES]

    def train_mode(self):
        """Decoder-side trainable, everything upstream of the codebook frozen."""
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

    def _split(self, latents):
        return list(latents.split([p * p for p in self.patch_nums], dim=1))

    def extract_latents(self, x):
        """(B, 3, H, W) in [-1, 1] -> (B, T) long, all scales concatenated."""
        with torch.no_grad():
            return torch.cat(self.model.img_to_idxBl(x), dim=1)

    def latent_to_image(self, latents):
        """(B, T) long -> (B, 3, H, W) in [-1, 1]; each scale map is embedded, upsampled to 16x16,
        passed through its phi and summed into f_hat (upstream's ``same_shape=True`` path)."""
        device = next(self.model.parameters()).device
        scales = self._split(latents.to(device))
        return self.model.idxBl_to_img(scales, same_shape=True, last_one=True)

    def standard_reconstruction(self, x):
        return self.latent_to_image(self.extract_latents(x))
