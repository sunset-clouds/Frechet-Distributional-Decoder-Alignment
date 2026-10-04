import torch
from torch import nn

from models.titok import registry
from models.ckpt_load import load_weights
from models.titok.tokenizer_arch import TiTok


class TiTokTokenizer(nn.Module):
    """TiTok L-32 / B-64; latents are (B, T) long token indices, T = 32 or 64. Decode: ViT
    ``decoder`` -> logits over the pixel codebook -> softmax mix (``pixel_quantize``) ->
    convolutional ``pixel_decoder``."""

    DECODER_MODULES = ("decoder", "pixel_quantize", "pixel_decoder")

    def __init__(self, tokenizer_name, ckpt_path="", resolution=registry.EVAL_SIZE):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        config = registry.config("tokenizer", tokenizer_name)
        self.token_length = int(config.model.vq_model.num_latent_tokens)
        self.model = TiTok(config)
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
        decoder = getattr(self.model.pixel_decoder, "module", self.model.pixel_decoder)
        return decoder.conv_out.weight

    def extract_latents(self, x):
        """(B, 3, H, W) in [-1, 1] -> (B, T) long token indices; ``encode`` is fed [0, 1]."""
        with torch.no_grad():
            _, result = self.model.encode((x + 1) / 2)
            return result["min_encoding_indices"].reshape(x.shape[0], -1)

    def latent_to_image(self, latents):
        """(B, T) long -> (B, 3, H, W) in [-1, 1]."""
        device = next(self.model.parameters()).device
        decoded = self.model.decode_tokens(latents.to(device).reshape(latents.shape[0], -1))
        return (decoded * 2 - 1).clamp(-1, 1)

    def standard_reconstruction(self, x):
        return self.latent_to_image(self.extract_latents(x))
