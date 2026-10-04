import torch
from torch import nn

from models.gigatok import registry
from models.ckpt_load import load_weights
from models.gigatok.tokenizer_arch import VQVitModelPlus, VQVitModelPlusArgs


class GigaTokTokenizer(nn.Module):
    """GigaTok VQ_SS256; latents are (B, 256) long token indices, decoded by ``s1to2decoder``
    (tokens -> spatial grid) then ``decoder`` (grid -> pixels)."""

    # everything after the quantiser
    DECODER_MODULES = ("post_quant_conv", "s1to2decoder", "decoder")

    def __init__(self, tokenizer_name, ckpt_path="", resolution=registry.EVAL_SIZE):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        params = registry.model_params("tokenizer", tokenizer_name)
        self.embed_dim = registry.CODEBOOK_EMBED_DIM
        self.latent_shape = registry.LATENT_SHAPE
        self.model = VQVitModelPlus(VQVitModelPlusArgs(**params["init_args"]))
        load_weights(self.model, ckpt_path or registry.ckpt_path("tokenizer", tokenizer_name),
                     f"tokenizer {tokenizer_name}", allow_missing=("codebook_used",))
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

    def extract_latents(self, x):
        """(B, 3, H, W) in [-1, 1] -> (B, T) long token indices."""
        with torch.no_grad():
            _, _, info = self.model.encode(x, return_code=True)
            return info[2].reshape(x.shape[0], -1)

    def latent_to_image(self, latents):
        """(B, T) long -> (B, 3, H, W) in [-1, 1]; codebook lookup at (B, C, 1, T), ``decode``
        returns (pixels, features)."""
        device = next(self.model.parameters()).device
        h, w = self.latent_shape
        shape = (latents.shape[0], self.embed_dim, h, w)
        z_q = self.model.quantize.get_codebook_entry(latents.reshape(-1).to(device), shape)
        return self.model.decode(z_q)[0].clamp(-1, 1)

    def standard_reconstruction(self, x):
        return self.latent_to_image(self.extract_latents(x))
