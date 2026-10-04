import torch
from torch import nn

from models.llamagen import registry
from models.llamagen.tokenizer_arch import Encoder, Decoder, VectorQuantizer


class VQModel(nn.Module):
    def __init__(self, codebook_size, codebook_embed_dim, codebook_l2_norm,
                 encoder_ch_mult, decoder_ch_mult, z_channels, dropout_p):
        super().__init__()
        self.encoder = Encoder(ch_mult=encoder_ch_mult, z_channels=z_channels, dropout=dropout_p)
        self.decoder = Decoder(ch_mult=decoder_ch_mult, z_channels=z_channels, dropout=dropout_p)
        self.quantize = VectorQuantizer(codebook_size, codebook_embed_dim, codebook_l2_norm)
        self.quant_conv = nn.Conv2d(z_channels, codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(codebook_embed_dim, z_channels, 1)

    def encode(self, x):
        z_e = self.quant_conv(self.encoder(x))
        z_q, indices = self.quantize(z_e)
        return z_e, z_q, indices

    def decode(self, z_q):
        return self.decoder(self.post_quant_conv(z_q))


class LlamaGenTokenizer(nn.Module):
    """LlamaGen VQ-16; latents are (B, T) long token indices, T = (H / 16) * (W / 16)."""

    DECODER_MODULES = ("post_quant_conv", "decoder")

    def __init__(self, tokenizer_name, ckpt_path="", resolution=registry.EVAL_SIZE):
        super(LlamaGenTokenizer, self).__init__()
        params = registry.model_params("tokenizer", tokenizer_name)
        self.tokenizer_name = tokenizer_name
        self.embed_dim = params["codebook_embed_dim"]
        self.latent_resolution = registry.latent_resolution(resolution)
        self.model = VQModel(**params)
        self._load(ckpt_path or registry.ckpt_path("tokenizer", tokenizer_name))
        self.train_mode()

    def _load(self, ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        weights = sd.get("model", sd.get("state_dict", sd))
        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        unexpected = [k for k in unexpected if "codebook_used" not in k]
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch for {self.tokenizer_name} at '{ckpt_path}': "
                f"missing={missing}, unexpected={unexpected}"
            )

    def train_mode(self):
        """Decoder-side trainable, everything upstream of the codebook frozen."""
        self.model.train()
        for module in (self.model.encoder, self.model.quant_conv, self.model.quantize):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        for module in (self.model.post_quant_conv, self.model.decoder):
            for param in module.parameters():
                param.requires_grad = True

    def eval_mode(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        return (list(self.model.post_quant_conv.parameters()) + list(self.model.decoder.parameters()))

    @property
    def last_layer(self):
        decoder = self.model.decoder
        if hasattr(decoder, "module"):
            decoder = decoder.module
        return decoder.conv_out.weight

    def extract_latents(self, x):
        """(B, 3, H, W) in [-1, 1] -> (B, T) long token indices, no gradient."""
        with torch.no_grad():
            _, _, indices = self.model.encode(x)
            return indices.reshape(x.shape[0], -1)

    def latent_to_image(self, latents):
        """(B, T) long -> (B, 3, H, W) in [-1, 1]."""
        h, w = self.latent_resolution
        shape = (latents.shape[0], self.embed_dim, h, w)
        z_q = self.model.quantize.get_codebook_entry(
            latents.reshape(-1).to(next(self.model.parameters()).device), shape
        )
        return self.model.decode(z_q).clamp(-1, 1)

    def standard_reconstruction(self, x):
        return self.latent_to_image(self.extract_latents(x))
