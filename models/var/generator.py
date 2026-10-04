import torch
from torch import nn

from models.var import registry
from models.ckpt_load import load_weights
from models.var.tokenizer_arch import VQVAE
from models.var.generator_arch import VAR, sample_with_top_k_top_p_


class VARGenerator(nn.Module):
    """Frozen VAR next-scale transformer, d16 / d20 / d24; ``standard_generation`` -> (B, 680) long,
    all ten scale maps concatenated. ``_sample`` is upstream's ``autoregressive_infer_cfg`` returning
    indices instead of pixels; ``self.vae`` supplies the codebook and phi for the next-scale input."""

    def __init__(self, generator_name, ckpt_path="", resolution=registry.EVAL_SIZE,
                 cfg_omega=None):
        super().__init__()
        self.generator_name = generator_name
        params = registry.model_params("generator", generator_name)
        registry._check_resolution(generator_name, resolution)
        self.token_length = registry.token_length(resolution, name=generator_name)
        self.cfg_omega = params["cfg_omega"] if cfg_omega is None else float(cfg_omega)

        self.vae = VQVAE(**registry.model_params("tokenizer", "var-vqvae"))
        load_weights(self.vae, registry.ckpt_path("tokenizer", "var-vqvae"), "vqvae for var")
        depth = params["depth"]
        self.model = VAR(vae_local=self.vae, num_classes=1000, depth=depth, embed_dim=depth * 64,
                         num_heads=depth, drop_rate=0., attn_drop_rate=0.,
                         drop_path_rate=0.1 * depth / 24, norm_eps=1e-6, shared_aln=False,
                         cond_drop_rate=0.1, attn_l2_norm=True, patch_nums=registry.PATCH_NUMS,
                         flash_if_available=True, fused_if_available=True)
        load_weights(self.model, ckpt_path or registry.ckpt_path("generator", generator_name),
                     f"generator {generator_name}")
        self.eval_mode()

    def eval_mode(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _sample(self, labels, cfg):
        m, quant = self.model, self.vae.quantize
        B = labels.shape[0]
        cond = m.class_emb(torch.cat([labels, torch.full_like(labels, m.num_classes)]))
        lvl_pos = m.lvl_embed(m.lvl_1L) + m.pos_1LC
        next_token_map = (cond.unsqueeze(1).expand(2 * B, m.first_l, -1)
                          + m.pos_start.expand(2 * B, m.first_l, -1) + lvl_pos[:, :m.first_l])
        f_hat = cond.new_zeros(B, m.Cvae, m.patch_nums[-1], m.patch_nums[-1])
        tokens, cur_l = [], 0
        top_k, top_p = registry.GENERATOR_DEFAULTS["top_k"], registry.GENERATOR_DEFAULTS["top_p"]
        for b in m.blocks:
            b.attn.kv_caching(True)
        try:
            for si, pn in enumerate(m.patch_nums):
                ratio = si / m.num_stages_minus_1
                cur_l += pn * pn
                cond_or_gss = m.shared_ada_lin(cond)
                x = next_token_map
                for b in m.blocks:
                    x = b(x=x, cond_BD=cond_or_gss, attn_bias=None)
                logits = m.get_logits(x, cond)
                t = cfg * ratio
                logits = (1 + t) * logits[:B] - t * logits[B:]
                idx = sample_with_top_k_top_p_(logits, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
                tokens.append(idx)
                h = quant.embedding(idx).transpose_(1, 2).reshape(B, m.Cvae, pn, pn)
                f_hat, next_token_map = quant.get_next_autoregressive_input(
                    si, len(m.patch_nums), f_hat, h)
                if si != m.num_stages_minus_1:
                    next_token_map = next_token_map.view(B, m.Cvae, -1).transpose(1, 2)
                    next_token_map = (m.word_embed(next_token_map)
                                      + lvl_pos[:, cur_l:cur_l + m.patch_nums[si + 1] ** 2])
                    next_token_map = next_token_map.repeat(2, 1, 1)
        finally:
            for b in m.blocks:
                b.attn.kv_caching(False)
        return torch.cat(tokens, dim=1)

    def standard_generation(self, batch_size, labels=None, use_cfg=True, device=None, seeds=None):
        """-> (batch_size, T) long token indices; ``seeds``: one torch.manual_seed per sample."""
        if device is None:
            device = next(self.model.parameters()).device
        if labels is None:
            labels = torch.randint(0, self.model.num_classes, (batch_size,), device=device)
        else:
            labels = labels.to(device)
        # guidance is (1 + cfg * ratio); cfg = 0 is off
        cfg = self.cfg_omega if use_cfg else 0.0
        if seeds is None:
            n = registry.SAMPLE_CHUNK
            return torch.cat([self._sample(labels[i:i + n], cfg) for i in range(0, batch_size, n)])
        out = torch.empty(batch_size, self.token_length, dtype=torch.long, device=device)
        for i, seed in enumerate(seeds):
            torch.manual_seed(int(seed))
            out[i] = self._sample(labels[i:i + 1], cfg)[0]
        return out
