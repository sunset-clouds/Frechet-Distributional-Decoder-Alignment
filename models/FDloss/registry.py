import os
from typing import Optional

FD_LOSS_DEFAULTS = {
    "ema_beta": 0.999,
    "queue_size": 50000,
    "use_sigma_ref_sqrt": True,
    "online_accumulate": False,
}

# valfd: FD between the ImageNet validation set and the generation reference.
# FD_r = FD / valfd; FDr6 is the mean of FD_r over the six extractors.


class ExtractorConfig:
    def __init__(
        self,
        model_name: str,
        feat_dim: int,
        target_size: int,
        gen_stats_path: str,
        rec_stats_path: str,
        train_stats_path: str,
        valfd: float,
        pool_type: str = "cls",
        is_inception: bool = False,
    ):
        self.model_name = model_name
        self.feat_dim = feat_dim
        self.target_size = target_size
        self.gen_stats_path = gen_stats_path
        self.rec_stats_path = rec_stats_path
        self.train_stats_path = train_stats_path
        self.valfd = valfd
        self.pool_type = pool_type
        self.is_inception = is_inception

    @property
    def stats_path(self):
        """Alias of ``gen_stats_path``."""
        return self.gen_stats_path

_GEN_STATS_ROOT = os.path.join("generation", "fid_stats")
_REC_STATS_ROOT = os.path.join("reconstruction", "fid_stats")
_TRAIN_STATS_ROOT = os.path.join("training", "fid_stats")

EXTRACTOR_REGISTRY = {
    "Inception-v3": ExtractorConfig(
        model_name="inception",
        feat_dim=2048,
        target_size=299,
        gen_stats_path=os.path.join(_GEN_STATS_ROOT, "gen_inception_stats.npz"),
        rec_stats_path=os.path.join(_REC_STATS_ROOT, "rec_inception_stats.npz"),
        train_stats_path=os.path.join(_TRAIN_STATS_ROOT, "train_inception_stats.npz"),
        valfd=1.6793,
        is_inception=True,
        pool_type = "cls",
    ),
    "ConvNeXt-v2": ExtractorConfig(
        model_name="convnext",
        feat_dim=1024,
        target_size=224,
        gen_stats_path=os.path.join(_GEN_STATS_ROOT, "gen_convnext_in256_t224_stats.npz"),
        rec_stats_path=os.path.join(_REC_STATS_ROOT, "rec_convnext_in256_t224_stats.npz"),
        train_stats_path=os.path.join(_TRAIN_STATS_ROOT, "train_convnext_in256_t224_stats.npz"),
        valfd=56.8005,
        is_inception=False,
        pool_type = "cls",
    ),
    "DINOv2": ExtractorConfig(
        model_name="vit_large_patch14_dinov2.lvd142m",
        feat_dim=1024,
        target_size=256,
        gen_stats_path=os.path.join(_GEN_STATS_ROOT,"gen_vit_large_patch14_dinov2_lvd142m_in256_t256_stats.npz"),
        rec_stats_path=os.path.join(_REC_STATS_ROOT,"rec_vit_large_patch14_dinov2_lvd142m_in256_t256_stats.npz"),
        train_stats_path=os.path.join(_TRAIN_STATS_ROOT,"train_vit_large_patch14_dinov2_lvd142m_in256_t256_stats.npz"),
        valfd=14.1843,
        is_inception=False,
        pool_type = "cls",
    ),
    "MAE": ExtractorConfig(
        model_name="vit_large_patch16_224.mae",
        feat_dim=1024,
        target_size=224,
        gen_stats_path=os.path.join(_GEN_STATS_ROOT,"gen_vit_large_patch16_224_mae_in256_t224_stats.npz"),
        rec_stats_path=os.path.join(_REC_STATS_ROOT,"rec_vit_large_patch16_224_mae_in256_t224_stats.npz"),
        train_stats_path=os.path.join(_TRAIN_STATS_ROOT,"train_vit_large_patch16_224_mae_in256_t224_stats.npz"),
        valfd=0.0424,
        is_inception=False,
        pool_type = "cls",
    ),
    "SigLIP2": ExtractorConfig(
        model_name="vit_so400m_patch16_siglip_256.v2_webli",
        feat_dim=1152,
        target_size=224,
        gen_stats_path=os.path.join(_GEN_STATS_ROOT,"gen_vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz"),
        rec_stats_path=os.path.join(_REC_STATS_ROOT,"rec_vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz"),
        train_stats_path=os.path.join(_TRAIN_STATS_ROOT,"train_vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz"),
        valfd=0.6051,
        is_inception=False,
        pool_type = "cls",
    ),
    "CLIP": ExtractorConfig(
        model_name="vit_large_patch14_clip_224.openai",
        feat_dim=1024,
        target_size=256,
        gen_stats_path=os.path.join(_GEN_STATS_ROOT,"gen_vit_large_patch14_clip_224_openai_in256_t256_stats.npz"),
        rec_stats_path=os.path.join(_REC_STATS_ROOT,"rec_vit_large_patch14_clip_224_openai_in256_t256_stats.npz"),
        train_stats_path=os.path.join(_TRAIN_STATS_ROOT,"train_vit_large_patch14_clip_224_openai_in256_t256_stats.npz"),
        valfd=5.4776,
        is_inception=False,
        pool_type = "cls",
    ),
}


def get_extractor_config(extractor_name: str) -> ExtractorConfig:
    if extractor_name not in EXTRACTOR_REGISTRY:
        names = ", ".join(EXTRACTOR_REGISTRY)
        raise ValueError(f"Unsupported extractor_name '{extractor_name}'. Available: {names}")
    return EXTRACTOR_REGISTRY[extractor_name]

def get_fd_loss_default(name: str, override: Optional[object] = None):
    if override is not None:
        return override
    if name not in FD_LOSS_DEFAULTS:
        raise KeyError(f"Unknown FD loss default '{name}'")
    return FD_LOSS_DEFAULTS[name]
