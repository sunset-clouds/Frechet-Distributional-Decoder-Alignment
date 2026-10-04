import logging
import torch
from torch import nn
import timm
from timm.data import resolve_data_config
import torch.nn.functional as F

from models.FDloss.registry import get_extractor_config
from models.FDloss.feature_extractor_util import InceptionV3

logger = logging.getLogger("FD_loss")

class FeatureExtractor(nn.Module):
    def __init__(self, extractor_name, device="cuda"):
        super(FeatureExtractor, self).__init__()
        self.extractor_name = extractor_name
        self.config = get_extractor_config(extractor_name)
        self.model_name = self.config.model_name
        self.target_size = self.config.target_size
        self.pool_type = self.config.pool_type
        self.model, _, _, _ = load_repr_model(name=self.model_name, device=device, target_size=self.target_size)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, x):
        ## x should be [-1, 1]
        x = x * 0.5 + 0.5 ##[-1, 1] to [0, 1]
        self.model.eval()

        cls_embed, avg_embed = self.model(x)

        if self.pool_type == "cls":
            return cls_embed
        if self.pool_type == "avg":
            return avg_embed

    def eval_extractor(self, x):
        ## x should be [-1, 1]
        x = x * 0.5 + 0.5 ##[-1, 1] to [0, 1]
        self.model.eval()

        cls_embed, feats_or_logits = self.model(x)
        return cls_embed, feats_or_logits

INCEPTION_URL = (
    "https://github.com/toshas/torch-fidelity/releases/download/"
    "v0.2.0/weights-inception-2015-12-05-6726825d.pth"
)

def load_inception(device="cuda", normalize=True):
    model = InceptionV3(normalize=normalize)
    state = torch.hub.load_state_dict_from_url(INCEPTION_URL, progress=True)
    model.load_state_dict(state)
    model.to(device).eval().requires_grad_(False)
    return model

def _preprocess(x, mean, std, target_size=None):
    """[0, 1] float -> resize to target_size -> normalize with mean / std."""
    if target_size is not None and (x.shape[-2] != target_size or x.shape[-1] != target_size):
        x = F.interpolate(x, size=(target_size, target_size), mode="bicubic",
                        align_corners=False, antialias=True)
    return (x - mean) / std


class TimmReprModel(torch.nn.Module):
    """Wraps a timm model as a frozen feature extractor.

    Handles preprocessing: [0, 1] -> resize -> normalize with the model's mean / std.
    Returns ``(cls_token, mean_token)``; CNN models return ``(spatially pooled features, None)``.
    """

    def __init__(self, model_name: str, device="cuda", target_size: int | None = None):
        super().__init__()

        logger.info(f"[TimmReprModel] Loading model: {model_name}")
        # dynamic_img_size/pad only supported by ViT-like models
        kwargs = dict(pretrained=True, num_classes=0)
        try:
            self.model = timm.create_model(model_name, dynamic_img_size=True, dynamic_img_pad=True, **kwargs)
        except TypeError:
            self.model = timm.create_model(model_name, **kwargs)
        self.model.to(device).eval().requires_grad_(False)
        self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 0)
        self.has_attn_pool = hasattr(self.model, "attn_pool") and self.model.attn_pool is not None
        self.feat_dim = self.model.num_features

        data_cfg = resolve_data_config(self.model.pretrained_cfg)
        native_size = data_cfg["input_size"][-1]  # (C, H, W) -> W
        if "naflex" in model_name.lower():
            native_size = 256
        if target_size is not None and target_size != native_size:
            self.target_size = target_size
            logger.info(f"[TimmReprModel] Overriding target_size: {native_size} -> {target_size}")
        else:
            self.target_size = native_size

        mean = torch.tensor(data_cfg["mean"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(data_cfg["std"], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        interpolation = data_cfg.get("interpolation", "bicubic")
        logger.info(
            f"[TimmReprModel] {model_name}: feat_dim={self.feat_dim}, "
            f"target_size={self.target_size}, interpolation={interpolation}, "
            f"mean={data_cfg['mean']}, std={data_cfg['std']}"
        )

    def forward(self, x: torch.Tensor):
        x = _preprocess(x, self.mean, self.std, self.target_size)
        feats = self.model.forward_features(x)
        # CNN models return (B, C, H, W); pool spatially
        if feats.ndim == 4:
            cls_token = feats.mean(dim=[2, 3])
            return cls_token, None
        # ViT models return (B, N, C)
        patch_tokens = feats[:, self.num_prefix_tokens :]
        mean_token = patch_tokens.mean(1)
        if self.num_prefix_tokens > 0:
            cls_token = feats[:, 0]
        elif self.has_attn_pool:
            pool = getattr(self.model, "pool", None) or getattr(self.model, "_pool", None)
            cls_token = pool(feats)
        else:
            cls_token = mean_token
        return cls_token, mean_token

def load_repr_model(name: str, device="cuda", target_size: int | None = None):
    """Load a representation feature extractor.

    Args:
        name: ``'inception'`` (299 input, with logits), ``'convnext'`` (ConvNeXt-v2 base, 224 input)
            or any timm model name.
        target_size: input resolution of a timm model; None = its pretrained_cfg size.

    Returns:
        (model, feat_dim, has_logits, target_size)
    """
    if name == "inception":
        net = load_inception(device=device, normalize=False)
        return net, 2048, True, 299
    elif name == "convnext":
        net = TimmReprModel("convnextv2_base.fcmae_ft_in22k_in1k", device=device, target_size=224)
        return net, net.feat_dim, False, net.target_size
    else:
        net = TimmReprModel(name, device=device, target_size=target_size)
        return net, net.feat_dim, False, net.target_size