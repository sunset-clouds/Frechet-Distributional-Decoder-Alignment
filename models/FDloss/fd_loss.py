import os
import torch
from torch import nn
from torch import distributed as dist

from models.FDloss.feature_extractor import FeatureExtractor
from models.FDloss.registry import get_fd_loss_default, get_extractor_config
from models.FDloss.fd_loss_util import (
    FeatureQueue,
    compute_frechet_distance_loss,
    diff_all_gather,
    precompute_sigma_ref_sqrt,
    load_mu_and_sigma_reference,
)


class _Space(nn.Module):
    """One feature space: a frozen extractor, its reference moments, and its own feature queue."""

    def __init__(self, extractor_name, fd_loss_type, fid_stats_dir, ema_beta, queue_size,
                 use_sigma_ref_sqrt, online_accumulate, weight):
        super().__init__()
        config = get_extractor_config(extractor_name)
        self.name = extractor_name
        self.weight = weight
        self.pool_type = config.pool_type
        self.feat_dim = config.feat_dim
        self.extractor = FeatureExtractor(extractor_name=extractor_name)

        mu_ref, sigma_ref = load_mu_and_sigma_reference(
            fid_stats_path=os.path.join(fid_stats_dir, config.train_stats_path),
            pool_type=config.pool_type,
        )
        self.register_buffer("mu_ref", mu_ref)
        self.register_buffer("sigma_ref", sigma_ref)
        self.register_buffer("sigma_ref_sqrt",
                             precompute_sigma_ref_sqrt(sigma_ref) if use_sigma_ref_sqrt else None)

        self.queue = FeatureQueue(size=queue_size, feat_dim=config.feat_dim,
                                  online_accum=online_accumulate,
                                  ema_beta=ema_beta if fd_loss_type == "ema" else 0.0)

    def features(self, x):
        """(B, 3, H, W) in [-1, 1] -> (B, feat_dim) fp32."""
        return self.extractor(x).float()

    def raw_fd(self, all_feats, online_accumulate, fd_loss_type):
        if online_accumulate or fd_loss_type == "ema":
            mu, sigma = self.queue.build_feats_stats(all_feats)
            return compute_frechet_distance_loss(mu_ref=self.mu_ref, sigma_ref=self.sigma_ref,
                                                 mu=mu, sigma=sigma,
                                                 sigma_ref_sqrt=self.sigma_ref_sqrt)
        snapshot = self.queue.build_feats_snapshot(all_feats)
        return compute_frechet_distance_loss(mu_ref=self.mu_ref, sigma_ref=self.sigma_ref,
                                             all_feats=snapshot,
                                             sigma_ref_sqrt=self.sigma_ref_sqrt)


class FrechetLoss(nn.Module):
    """Frechet distance to the training reference statistics, summed over one or more feature spaces.

    ``forward(x)``: (B, 3, H, W) in [-1, 1] -> scalar, with per space
        normalized_j = raw_j / (raw_j.detach() + eps)
        L_FD = world_size * sum_j w_j * normalized_j
    """

    def __init__(self, extractor_names, fd_loss_type, fid_stats_dir, ema_beta=None,
                 queue_size=None, weights=None, norm_eps=0.01):
        super().__init__()
        if isinstance(extractor_names, str):
            extractor_names = [name.strip() for name in extractor_names.split(",") if name.strip()]
        if not extractor_names:
            raise ValueError("FrechetLoss needs at least one extractor name")
        if weights is None:
            weights = [1.0] * len(extractor_names)
        if len(weights) != len(extractor_names):
            raise ValueError(f"got {len(weights)} weights for {len(extractor_names)} extractors")

        self.fd_loss_type = fd_loss_type.lower()
        if self.fd_loss_type not in ("ema", "queue"):
            raise ValueError("fd_loss_type must be either 'ema' or 'queue'")
        self.norm_eps = norm_eps

        self.ema_beta = get_fd_loss_default("ema_beta", ema_beta)
        self.queue_size = get_fd_loss_default("queue_size", queue_size)
        self.use_sigma_ref_sqrt = get_fd_loss_default("use_sigma_ref_sqrt")
        self.online_accumulate = get_fd_loss_default("online_accumulate")

        self.spaces = nn.ModuleList([
            _Space(name, self.fd_loss_type, fid_stats_dir, self.ema_beta, self.queue_size,
                   self.use_sigma_ref_sqrt, self.online_accumulate, weight)
            for name, weight in zip(extractor_names, weights)
        ])
        self.extractor_names = list(extractor_names)
        # un-normalised FD per space from the last forward
        self.last_raw = {}

    def raw_log(self):
        """-> {'fd_inception': tensor, ...} for the training log."""
        short = lambda name: "".join(c for c in name.lower() if c.isalnum()).replace("v3", "") \
                                                                            .replace("v2", "")
        return {f"fd_{short(name)}": value for name, value in self.last_raw.items()}

    def eval_extractors(self):
        for space in self.spaces:
            space.extractor.eval()

    def forward(self, x):
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        total = None
        self.last_raw = {}
        for space in self.spaces:
            embeds = space.features(x)
            with torch.autocast(device_type=embeds.device.type, enabled=False):
                all_feats = diff_all_gather(embeds)
                raw = space.raw_fd(all_feats, self.online_accumulate, self.fd_loss_type)
                normalized = raw / (raw.detach() + self.norm_eps)
                term = space.weight * normalized
                total = term if total is None else total + term
                space.queue.enqueue(all_feats.detach())
            self.last_raw[space.name] = raw.detach()
        return world_size * total

    def updates(self, x, filled):
        """Fill every queue from the same batch. -> the new filled count."""
        new_filled = filled
        for space in self.spaces:
            embeds = space.features(x)
            with torch.autocast(device_type=embeds.device.type, enabled=False):
                all_feats = diff_all_gather(embeds)
                count = min(all_feats.shape[0], self.queue_size - filled)
                if space.queue.ema_stats:
                    space.queue.accumulate_batch(all_feats[:count])
                else:
                    space.queue.feats[filled:filled + count] = all_feats[:count].float()
            new_filled = filled + count
        return new_filled

    def finalize(self):
        for space in self.spaces:
            if space.queue.ema_stats:
                space.queue._finalize_streaming_init()
            else:
                space.queue.ptr.zero_()
                if self.online_accumulate:
                    space.queue._init_accumulators()
