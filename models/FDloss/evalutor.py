"""Frechet distance evaluator for any registered extractor; Inception Score when the extractor
has logits (InceptionV3)."""

import os

import numpy as np
import torch
import torch.distributed as tdist
from models.FDloss.feature_extractor import FeatureExtractor
from models.FDloss.registry import get_extractor_config

class FDEvaluator:
    """Accumulates feature moments over an evaluation; ``finalize`` -> {fd, inception_score, num_images}."""
    def __init__(self, extractor_name, fid_stats_dir, eval_mode="reconstruction"):
        self.repr_model = FeatureExtractor(extractor_name)
        self.config = get_extractor_config(extractor_name)
        self.feat_dim = self.config.feat_dim
        self.has_logits = self.config.is_inception
        self.pool_type = self.config.pool_type
        self.device = torch.device("cuda")

        self.ref_stats_path = self.config.rec_stats_path if eval_mode=="reconstruction" else self.config.gen_stats_path        
        ref = np.load(os.path.join(fid_stats_dir, self.ref_stats_path))
        self.ref_mu = ref["mu"].astype(np.float64)
        self.ref_sigma = ref["sigma"].astype(np.float64)
        self.reset()

    def reset(self):
        """Clear accumulators for a new evaluation run."""
        self.feat_sum = torch.zeros(self.feat_dim, dtype=torch.float64, device=self.device)
        self.feat_outer = torch.zeros(self.feat_dim, self.feat_dim, dtype=torch.float64, device=self.device)
        self.count = 0
        self._logits: list[torch.Tensor] = []

    @torch.inference_mode()
    def update(self, images: torch.Tensor):
        """(B, 3, H, W) in [-1, 1] -> accumulate feature sums (and logits)."""
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            feats, feats_or_logits = self.repr_model.eval_extractor(images)
        self._accumulate(feats, feats_or_logits)

    @torch.inference_mode()
    def _accumulate(self, feats: torch.Tensor, logits: torch.Tensor | None):
        feats64 = feats.double()
        self.feat_sum.add_(feats64.sum(0))
        self.feat_outer.addmm_(feats64.T, feats64)
        self.count += feats.shape[0]
        if self.has_logits and logits is not None:
            self._logits.append(logits.cpu())

    def _aggregate(self):
        """Reduce sufficient statistics and gather logits to rank 0."""
        distributed = tdist.is_available() and tdist.is_initialized() and tdist.get_world_size() > 1
        if not distributed:
            return

        rank = tdist.get_rank()
        world_size = tdist.get_world_size()

        tdist.reduce(self.feat_sum, dst=0, op=tdist.ReduceOp.SUM)
        tdist.reduce(self.feat_outer, dst=0, op=tdist.ReduceOp.SUM)
        count_t = torch.tensor([self.count], dtype=torch.long, device=self.device)
        tdist.reduce(count_t, dst=0, op=tdist.ReduceOp.SUM)
        self.count = count_t.item()

        if not self.has_logits or not self._logits:
            return

        local_logits = torch.cat(self._logits, dim=0).to(self.device)
        gathered = gather_features(local_logits, world_size, rank, self.device)
        if rank == 0:
            self._logits = [gathered.cpu()]
        else:
            self._logits = []

    def _compute_metrics(self):
        N = self.count
        s = self.feat_sum.cpu().numpy()
        S = self.feat_outer.cpu().numpy()
        mu = (s / N).astype(np.float64)
        sigma = ((S - np.outer(s, s) / N) / (N - 1)).astype(np.float64)

        fd = compute_fid(mu, sigma, self.ref_mu, self.ref_sigma)

        inception_score = None
        if self.has_logits and self._logits:
            logits = torch.cat(self._logits, dim=0)
            inception_score, _ = compute_isc(logits)

        return {"fd": fd, "inception_score": inception_score, "num_images": N}

    def finalize(self) -> dict:
        """Reduce across ranks, compute on rank 0, broadcast -> {fd, inception_score, num_images}."""
        self._aggregate()
        distributed = tdist.is_available() and tdist.is_initialized() and tdist.get_world_size() > 1

        if not distributed:
            return self._compute_metrics()

        rank = tdist.get_rank()
        if rank == 0:
            metrics = self._compute_metrics()
            is_val = metrics["inception_score"]
            if is_val is None:
                is_val = -1.0
            buf = torch.tensor(
                [metrics["fd"], is_val, float(metrics["num_images"])],
                dtype=torch.float64, device=self.device,
            )
        else:
            buf = torch.zeros(3, dtype=torch.float64, device=self.device)

        tdist.broadcast(buf, src=0)
        is_val = buf[1].item()
        return {
            "fd": buf[0].item(),
            "inception_score": is_val if is_val >= 0 else None,
            "num_images": int(buf[2].item()),
        }

def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Frechet distance between two Gaussians (matches OpenAI guided-diffusion evaluator)."""
    from scipy import linalg

    mu1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    mu2 = np.atleast_1d(np.asarray(mu2, dtype=np.float64))
    sigma1 = np.atleast_2d(np.asarray(sigma1, dtype=np.float64))
    sigma2 = np.atleast_2d(np.asarray(sigma2, dtype=np.float64))

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))

def compute_isc(logits, splits=10, rng_seed=2020):
    """Inception score (matches torch_fidelity formula). Returns (mean, std)."""
    if logits.dim() != 2:
        raise ValueError(f"Expected 2D logits tensor, got {logits.dim()}D")
    N = logits.shape[0]

    rng = np.random.RandomState(rng_seed)
    logits = logits[rng.permutation(N)].double()
    p = logits.softmax(dim=1)
    log_p = logits.log_softmax(dim=1)

    scores = []
    for i in range(splits):
        lo = i * N // splits
        hi = (i + 1) * N // splits
        p_chunk = p[lo:hi]
        log_p_chunk = log_p[lo:hi]
        q = p_chunk.mean(dim=0, keepdim=True)
        kl = (p_chunk * (log_p_chunk - q.log())).sum(1).mean().exp().item()
        scores.append(kl)
    return float(np.mean(scores)), float(np.std(scores))

def gather_features(
    local_feats: torch.Tensor,
    world_size: int,
    rank: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Gather variable-length (n_i, D) tensors to rank 0 -> (sum n_i, D) there, None elsewhere."""
    local_n_t = torch.tensor([local_feats.shape[0]], dtype=torch.long, device=device)
    all_n = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    tdist.all_gather(all_n, local_n_t)
    max_n = max(t.item() for t in all_n)
    if local_feats.shape[0] < max_n:
        pad = torch.zeros(max_n - local_feats.shape[0], local_feats.shape[1],
                        dtype=local_feats.dtype, device=device)
        local_feats = torch.cat([local_feats, pad], dim=0)
    if rank == 0:
        gathered = [torch.zeros_like(local_feats) for _ in range(world_size)]
        tdist.gather(local_feats, gather_list=gathered, dst=0)
        trimmed = [g[:n.item()] for g, n in zip(gathered, all_n)]
        return torch.cat(trimmed, dim=0)
    else:
        tdist.gather(local_feats, dst=0)
        return None
