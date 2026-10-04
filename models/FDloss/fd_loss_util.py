import logging
import torch
import numpy as np
import torch.distributed

logger = logging.getLogger(__name__)

class FeatureQueue(torch.nn.Module):
    """Circular buffer of (size, feat_dim) features, or running EMA moments when ema_beta > 0."""

    def __init__(
        self,
        size: int = 50000,
        feat_dim: int = 2048,
        online_accum: bool = False,
        ema_beta: float = 0.0,
    ):
        super().__init__()
        self.size = size
        self.feat_dim = feat_dim
        self.online_accum = online_accum
        self.ema_beta = ema_beta
        self.ema_stats = ema_beta > 0.0

        if self.ema_stats:
            self.register_buffer("mu_ema", torch.zeros(feat_dim, dtype=torch.float64))
            self.register_buffer("m2_ema", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))
            self.register_buffer("_ema_count", torch.zeros(1, dtype=torch.long))
        else:
            self.register_buffer("feats", torch.empty(size, feat_dim))
            self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
            if online_accum and size > 0:
                self.register_buffer("feat_sum_old", torch.zeros(feat_dim, dtype=torch.float64))
                self.register_buffer("feat_outer_old", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))

    @property
    def pointer(self) -> int:
        return int(self.ptr.item())

    @torch.no_grad()
    def _init_accumulators(self):
        """Compute feat_sum_old and feat_outer_old from current queue contents."""
        feats_d = self.feats.double()
        self.feat_sum_old.copy_(feats_d.sum(0))
        self.feat_outer_old.copy_(feats_d.T @ feats_d)

    @torch.no_grad()
    def accumulate_batch(self, feats: torch.Tensor):
        """Add a (B, feat_dim) batch to the streaming sums used by EMA init."""
        feats_d = feats.detach().float().double()
        self.mu_ema.add_(feats_d.sum(0))
        self.m2_ema.addmm_(feats_d.T, feats_d)
        self._ema_count += feats_d.shape[0]

    @torch.no_grad()
    def _finalize_streaming_init(self):
        """Normalize accumulated sums into moments (mu, E[xx^T])."""
        count = self._ema_count.item()
        if count == 0:
            logger.warning("[FeatureQueue] EMA streaming init: no features accumulated")
            return
        self.mu_ema.div_(count)
        self.m2_ema.div_(count)
        logger.info(f"[FeatureQueue] EMA init done: {count} features (beta={self.ema_beta})")

    def build_feats_stats(self, new_feats: torch.Tensor):
        """(B, feat_dim) -> (mu, sigma) with gradient through new_feats; EMA moments if enabled,
        else the online accumulators."""
        if self.ema_stats:
            return self._build_feats_stats_ema(new_feats)

        new_d = new_feats.double()
        B = new_d.shape[0]
        N = self.size

        evicted = self._get_evicted_feats(B).double()
        sum_old = self.feat_sum_old - evicted.sum(0)
        outer_old = self.feat_outer_old - evicted.T @ evicted

        feat_sum = sum_old.detach() + new_d.sum(0)
        feat_outer = outer_old.detach() + new_d.T @ new_d

        mu = feat_sum / N
        sigma = (feat_outer - feat_sum.unsqueeze(1) * feat_sum.unsqueeze(0) / N) / (N - 1)
        return mu, sigma

    def _build_feats_stats_ema(self, new_feats: torch.Tensor):
        """(mu, sigma) from the EMA moments blended with new_feats."""
        beta = self.ema_beta
        new_d = new_feats.double()
        B = new_d.shape[0]

        mu = beta * self.mu_ema.detach() + (1.0 - beta) * new_d.mean(0)
        m2 = beta * self.m2_ema.detach() + (1.0 - beta) * (new_d.T @ new_d) / B

        sigma = m2 - mu.unsqueeze(1) * mu.unsqueeze(0)
        return mu, sigma

    def _snapshot(self, buf: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        """Detached copy of *buf* with the pointer region replaced by *new*."""
        if self.size == 0:
            return new
        n = new.shape[0]
        snap = buf.clone().detach()
        ptr = self.pointer
        if ptr + n <= self.size:
            snap[ptr : ptr + n] = new
        else:
            first = self.size - ptr
            snap[ptr : self.size] = new[:first]
            snap[: n - first] = new[first:]
        return snap

    def build_feats_snapshot(self, new_feats: torch.Tensor) -> torch.Tensor:
        """Return (size, feat_dim) with the pointer region carrying autograd."""
        return self._snapshot(self.feats, new_feats)

    @torch.no_grad()
    def enqueue(self, new_feats: torch.Tensor):
        """Overwrite the oldest entries with detached new_feats; EMA mode updates the running
        moments instead. No-op when size=0."""
        if self.size == 0:
            return

        n = new_feats.shape[0]
        new_det = new_feats.detach().float()

        if self.ema_stats:
            beta = self.ema_beta
            new_d = new_det.double()
            self.mu_ema.mul_(beta).add_(new_d.mean(0), alpha=1.0 - beta)
            self.m2_ema.mul_(beta).addmm_(new_d.T, new_d, alpha=(1.0 - beta) / n)
            return

        ptr = self.pointer

        if self.online_accum:
            evicted = self._get_evicted_feats(n).double()
            new_d = new_det.double()
            self.feat_sum_old.add_(new_d.sum(0) - evicted.sum(0))
            self.feat_outer_old.add_(new_d.T @ new_d - evicted.T @ evicted)

        if ptr + n <= self.size:
            self.feats[ptr : ptr + n] = new_det
        else:
            first = self.size - ptr
            self.feats[ptr : self.size] = new_det[:first]
            self.feats[: n - first] = new_det[first:]
        self.ptr[0] = (ptr + n) % self.size

    def _get_evicted_feats(self, n: int) -> torch.Tensor:
        """(n, feat_dim) entries the next enqueue of size n overwrites."""
        ptr = self.pointer
        if ptr + n <= self.size:
            return self.feats[ptr : ptr + n]
        first = self.size - ptr
        return torch.cat([self.feats[ptr : self.size], self.feats[: n - first]], dim=0)

class _DiffAllGather(torch.autograd.Function):
    """All-gather along dim 0 with gradient for the local rank's chunk only."""

    @staticmethod
    def forward(ctx, tensor):
        world_size = torch.distributed.get_world_size()
        ctx.rank = torch.distributed.get_rank()
        ctx.batch_size = tensor.shape[0]
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, tensor.contiguous())
        gathered[ctx.rank] = tensor  # preserve local autograd graph
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        chunk = ctx.batch_size
        return grad_output[ctx.rank * chunk : (ctx.rank + 1) * chunk].contiguous()


def diff_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """(B, D) -> (world_size * B, D); identity when not distributed."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1):
        return tensor
    return _DiffAllGather.apply(tensor)


def precompute_sigma_ref_sqrt(sigma_ref: torch.Tensor) -> torch.Tensor:
    """sigma_ref^{1/2} via eigendecomposition."""
    eigvals, eigvecs = torch.linalg.eigh(sigma_ref)
    eigvals = torch.clamp(eigvals, min=0)
    return eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T


def _compute_trace_term(
    sigma: torch.Tensor,
    sigma_ref: torch.Tensor,
    sigma_ref_sqrt: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """tr(sigma) + tr(sigma_ref) - 2*tr(sqrtm(sigma @ sigma_ref)); without sigma_ref_sqrt, None if
    sigma @ sigma_ref is not finite."""
    if sigma_ref_sqrt is not None:
        # tr(sqrtm(sigma @ sigma_ref)) = sum sqrt(eigvalsh(sigma_ref^{1/2} sigma sigma_ref^{1/2}))
        M = sigma_ref_sqrt @ sigma @ sigma_ref_sqrt
        M = 0.5 * (M + M.T)
        evals = torch.linalg.eigvalsh(M)
        evals = torch.clamp(evals, min=0)
        tr_covmean = torch.sum(torch.sqrt(evals))
    else:
        product = sigma @ sigma_ref
        if not torch.isfinite(product).all():
            return None
        eigvals = torch.linalg.eigvals(product).real
        eigvals = torch.clamp(eigvals, min=0)
        tr_covmean = torch.sum(torch.sqrt(eigvals))

    return torch.diagonal(sigma).sum() + torch.diagonal(sigma_ref).sum() - 2.0 * tr_covmean


def compute_frechet_distance_loss(
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    all_feats: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
    sigma_ref_sqrt: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable Frechet distance from ``all_feats`` (N, D), N >= 2, or from ``mu`` and ``sigma``.
    -> fp32 scalar."""
    if all_feats is not None:
        n_samples = all_feats.shape[0]
        if n_samples < 2:
            logger.warning(f"[compute_frechet_distance_loss] Only {n_samples} sample(s) — need >= 2")
            return torch.tensor(1e6, device=all_feats.device, dtype=torch.float32, requires_grad=True)
        mu = all_feats.mean(dim=0)
        feats_c = all_feats - mu
        sigma = (feats_c.T @ feats_c) / (n_samples - 1)
    elif mu is None or sigma is None:
        raise ValueError("Provide either all_feats or both mu and sigma")

    compute_dtype = sigma.dtype
    mu_ref = mu_ref.to(dtype=compute_dtype)
    sigma_ref = sigma_ref.to(dtype=compute_dtype)
    if sigma_ref_sqrt is not None:
        sigma_ref_sqrt = sigma_ref_sqrt.to(dtype=compute_dtype)

    diff = mu - mu_ref
    mean_term = diff.dot(diff)

    trace_term = _compute_trace_term(sigma, sigma_ref, sigma_ref_sqrt)
    if trace_term is None:
        device = all_feats.device if all_feats is not None else mu.device
        logger.warning("[compute_frechet_distance_loss] NaN/Inf in covariance product — returning fallback")
        return torch.tensor(1e6, device=device, dtype=torch.float32)

    return (mean_term + trace_term).float()


def load_mu_and_sigma_reference(fid_stats_path: str, pool_type: str = "cls"):
    """.npz with ``mu``/``sigma`` (and ``avg_mu``/``avg_sigma`` for pool_type 'avg') ->
    (mu_ref, sigma_ref) as CUDA float64."""
    ref = np.load(fid_stats_path)
    if pool_type == "avg":
        if "avg_mu" not in ref:
            raise KeyError(
                f"pool_type='avg' but {fid_stats_path} has no 'avg_mu'. "
                f"Available: {list(ref.keys())}"
            )
        mu_ref = torch.tensor(ref["avg_mu"], device="cuda", dtype=torch.float64)
        sigma_ref = torch.tensor(ref["avg_sigma"], device="cuda", dtype=torch.float64)
    else:
        mu_ref = torch.tensor(ref["mu"], device="cuda", dtype=torch.float64)
        sigma_ref = torch.tensor(ref["sigma"], device="cuda", dtype=torch.float64)
    return mu_ref, sigma_ref