import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as tvf

from models.FDloss.feature_extractor import FeatureExtractor
from models.FDloss.registry import get_extractor_config


REPRESENTATION_EXTRACTORS = (
    ("Incep.", "Inception-v3"),
    ("ConvNeXt", "ConvNeXt-v2"),
    ("DINOv2", "DINOv2"),
    ("MAE", "MAE"),
    ("SigLIP", "SigLIP2"),
    ("CLIP", "CLIP"),
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


class FlatImageDataset(Dataset):
    def __init__(self, image_dir, resolution=256):
        self.image_dir = Path(image_dir)
        self.resolution = int(resolution)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        self.paths = sorted(
            path for path in self.image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            if image.size != (self.resolution, self.resolution):
                raise ValueError(
                    f"Expected {self.resolution}x{self.resolution} images, got "
                    f"{image.size} at {self.paths[index]}"
                )
            return tvf.to_tensor(image).mul(2).sub(1)


def frechet_distance(mu, sigma, ref_mu, ref_sigma, eps=1e-6):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    ref_mu = np.asarray(ref_mu, dtype=np.float64)
    ref_sigma = np.asarray(ref_sigma, dtype=np.float64)
    difference = mu - ref_mu
    covariance_mean, _ = linalg.sqrtm(sigma.dot(ref_sigma), disp=False)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(sigma.shape[0]) * eps
        covariance_mean = linalg.sqrtm((sigma + offset).dot(ref_sigma + offset))
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diagonal(covariance_mean).imag, 0, atol=1e-3):
            raise ValueError(f"FID covariance has imaginary component {np.abs(covariance_mean.imag).max()}")
        covariance_mean = covariance_mean.real
    return float(
        difference.dot(difference)
        + np.trace(sigma)
        + np.trace(ref_sigma)
        - 2.0 * np.trace(covariance_mean)
    )


def inception_score(logits, splits=10, seed=2020):
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits, got shape {tuple(logits.shape)}")
    if logits.shape[0] == 0 or splits <= 0:
        raise ValueError("Inception score needs nonempty logits and positive splits")
    splits = min(splits, logits.shape[0])
    generator = np.random.RandomState(seed)
    order = torch.from_numpy(generator.permutation(logits.shape[0])).long()
    logits = logits[order].double()
    probabilities = logits.softmax(dim=1)
    log_probabilities = logits.log_softmax(dim=1)
    scores = []
    for split in range(splits):
        start = split * logits.shape[0] // splits
        end = (split + 1) * logits.shape[0] // splits
        probability = probabilities[start:end]
        log_probability = log_probabilities[start:end]
        marginal = probability.mean(dim=0, keepdim=True)
        score = (probability * (log_probability - marginal.log())).sum(dim=1).mean().exp()
        scores.append(float(score))
    return float(np.mean(scores)), float(np.std(scores))


def _distributed_context():
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-representation evaluation requires a CUDA GPU")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)
    return 0, 1, torch.device("cuda", local_rank)


def _reduce_statistics(feature_sum, feature_outer, count, rank, world_size):
    if world_size > 1:
        dist.reduce(feature_sum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(feature_outer, dst=0, op=dist.ReduceOp.SUM)
        count_tensor = torch.tensor([count], dtype=torch.long, device=feature_sum.device)
        dist.reduce(count_tensor, dst=0, op=dist.ReduceOp.SUM)
        count = int(count_tensor.item())
    if rank != 0:
        return None, None, count
    summed = feature_sum.cpu().numpy()
    outer = feature_outer.cpu().numpy()
    mu = summed / count
    sigma = (outer - np.outer(summed, summed) / count) / (count - 1)
    return mu.astype(np.float64), sigma.astype(np.float64), count


def _gather_logits(local_logits, device, rank, world_size):
    local_logits = torch.cat(local_logits, dim=0).to(device=device, dtype=torch.float32)
    if world_size == 1:
        return local_logits.cpu()

    local_count = torch.tensor([local_logits.shape[0]], dtype=torch.long, device=device)
    counts = [torch.zeros_like(local_count) for _ in range(world_size)]
    dist.all_gather(counts, local_count)
    counts = [int(value.item()) for value in counts]
    max_count = max(counts)
    if local_logits.shape[0] < max_count:
        padding = torch.zeros(
            max_count - local_logits.shape[0], local_logits.shape[1],
            dtype=local_logits.dtype, device=device,
        )
        local_logits = torch.cat([local_logits, padding], dim=0)
    gathered = [torch.empty_like(local_logits) for _ in range(world_size)]
    dist.all_gather(gathered, local_logits)
    if rank != 0:
        return None
    return torch.cat([tensor[:count].cpu() for tensor, count in zip(gathered, counts)], dim=0)


def _stats_path(extractor_name, stats_root):
    configured = Path(get_extractor_config(extractor_name).gen_stats_path)
    path = Path(stats_root) / configured
    # Also accept an explicitly supplied flat directory of generation statistics.
    flat_path = Path(stats_root) / configured.name
    return path if path.is_file() or not flat_path.is_file() else flat_path


@torch.inference_mode()
def evaluate_representations(
    image_dir,
    output_json,
    output_csv,
    stats_root="reference_stats",
    batch_size=64,
    workers=8,
    expected_samples=50000,
    allow_nonstandard_sample_count=False,
):
    rank, world_size, device = _distributed_context()
    dataset = FlatImageDataset(image_dir)
    if expected_samples != 50000 and not allow_nonstandard_sample_count:
        raise ValueError(
            "Nonstandard sample counts are smoke tests. Pass "
            "--allow_nonstandard_sample_count explicitly."
        )
    if len(dataset) != expected_samples and not allow_nonstandard_sample_count:
        raise ValueError(
            f"The standard protocol requires exactly {expected_samples} images; found {len(dataset)}. "
            "Use --allow_nonstandard_sample_count only for smoke tests."
        )

    if len(dataset) < max(2, world_size):
        raise ValueError("Use at least two images and at least one image per rank")
    stats_root = Path(stats_root)
    required_stats = [_stats_path(name, stats_root) for _, name in REPRESENTATION_EXTRACTORS]
    missing_stats = [str(path) for path in required_stats if not path.is_file()]
    if missing_stats:
        raise FileNotFoundError("Missing representation reference statistics:\n" + "\n".join(missing_stats))

    base, remainder = divmod(len(dataset), world_size)
    start = rank * base + min(rank, remainder)
    end = start + base + int(rank < remainder)
    local_indices = list(range(start, end))
    loader = DataLoader(
        Subset(dataset, local_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    results = {}
    score_mean = score_std = None

    for table_name, extractor_name in REPRESENTATION_EXTRACTORS:
        if rank == 0:
            print(f"Loading {extractor_name} and evaluating {len(dataset)} images...", flush=True)
        extractor = FeatureExtractor(extractor_name, device=device).to(device).eval()
        feat_dim = extractor.config.feat_dim
        feature_sum = torch.zeros(feat_dim, dtype=torch.float64, device=device)
        feature_outer = torch.zeros(feat_dim, feat_dim, dtype=torch.float64, device=device)
        count = 0
        local_logits = []

        for images in loader:
            images = images.to(device, non_blocking=True)
            # [-1, 1] inputs under bf16 autocast
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if extractor_name == "Inception-v3":
                    features, logits = extractor.eval_extractor(images)
                    local_logits.append(logits.float().cpu())
                else:
                    features = extractor(images)
            features = features.double()
            feature_sum.add_(features.sum(dim=0))
            feature_outer.addmm_(features.T, features)
            count += features.shape[0]

        mu, sigma, total = _reduce_statistics(
            feature_sum, feature_outer, count, rank, world_size
        )
        if extractor_name == "Inception-v3":
            logits = _gather_logits(local_logits, device, rank, world_size)
            if rank == 0:
                score_mean, score_std = inception_score(logits)

        if rank == 0:
            stats_path = _stats_path(extractor_name, stats_root)
            mu_key, sigma_key = (
                ("avg_mu", "avg_sigma") if extractor.pool_type == "avg" else ("mu", "sigma")
            )
            with np.load(stats_path) as reference:
                results[table_name] = frechet_distance(
                    mu, sigma, reference[mu_key], reference[sigma_key]
                )
            print(f"{table_name}: {results[table_name]:.4f}", flush=True)

        del extractor, feature_sum, feature_outer
        torch.cuda.empty_cache()
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        row = {
            **results,
            "FID": results["Incep."],
            "IS": score_mean,
            "IS std": score_std,
            "N": len(dataset),
            "protocol": "multi-representation-fd-v2",
            "reference_stats_root": str(stats_root),
            "extractors": ",".join(name for name, _ in REPRESENTATION_EXTRACTORS),
            "image_dir": str(Path(image_dir).resolve()),
        }
        output_json = Path(output_json)
        output_csv = Path(output_csv)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        print("\nMulti-representation evaluation")
        print(" | ".join(f"{name}: {row[name]:.4f}" for name, _ in REPRESENTATION_EXTRACTORS))
        print(f"FID: {row['FID']:.4f} | IS: {row['IS']:.1f}")
        print(f"Saved JSON: {output_json}\nSaved CSV: {output_csv}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return results if rank == 0 else None
