"""Compute FD reference statistics (mu, sigma) for representation models.

Supports ImageFolder inputs, plain Python on one GPU or torchrun on multiple GPUs.
Output: .npz with keys "mu", "sigma" (and "avg_mu", "avg_sigma" for dual-output models).

Usage:
    python compute_repr_stats.py \
        --model vit_base_patch14_dinov2.lvd142m \
        --data_path /path/to/imagenet --img_size 256
"""

import argparse
import logging
import os
from datetime import timedelta
import time

import numpy as np
import torch
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

logger = logging.getLogger("FD_loss")

from models.FDloss.feature_extractor import load_repr_model
from data.augmentation import center_crop_arr

def parse_args():
    p = argparse.ArgumentParser(description="Compute repr-model FID reference stats")
    p.add_argument("--model", type=str, required=True,
                    help="'inception' or timm model name (e.g. 'vit_base_patch14_dinov2.lvd142m')")
    p.add_argument("--data_path", type=str, default="data/imagenet",
                    help="ImageNet root dir with a 'train/' subfolder")
    p.add_argument("--num_images", type=int, default=None,
                    help="optional number of images to use")
    p.add_argument("--img_size", type=int, default=256,
                    help="center-crop resolution (default: 256)")
    p.add_argument("--batch_size", type=int, default=256,
                    help="batch size per GPU (default: 256)")
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--target_size", type=int, default=None,
                    help="override model's native target resolution for preprocessing")
    p.add_argument("--output_dir", type=str, default="reference_stats/training/fid_stats",
                    help="directory to save the .npz file")
    p.add_argument("--output_name", type=str, default=None,
                    help="override output filename")
    return p.parse_args()


def setup_distributed():
    """Plain Python needs no process group; torchrun supplies rank and device settings."""
    if not torch.cuda.is_available():
        raise RuntimeError("Computing reference statistics requires a CUDA GPU")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if int(os.environ.get("WORLD_SIZE", 1)) > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def build_dataloader(data_path, img_size, batch_size, num_workers, rank, world_size, num_images=None):
    """Build dataloader for an ImageFolder dataset."""
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)
    total = len(dataset) if num_images is None else min(len(dataset), num_images)
    if total < max(2, world_size):
        raise ValueError("Use at least two images and at least one image per rank")
    # Partition without padding: every selected image contributes exactly once.
    shard = Subset(dataset, range(rank, total, world_size))
    loader = DataLoader(shard, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True,
                        shuffle=False, drop_last=False)
    return loader, total


@torch.inference_mode()
def extract_stats(model, loader, feat_dim, rank, world_size,
                  max_images_per_rank=None, has_logits=False):
    """Accumulate sufficient statistics (sum, outer-product) from repr model features.

    Returns (cls_mu, cls_sigma, avg_mu, avg_sigma, count) on rank 0, else Nones.
    Dual-output models (timm ViTs) produce avg_mu/avg_sigma; others return None.
    """
    device = torch.device("cuda")

    cls_sum = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    cls_outer = torch.zeros(feat_dim, feat_dim, dtype=torch.float64, device=device)
    avg_sum = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    avg_outer = torch.zeros(feat_dim, feat_dim, dtype=torch.float64, device=device)
    has_avg = False
    count = 0

    desc = f"[rank {rank}] extracting features" if world_size > 1 else "extracting features"
    pbar = tqdm(loader, desc=desc, position=rank, disable=False)

    for images, _ in pbar:
        if max_images_per_rank is not None:
            remaining = max_images_per_rank - count
            if remaining <= 0:
                break
            images = images[:remaining]
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            cls_token, avg_pooled_token_or_logits = model(images)
            avg_pooled_token = None if has_logits else avg_pooled_token_or_logits

        cls64 = cls_token.double()
        cls_sum.add_(cls64.sum(0))
        cls_outer.addmm_(cls64.T, cls64)

        if avg_pooled_token is not None:
            has_avg = True
            avg64 = avg_pooled_token.double()
            avg_sum.add_(avg64.sum(0))
            avg_outer.addmm_(avg64.T, avg64)

        count += cls_token.shape[0]
        pbar.set_postfix({"images": count})

        if max_images_per_rank is not None and count >= max_images_per_rank:
            break

    if world_size > 1:
        dist.reduce(cls_sum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(cls_outer, dst=0, op=dist.ReduceOp.SUM)
        if has_avg:
            dist.reduce(avg_sum, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(avg_outer, dst=0, op=dist.ReduceOp.SUM)
        count_t = torch.tensor([count], dtype=torch.long, device=device)
        dist.reduce(count_t, dst=0, op=dist.ReduceOp.SUM)
        count = count_t.item()

    if rank == 0:
        def _compute_mu_sigma(s, S, n):
            s_np = s.cpu().numpy()
            mu = s_np / n
            sigma = (S.cpu().numpy() - np.outer(s_np, s_np) / n) / (n - 1)
            return mu, sigma

        cls_mu, cls_sigma = _compute_mu_sigma(cls_sum, cls_outer, count)
        avg_mu, avg_sigma = None, None
        if has_avg:
            avg_mu, avg_sigma = _compute_mu_sigma(avg_sum, avg_outer, count)
        return cls_mu, cls_sigma, avg_mu, avg_sigma, count
    return None, None, None, None, count


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("httpx", "timm", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    args = parse_args()
    rank, world_size = setup_distributed()
    if rank != 0:
        logger.setLevel(logging.WARNING)

    logger.info(f"Computing stats: model={args.model}, img_size={args.img_size}, gpus={world_size}")

    repr_model, feat_dim, has_logits, target_size = load_repr_model(
        args.model, device="cuda", target_size=args.target_size,
    )

    loader, total_images = build_dataloader(
        args.data_path, args.img_size, args.batch_size,
        args.num_workers, rank, world_size, num_images=args.num_images,
    )

    logger.info(f"Dataset: {total_images} images ({len(loader.dataset)} on rank {rank})")

    t0 = time.perf_counter()
    cls_mu, cls_sigma, avg_mu, avg_sigma, count = extract_stats(
        repr_model, loader, feat_dim, rank, world_size,
        has_logits=has_logits,
    )
    elapsed = time.perf_counter() - t0

    logger.info(f"Processed {count} images in {elapsed:.1f}s ({count / elapsed:.0f} img/s)")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.output_name:
            fname = args.output_name
        else:
            safe_name = args.model.replace("/", "_").replace(".", "_")
            if safe_name == "inception":
                target_size = 256
            fname = f"{safe_name}_in{args.img_size}_t{target_size}_stats.npz"
        out_path = os.path.join(args.output_dir, fname)

        save_dict = {"mu": cls_mu, "sigma": cls_sigma}
        if avg_mu is not None:
            save_dict["avg_mu"] = avg_mu
            save_dict["avg_sigma"] = avg_sigma
        np.savez(out_path, **save_dict)

        logger.info(f"Saved {out_path} (n={count}, feat_dim={cls_mu.shape[0]})")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
