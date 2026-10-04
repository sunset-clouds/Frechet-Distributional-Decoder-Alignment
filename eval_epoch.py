import contextlib
import os
import time

import torch
from torch import distributed as dist
from tqdm import tqdm

from metric.metric import LPIPS, PSNR, SSIM
from utils.util import Pack
from models.FDloss.evalutor import FDEvaluator
from models.FDloss.registry import get_extractor_config


@contextlib.contextmanager
def timed(label):
    """Context manager; prints the block's wall time on rank 0."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if is_main_process():
        print(f"[eval] {label}: {time.perf_counter() - start:.1f}s", flush=True)


def fdr6(fd_by_space):
    """{space: FD} -> mean over spaces of FD / valFD."""
    return sum(fd / get_extractor_config(name).valfd
               for name, fd in fd_by_space.items()) / len(fd_by_space)


def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def finalize_fd(metric, name):
    """-> the space's FD, or NaN when finalize raises ValueError."""
    try:
        return metric.finalize()["fd"]
    except ValueError as err:
        print(f"[eval] {name} FD failed, reporting NaN: {err}", flush=True)
        return float("nan")


def eval_reconstruction_performance(model, epoch, val_dataloader, len_val_set, fid_stats_dir):
    module = model.module if hasattr(model, "module") else model
    module.eval_mode()

    device = next(module.parameters()).device
    with timed("reconstruction / build 3 image metrics + 6 extractors"):
        psnr_metric = PSNR(device=device)
        ssim_metric = SSIM()
        lpips_metric = LPIPS(device=device)
        inception_metric = FDEvaluator(extractor_name='Inception-v3', eval_mode="reconstruction", fid_stats_dir=fid_stats_dir)
        convnext_metric = FDEvaluator(extractor_name='ConvNeXt-v2', eval_mode="reconstruction", fid_stats_dir=fid_stats_dir)
        dino_metric = FDEvaluator(extractor_name='DINOv2', eval_mode="reconstruction", fid_stats_dir=fid_stats_dir)
        mae_metric = FDEvaluator(extractor_name='MAE', eval_mode="reconstruction", fid_stats_dir=fid_stats_dir)
        siglip_metric = FDEvaluator(extractor_name='SigLIP2', eval_mode="reconstruction", fid_stats_dir=fid_stats_dir)
        clip_metric = FDEvaluator(extractor_name='CLIP', eval_mode="reconstruction", fid_stats_dir=fid_stats_dir)

    psnr, ssim, lpips, rec_loss, rFID, rIS, rFD_convnext, rFD_dino, rFD_mae, rFD_siglip, rFD_clip, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    iterator = tqdm(val_dataloader, desc=f"Eval epoch {epoch}", disable=not is_main_process())
    loop = timed(f"reconstruction / {len_val_set} images through decoder + 3 metrics + 6 extractors")
    loop.__enter__()
    for x, labels in iterator:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = x.size(0)

        with torch.no_grad():
            x_rec = module.standard_reconstruction(x)
            inception_metric.update(x_rec)
            convnext_metric.update(x_rec)
            dino_metric.update(x_rec)
            mae_metric.update(x_rec)
            siglip_metric.update(x_rec)
            clip_metric.update(x_rec)

            x_norm = (x + 1.0) / 2.0
            x_rec_norm = torch.clamp((x_rec + 1.0) / 2.0, 0, 1)

            batch_lpips = lpips_metric(x_norm, x_rec_norm).sum()
            batch_psnr = psnr_metric(x_norm, x_rec_norm).sum()
            batch_ssim = ssim_metric(x_norm, x_rec_norm).sum()
            batch_rec_loss = (x - x_rec).square().flatten(1).mean(1).sum()
            batch_total = torch.tensor(batch_size, device=device, dtype=torch.float32)

        if dist.is_available() and dist.is_initialized():
            for value in [batch_lpips, batch_psnr, batch_ssim, batch_rec_loss, batch_total]:
                dist.all_reduce(value, op=dist.ReduceOp.SUM)

        lpips += batch_lpips.item()
        psnr += batch_psnr.item()
        ssim += batch_ssim.item()
        rec_loss += batch_rec_loss.item()
        total_num += batch_total.item()

    loop.__exit__(None, None, None)

    with timed("reconstruction / finalize inception"):
        inception_results = inception_metric.finalize()
    rFID = inception_results["fd"]
    rIS = inception_results["inception_score"]

    with timed("reconstruction / finalize convnext dino mae siglip clip"):
        rFD_convnext = finalize_fd(convnext_metric, "rFD_convnext")
        rFD_dino = finalize_fd(dino_metric, "rFD_dino")
        rFD_mae = finalize_fd(mae_metric, "rFD_mae")
        rFD_siglip = finalize_fd(siglip_metric, "rFD_siglip")
        rFD_clip = finalize_fd(clip_metric, "rFD_clip")

    module.train_mode()
    denom = max(float(len_val_set), 1.0)
    return Pack(
        psnr=psnr / denom,
        ssim=ssim / denom,
        lpips=lpips / denom,
        rec_loss=rec_loss / max(total_num, 1.0),
        rFID = rFID, 
        rIS = rIS, 
        rFD_convnext = rFD_convnext,
        rFD_dino = rFD_dino,
        rFD_mae = rFD_mae,
        rFD_siglip = rFD_siglip,
        rFD_clip = rFD_clip,
        total_num=total_num
    )

def eval_generative_performance(
    model,
    generator,
    fid_stats_dir,
    eval_num_samples=50_000,
    batch_size=256,
    num_classes=1_000,
    seed=0,
    use_cfg=True,
    generation_dtype="bf16",
    ):
    module = model.module if hasattr(model, "module") else model
    module.eval_mode()
    generator.eval()

    device = next(module.parameters()).device

    with timed("generation / build 6 extractors"):
        inception_metric = FDEvaluator(extractor_name="Inception-v3",eval_mode="generation",fid_stats_dir=fid_stats_dir)
        convnext_metric = FDEvaluator(extractor_name="ConvNeXt-v2",eval_mode="generation",fid_stats_dir=fid_stats_dir)
        dino_metric = FDEvaluator(extractor_name="DINOv2",eval_mode="generation",fid_stats_dir=fid_stats_dir)
        mae_metric = FDEvaluator(extractor_name="MAE",eval_mode="generation",fid_stats_dir=fid_stats_dir)
        siglip_metric = FDEvaluator(extractor_name="SigLIP2",eval_mode="generation",fid_stats_dir=fid_stats_dir)
        clip_metric = FDEvaluator(extractor_name="CLIP",eval_mode="generation",fid_stats_dir=fid_stats_dir)

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1

    # per-rank RNG streams
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)

    # label i = i % num_classes
    labels = torch.arange(eval_num_samples, dtype=torch.long) % num_classes
    n_total = len(labels)

    # this rank's strided slice of the global indices
    my_global = torch.arange(rank, n_total, world_size, dtype=torch.long)

    iterator = range(0, len(my_global), batch_size)
    iterator = tqdm(
        iterator,
        desc="Evaluate generation",
        disable=not is_main_process(),
    )

    autocast_enabled = generation_dtype == "bf16" and device.type == "cuda"
    loop = timed(f"generation / {eval_num_samples} samples through AR + decoder + 6 extractors")
    loop.__enter__()

    with torch.inference_mode():
        for i in iterator:
            g = my_global[i : i + batch_size]

            batch_labels = labels[g].to(
                device=device,
                non_blocking=True,
            )

            # one seed per batch from (seed, batch start index)
            torch.manual_seed(seed * 1_000_003 + i)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                z_gen = generator.standard_generation(
                    batch_size=len(g),
                    labels=batch_labels,
                    use_cfg=use_cfg,
                    device=device,
                )
                x_gen = module.latent_to_image(z_gen)

            inception_metric.update(x_gen)
            convnext_metric.update(x_gen)
            dino_metric.update(x_gen)
            mae_metric.update(x_gen)
            siglip_metric.update(x_gen)
            clip_metric.update(x_gen)

    loop.__exit__(None, None, None)

    with timed("generation / finalize inception"):
        inception_results = inception_metric.finalize()
    gFID = inception_results["fd"]
    gIS = inception_results["inception_score"]

    with timed("generation / finalize convnext dino mae siglip clip"):
        gFD_convnext = finalize_fd(convnext_metric, "gFD_convnext")
        gFD_dino = finalize_fd(dino_metric, "gFD_dino")
        gFD_mae = finalize_fd(mae_metric, "gFD_mae")
        gFD_siglip = finalize_fd(siglip_metric, "gFD_siglip")
        gFD_clip = finalize_fd(clip_metric, "gFD_clip")

    module.train_mode()

    return Pack(
        gFID=gFID,
        gIS=gIS,
        gFDr6=fdr6({"Inception-v3": gFID, "ConvNeXt-v2": gFD_convnext, "DINOv2": gFD_dino,
                    "MAE": gFD_mae, "SigLIP2": gFD_siglip, "CLIP": gFD_clip}),
        gFD_convnext=gFD_convnext,
        gFD_dino=gFD_dino,
        gFD_mae=gFD_mae,
        gFD_siglip=gFD_siglip,
        gFD_clip=gFD_clip,
    )