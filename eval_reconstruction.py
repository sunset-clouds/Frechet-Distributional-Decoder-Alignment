import os
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from data.augmentation import center_crop_arr
from metric.metric import PSNR, LPIPS, SSIM
from models.tokenizer import Tokenizer
from utils.npz import create_npz_from_sample_folder
from utils.eval_utils import compute_openai_fid, load_tokenizer_checkpoint, save_eval_results
from utils.distributed import init_distributed_mode
import config

class ShardWithIndex(Dataset):
    """One rank's slice of a dataset; yields (image, global index)."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        gi = int(self.indices[i])
        x, _ = self.dataset[gi]
        return x, gi

def load_dataset(args):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    data_path = args.dataset_dir

    if args.dataset_name == "ImageNet":
        val_set = ImageFolder(root=os.path.join(data_path, 'val'), transform=transform)
    len_val_set = min(len(val_set), args.eval_num_samples)
    # strided shard: rank r owns indices r, r+world_size, ...
    my_idx = np.arange(args.rank, len_val_set, args.world_size)
    shard = ShardWithIndex(val_set, my_idx)
    dataloader = DataLoader(
        shard,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )
    return dataloader, len_val_set

def main_worker(args):
    """Each rank reconstructs its shard; metrics are all-reduced; rank 0 packs the npz and optionally scores FID."""
    if "RANK" in os.environ:
        init_distributed_mode(args)                          # torchrun
    else:                                                    # plain python, single process
        args.rank, args.world_size, args.gpu, args.distributed = 0, 1, 0, False
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    if args.eval_num_samples <= 0:
        raise ValueError("--eval_num_samples must be positive")
    if args.global_batch_size % args.world_size != 0:
        raise ValueError("--global_batch_size must be divisible by the evaluation world size")
    if args.batch_size <= 0:
        raise ValueError("Per-rank batch size must be positive; increase --global_batch_size")

    val_dataloader, len_val_set = load_dataset(args)

    reconstruction_name = '{}_{}_{}_{}'.format(
        args.dataset_name,
        args.tokenizer_name,
        args.resolution,
        args.eval_name,
    )
    reconstruction_path = os.path.join(args.reconstruction_dir, reconstruction_name)
    if args.rank == 0:
        os.makedirs(reconstruction_path, exist_ok=True)
    if args.distributed:
        dist.barrier()

    tokenizer = Tokenizer(tokenizer_name=args.tokenizer_name, tokenizer_ckpt_path=args.tokenizer_ckpt_path).to(device)
    load_tokenizer_checkpoint(tokenizer, args.eval_checkpoint)
    tokenizer.eval_mode()

    psnr_metric = PSNR(device=device)
    ssim_metric = SSIM()
    lpips_metric = LPIPS(device=device)
    ssim, psnr, lpips, count = 0.0, 0.0, 0.0, 0
    iterator = tqdm(val_dataloader, desc="reconstruct") if args.rank == 0 else val_dataloader
    for x, gi in iterator:
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            x_rec = tokenizer.standard_reconstruction(x)
            samples = torch.clamp(127.5 * x_rec + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy() 

            # one PNG per sample, named by global index
            for k, sample in enumerate(samples):
                Image.fromarray(sample).save(f"{reconstruction_path}/{int(gi[k]):06d}.png")

            x_norm = (x + 1.0)/2.0
            x_rec_norm = (x_rec + 1.0)/2.0

            batch_lpips = lpips_metric(x_norm, x_rec_norm).sum()
            batch_psnr = psnr_metric(x_norm, x_rec_norm).sum()
            batch_ssim = ssim_metric(x_norm, x_rec_norm).sum()

            ssim += batch_ssim.item()
            psnr += batch_psnr.item()
            lpips += batch_lpips.item()
            count += x.size(0)

    # sum metrics and sample count across ranks
    if args.distributed:
        stats = torch.tensor([psnr, ssim, lpips, float(count)], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        psnr, ssim, lpips, count = stats.tolist()

    if args.rank == 0:
        print("PSNR:"+str(psnr/count)+"  SSIM:"+str(ssim/count)+ "  LPIPS:"+str(lpips/count))

    if args.distributed:
        dist.barrier()
    if args.rank == 0:
        sample_npz = create_npz_from_sample_folder(reconstruction_path, num=len_val_set)
        if args.compute_fid:
            fid, is_score = compute_openai_fid(
                args.fid_reference_path,
                sample_npz,
                args.inception_graph_path,
                args.fid_batch_size,
            )
            result_path = os.path.join(reconstruction_path, "rfid.json")
            save_eval_results(
                result_path,
                metric="rFID",
                fid=fid,
                inception_score=is_score,
                num_samples=len_val_set,
                reference_npz=args.fid_reference_path,
                sample_npz=sample_npz,
                checkpoint=args.eval_checkpoint,
                psnr=psnr / count,
                ssim=ssim / count,
                lpips=lpips / count,
            )
            print(f"rFID: {fid:.4f}, IS: {is_score:.4f}")
            print(f"Saved evaluation results to {result_path}")
        else:
            print(f"Saved rFID samples to {sample_npz}; run eval_fid.py in the TensorFlow environment.")
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':
    args = config.parse_arg()
    main_worker(args)
