import os
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image

from models.tokenizer import Tokenizer
from models.generator import Generator
from utils.npz import create_npz_from_sample_folder
from utils.eval_utils import compute_openai_fid, load_tokenizer_checkpoint, save_eval_results
from utils.distributed import init_distributed_mode
import config

NUM_CLASSES = 1000

def build_labels(num_samples, num_classes=NUM_CLASSES):
    """-> (num_samples,) int64; label i = i % num_classes."""
    repeats = (num_samples + num_classes - 1) // num_classes
    return np.tile(np.arange(num_classes, dtype=np.int64), repeats)[:num_samples]

def main_worker(args):
    """Each rank generates its own shard of PNGs; rank 0 packs the npz and optionally scores FID."""
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

    # per-rank seed offset
    torch.manual_seed(args.seed + args.rank)
    torch.cuda.manual_seed_all(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)

    labels = build_labels(args.eval_num_samples)
    n_total = len(labels)

    cfg_name = "cfg" if args.use_cfg else "no_cfg"
    generation_name = '{}_{}_{}_{}_{}'.format(
        args.dataset_name,
        args.tokenizer_name,
        args.generator_name,
        args.eval_name,
        cfg_name,
    )
    generation_path = os.path.join(args.generation_dir, generation_name)
    if args.rank == 0:
        os.makedirs(generation_path, exist_ok=True)
    if args.distributed:
        dist.barrier()

    tokenizer = Tokenizer(tokenizer_name=args.tokenizer_name, tokenizer_ckpt_path=args.tokenizer_ckpt_path).to(device)
    load_tokenizer_checkpoint(tokenizer, args.eval_checkpoint)
    tokenizer.eval_mode()
    generator = Generator(
        generator_name=args.generator_name,
        generator_ckpt_path=args.generator_ckpt_path,
        resolution=args.resolution,
        cfg_omega=args.cfg_omega,
    ).to(device)
    generator.eval_mode()

    if args.sampling_protocol == "fd_loss":
        base, remainder = divmod(n_total, args.world_size)
        start = args.rank * base + min(args.rank, remainder)
        end = start + base + int(args.rank < remainder)
        my_global = np.arange(start, end)
    else:
        # strided shard: rank r owns indices r, r+world_size, ...
        my_global = np.arange(args.rank, n_total, args.world_size)
    iterator = range(0, len(my_global), args.batch_size)
    if args.rank == 0:
        iterator = tqdm(iterator, desc="generate")

    for i in iterator:
        g = my_global[i:i + args.batch_size]
        y = torch.from_numpy(labels[g]).to(device, non_blocking=True)
        seeds = None
        if args.sampling_protocol == "indexed":
            seeds = torch.from_numpy(g.astype(np.int64) ^ np.int64(args.seed)).to(device)
        autocast_enabled = args.generation_dtype == "bf16"
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            z_gen = generator.standard_generation(
                batch_size=len(g),
                labels=y,
                use_cfg=args.use_cfg,
                device=device,
                seeds=seeds,
            )
            x_gen = tokenizer.latent_to_image(z_gen)
            if args.sampling_protocol == "fd_loss":
                samples = (x_gen * 0.5 + 0.5).mul(255).round().clamp(0, 255)
            else:
                samples = torch.clamp(127.5 * x_gen + 128.0, 0, 255)
            samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # one PNG per sample, named by global index
        for j, sample in enumerate(samples):
            Image.fromarray(sample).save(f"{generation_path}/{int(g[j]):06d}.png")

    if args.distributed:
        dist.barrier()
    if args.rank == 0:
        manifest_path = os.path.join(generation_path, "generation_manifest.json")
        save_eval_results(
            manifest_path,
            generator_name=args.generator_name,
            generator_checkpoint=os.path.abspath(args.generator_ckpt_path),
            tokenizer_name=args.tokenizer_name,
            tokenizer_checkpoint=os.path.abspath(args.tokenizer_ckpt_path),
            decoder_checkpoint=os.path.abspath(args.eval_checkpoint) if args.eval_checkpoint else None,
            use_cfg=args.use_cfg,
            cfg_omega=generator.model.cfg_omega if args.use_cfg else None,
            sampling_protocol=args.sampling_protocol,
            generation_dtype=args.generation_dtype,
            seed=args.seed,
            num_samples=n_total,
            world_size=args.world_size,
        )
        if args.skip_npz:
            print(f"Saved {n_total} generated images to {generation_path}")
        else:
            sample_npz = create_npz_from_sample_folder(
                generation_path,
                num=n_total,
                labels=labels,
            )
        if args.compute_fid and args.skip_npz:
            raise ValueError("--compute_fid requires NPZ packing; remove --skip_npz")
        if args.compute_fid:
            fid, is_score = compute_openai_fid(
                args.fid_reference_path,
                sample_npz,
                args.inception_graph_path,
                args.fid_batch_size,
            )
            result_path = os.path.join(generation_path, f"gfid_{cfg_name}.json")
            save_eval_results(
                result_path,
                metric="gFID",
                fid=fid,
                inception_score=is_score,
                num_samples=n_total,
                use_cfg=args.use_cfg,
                reference_npz=args.fid_reference_path,
                sample_npz=sample_npz,
                checkpoint=args.eval_checkpoint,
            )
            print(f"gFID: {fid:.4f}, IS: {is_score:.4f}")
            print(f"Saved evaluation results to {result_path}")
        elif not args.skip_npz:
            print(f"Saved gFID samples to {sample_npz}; run eval_fid.py in the TensorFlow environment.")
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':
    args = config.parse_arg()
    main_worker(args)
