"""Sample latents into one .npz: discrete tokens as int16/int32 or continuous latents as float16,
with labels (N,) int16. Both latent types use the "tokens" key.
Each rank streams its strided shard to a memmap with a progress file (resumable); rank 0 merges.

    python precompute_latents.py --generator_name llamagen-B_256 \
        --num_samples 1280000 --batch_size 256 --out pools/llamagen-B_256_train.npz
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from models.generator import Generator
from models.registries import family_registry


def parse_args():
    p = argparse.ArgumentParser(description="Sample a pool of generation-time latents")
    p.add_argument("--generator_name", type=str, required=True)
    p.add_argument("--generator_ckpt_path", type=str, default="")
    p.add_argument("--num_samples", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=256,
                   help="per-rank batch size")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--cfg_omega", type=float, default=None,
                   help="guidance scale; None = registry value")
    p.add_argument("--use_cfg", action="store_true", default=True)
    p.add_argument("--no_cfg", dest="use_cfg", action="store_false")
    p.add_argument("--seed", type=int, default=0,
                   help="base seed; --deterministic uses seed ^ index per sample")
    p.add_argument("--deterministic", action="store_true", default=False,
                   help="seed each sample separately")
    p.add_argument("--restart", action="store_true", default=False,
                   help="discard existing shards instead of resuming")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"],
                   help="autocast dtype for sampling; bf16 also enables TF32")
    p.add_argument("--sample_chunk", type=int, default=None,
                   help="VAR sampling chunk (models/var/registry.SAMPLE_CHUNK); None = registry value")
    p.add_argument("--compile", type=str, default="none", choices=["none", "default", "cudagraph"],
                   help="torch.compile the generator's transformer forward")
    return p.parse_args()


def interleaved_labels(num_samples, num_classes):
    """-> (num_samples,) int16; index i has class i % num_classes."""
    return (np.arange(num_samples, dtype=np.int64) % num_classes).astype(np.int16)


def shard_paths(out, rank):
    base = os.path.splitext(out)[0]
    return f"{base}.rank{rank}.npy", f"{base}.rank{rank}.progress"


def main():
    args = parse_args()
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    reg = family_registry(generator_name=args.generator_name)
    if reg is None:
        raise ValueError(f"No registry for generator {args.generator_name}")
    if not args.generator_ckpt_path:
        args.generator_ckpt_path = reg.ckpt_path(
            "generator", args.generator_name, args.resolution)

    token_length = reg.token_length(args.resolution, name=args.generator_name)
    codebook = getattr(reg, "CODEBOOK_SIZE", None)
    if codebook is None:
        dtype = np.float16
    else:
        dtype = np.int16 if codebook <= np.iinfo(np.int16).max else np.int32
    # per-rank seed offset
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    labels = interleaved_labels(args.num_samples, args.num_classes)

    # strided shard: rank r owns indices r, r+world_size, ...
    my_idx = np.arange(rank, args.num_samples, world_size)
    tok_path, prog_path = shard_paths(args.out, rank)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    done = 0
    if args.restart:
        for p in (tok_path, prog_path):
            if os.path.exists(p):
                os.remove(p)
    if os.path.exists(tok_path) and os.path.exists(prog_path):
        existing = np.lib.format.open_memmap(tok_path, mode="r")
        if existing.shape == (len(my_idx), token_length):
            done = int(open(prog_path).read().strip() or 0)
        else:
            print(f"[rank {rank}] shard shape {existing.shape} does not match "
                  f"{(len(my_idx), token_length)}; starting over", flush=True)
        del existing
    mode = "r+" if done else "w+"
    tokens_local = np.lib.format.open_memmap(
        tok_path, mode=mode, dtype=dtype, shape=(len(my_idx), token_length))
    if done and rank == 0:
        print(f"[resume] {done}/{len(my_idx)} per rank already sampled", flush=True)

    generator = Generator(args.generator_name, args.generator_ckpt_path, args.resolution,
                          cfg_omega=args.cfg_omega).to(device)
    generator.eval_mode()
    if args.sample_chunk is not None:
        from models.var import registry as var_registry
        var_registry.SAMPLE_CHUNK = args.sample_chunk
    if args.dtype == "bf16":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.compile != "none":
        core = generator.model.model if hasattr(generator.model, "model") else generator.model
        if args.compile == "cudagraph":
            core.forward = torch.compile(core.forward, mode="reduce-overhead", dynamic=False)
        else:
            core.forward = torch.compile(core.forward, dynamic=True)
    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                      enabled=(args.dtype == "bf16" and device.type == "cuda"))

    it = range(done, len(my_idx), args.batch_size)
    for i in tqdm(it, desc="sample", disable=rank != 0):
        idx = my_idx[i:i + args.batch_size]
        y = torch.from_numpy(labels[idx].astype(np.int64)).to(device)
        seeds = (args.seed ^ idx) if args.deterministic else None
        with autocast():
            out = generator.standard_generation(len(idx), labels=y, use_cfg=args.use_cfg,
                                                device=device, seeds=seeds)
        tokens_local[i:i + len(idx)] = out.cpu().numpy().astype(dtype)
        tokens_local.flush()
        with open(prog_path, "w") as f:
            f.write(str(i + len(idx)))

    if distributed:
        dist.barrier()

    if rank == 0:
        tokens = np.empty((args.num_samples, token_length), dtype=dtype)
        for r in range(world_size):
            path, _ = shard_paths(args.out, r)
            shard = np.lib.format.open_memmap(path, mode="r")
            tokens[np.arange(r, args.num_samples, world_size)] = shard
            del shard
        np.savez(args.out, tokens=tokens, labels=labels,
                 generator_name=args.generator_name,
                 generator_ckpt_path=args.generator_ckpt_path,
                 resolution=args.resolution, use_cfg=args.use_cfg,
                 cfg_omega=generator.model.cfg_omega, seed=args.seed)
        print(f"[pool] {args.out}  tokens {tokens.shape} {tokens.dtype}  "
              f"{tokens.nbytes / 1024 ** 2:.0f} MB", flush=True)
        for r in range(world_size):
            for p in shard_paths(args.out, r):
                if os.path.exists(p):
                    os.remove(p)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
