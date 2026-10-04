import contextlib
import os
import sys
import time

import torch
import pandas as pd
from torch import nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import config
from utils.util import Logger, LossManager, save_checkpoint, schedule_lr
from utils.distributed import init_distributed_mode
from data.dataloader import build_dataloader, build_latent_pool_loader
from models.losses import TrainingLoss
from models.tokenizer import Tokenizer
from models.generator import Generator 
from eval_epoch import eval_reconstruction_performance, eval_generative_performance

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def unpack_batch(batch, device, pool_only=False):
    """batch -> (real images or None, generated latents or None); pool_only marks a (z_g, label) batch."""
    if pool_only:
        return None, batch[0].to(device, non_blocking=True)
    x, z_g = batch[0], None
    if len(batch) >= 3:
        z_g = batch[1].to(device, non_blocking=True)
    return x.to(device, non_blocking=True), z_g


def autocast_context(args, device):
    """-> a context-manager factory for the training forward's autocast precision."""
    if args.amp_dtype == "fp32":
        return contextlib.nullcontext
    dtype = {"bf16": torch.bfloat16}[args.amp_dtype]
    return lambda: torch.autocast(device_type="cuda", dtype=dtype)


def is_main_process():
    """-> True on rank 0."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get('LOCAL_RANK', 0)) == 0

def main_worker(args):
    """Train and evaluate the decoder; with --eval_only or --eval_checkpoint, evaluate only."""
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    init_distributed_mode(args)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    assert args.global_batch_size % world_size == 0, "Batch size must be divisible by world size."
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    
    model = Tokenizer(tokenizer_name=args.tokenizer_name, tokenizer_ckpt_path=args.tokenizer_ckpt_path)
    model = model.to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    loss_fn = TrainingLoss(args)

    model_para = model.trainable_parameters()
    disc_para = list(loss_fn.discriminator.parameters()) if loss_fn.has_discriminator else []
    
    optimizer = torch.optim.AdamW(model_para, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    optimizer_disc = torch.optim.AdamW(disc_para, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay) if disc_para else None

    # --eval_checkpoint scores a checkpoint; bare --eval_only scores the released decoder
    eval_only = args.eval_only or bool(args.eval_checkpoint)

    # pool_only: no loss term reads a real training image
    needs_real_images = (args.rec_weight > 0 or args.perceptual_weight > 0 or args.disc_weight > 0)
    pool_only = bool(args.latent_pool) and not needs_real_images

    train_dataloader, train_sampler, len_train_set = None, None, 0
    if eval_only:
        source = "not built, eval only"
    elif pool_only:
        train_dataloader, train_sampler, len_train_set = build_latent_pool_loader(
            args, args.latent_pool)
        source = f"latent pool {args.latent_pool}"
    else:
        train_dataloader, train_sampler, len_train_set = build_dataloader(args, split='train')
        source = "images"
    val_dataloader, _, len_val_set = build_dataloader(args, split='val')

    if is_main_process():
        print(f"Train size: {len_train_set} ({source}), Val size: {len_val_set}")

    if getattr(args, "distributed", False):
        model = DDP(model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    model_module = model.module if hasattr(model, "module") else model
    model_module.train_mode()

    loss_fn = loss_fn.to(device)
    if getattr(args, "distributed", False) and any(param.requires_grad for param in loss_fn.parameters()):
        loss_fn = DDP(loss_fn, device_ids=[args.gpu], find_unused_parameters=True)

    loss_fn.train()
    loss_module = loss_fn.module if hasattr(loss_fn, "module") else loss_fn
    if loss_module.perceptual_loss is not None:
        loss_module.perceptual_loss.eval()
    if loss_module.fd_loss is not None:
        loss_module.fd_loss.eval_extractors()

    reconstruction_results_eval = {'epoch': [], 'rec_loss': [], 'psnr': [], 'ssim': [], 'lpips': [], 'rFID':[], 'rIS':[], 'rFD_convnext':[], 'rFD_dino':[], 'rFD_mae':[], 'rFD_siglip':[], 'rFD_clip':[]}
    generation_results_eval = {'epoch': [], 'gFID':[], 'gIS':[], 'gFDr6':[], 'gFD_convnext':[], 'gFD_dino':[], 'gFD_mae':[], 'gFD_siglip':[], 'gFD_clip':[]}
    train_loss = LossManager()

    if args.eval_epochs <= 0:
        raise ValueError("eval_epochs must be positive")

    # training forward only; generation during evaluation uses --generation_dtype
    autocast = autocast_context(args, device)

    if args.eval_checkpoint:
        state = torch.load(args.eval_checkpoint, map_location="cpu", weights_only=False)
        model_module.load_state_dict(state["model"])
        if is_main_process():
            print(f"[eval_only] loaded epoch {state.get('epoch', '?')} from {args.eval_checkpoint}")

    if is_main_process():
        print("\nStart evaluation..." if eval_only else "\nStart training...")
        if eval_only and not args.eval_checkpoint:
            print("[eval_only] released decoder, no checkpoint loaded")

    if args.fd_weight > 0.0 and not eval_only:
        # prime the EMA moments from --warmup_pool, else from the training loader
        filled = 0
        queue_size = loss_module.fd_loss.queue_size
        if args.warmup_pool:
            warm_loader, _, warm_size = build_latent_pool_loader(args, args.warmup_pool)
            warm_pool_only = True
            if warm_size < queue_size and is_main_process():
                print(f"[warmup] pool has {warm_size} entries for queue_size={queue_size}; "
                      f"using the available complete batches")
        else:
            warm_loader = train_dataloader
            warm_pool_only = pool_only
            if train_sampler is not None:
                train_sampler.set_epoch(0)
        with torch.no_grad():
            for batch in warm_loader:
                x, z_g = unpack_batch(batch, device, pool_only=warm_pool_only)
                with autocast():
                    x_warm = model_module.latent_to_image(z_g) if z_g is not None else model(x)
                    filled = loss_module.fd_loss.updates(x_warm, filled=filled)
                if filled >= queue_size:
                    break
        if filled < 2:
            raise ValueError("FD warmup needs at least two samples in complete batches; "
                             "provide a larger pool or reduce --global_batch_size")
        if args.fd_loss_type == "queue" and filled < queue_size:
            raise ValueError(f"FD queue warmup filled {filled}/{queue_size} entries; "
                             "provide more warmup samples or reduce --fd_queue_size")
        loss_module.fd_loss.finalize()
        if is_main_process():
            print(f"[warmup] {filled} samples from "
                  f"{args.warmup_pool or 'the training loader'}")

    total_epochs = 1 if eval_only else args.epochs
    # the lr schedule spans all epochs
    steps_per_epoch = len(train_dataloader) if train_dataloader is not None else 0
    total_steps = steps_per_epoch * total_epochs
    global_step = 0
    start_epoch = 1
    for epoch in range(start_epoch, total_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{total_epochs}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'=' * 60}")

        start_time = time.time()
        for step, batch in enumerate([] if eval_only else train_dataloader):
            if args.lr_schedule == "cosine" or args.warmup_steps > 0:
                peak = args.lr
                floor = args.min_lr if args.lr_schedule == "cosine" else args.lr
                schedule_lr(optimizer, global_step, total_steps, peak, floor, args.warmup_steps)
            global_step += 1
            x, z_g = unpack_batch(batch, device, pool_only=pool_only)

            optimizer.zero_grad(set_to_none=True)
            need_recon = (args.rec_weight > 0 or args.perceptual_weight > 0
                          or (optimizer_disc is not None and epoch >= args.disc_start_epoch))
            with autocast():
                x_g = model_module.latent_to_image(z_g) if z_g is not None else None
                x_rec = model(x) if need_recon else None
                gen_loss, gen_loss_pack = loss_fn(
                    x, x_rec,
                    optimizer_idx=0,
                    cur_epoch=epoch,
                    last_layer=model_module.last_layer,
                    generated=x_g,
                )
            gen_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model_para, 1.0)
            if torch.isfinite(grad_norm).item():
                optimizer.step()
            elif is_main_process():
                print(f"[step {step}] NaN/Inf generator grad norm; skipping optimizer step")

            disc_active = optimizer_disc is not None and epoch >= args.disc_start_epoch
            if disc_active:
                optimizer_disc.zero_grad(set_to_none=True)
                with autocast():
                    d_loss, d_loss_pack = loss_fn(
                        x, x_rec.detach(),
                        optimizer_idx=1,
                        cur_epoch=epoch,
                    )
                d_loss.backward()
                disc_grad_norm = torch.nn.utils.clip_grad_norm_(disc_para, 1.0)
                if torch.isfinite(disc_grad_norm).item():
                    optimizer_disc.step()
                elif is_main_process():
                    print(f"[step {step}] NaN/Inf discriminator grad norm; skipping optimizer step")
                gen_loss_pack.add(d_loss_pack)

            train_loss.add_loss(gen_loss_pack)
            if is_main_process() and step == 9:
                print(f"[memory] peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB at "
                      f"batch {args.global_batch_size}", flush=True)

            if is_main_process() and (step < 10 or (step + 1) % 10 == 0):
                print(train_loss.pprint(
                    window=50,
                    prefix=(f'Epoch [{epoch}/{args.epochs}] Iter [{step+1}/{len(train_dataloader)}] '
                            f'lr {optimizer.param_groups[0]["lr"]:.2e}'),
                ))

        train_loss.clear()
        epoch_time = time.time() - start_time

        if is_main_process():
            print(f"Epoch {epoch} completed in {epoch_time:.1f}s")

        should_evaluate = eval_only or epoch % args.eval_epochs == 0
        if should_evaluate and not eval_only and is_main_process():
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint-{args.saver_name_pre}-{epoch}.pth.tar')
            save_checkpoint({
                'epoch': epoch,
                'model': model_module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'discriminator': loss_module.discriminator.state_dict() if loss_module.has_discriminator else None,
                'optimizer_disc': optimizer_disc.state_dict() if optimizer_disc is not None else None,
                'args': vars(args),
            }, is_best=False, filename=checkpoint_path)

        if getattr(args, "distributed", False):
            dist.barrier()

        if not should_evaluate:
            continue

        # evaluation runs on a forked RNG state
        with torch.random.fork_rng(devices=[device]):
            with torch.no_grad():
                reconstruction_pack = eval_reconstruction_performance(
                    model,
                    epoch,
                    val_dataloader,
                    len_val_set,
                    fid_stats_dir=args.fid_stats_dir,
                )

            if is_main_process():
                reconstruction_results_eval['epoch'].append(epoch)
                reconstruction_results_eval['rec_loss'].append(reconstruction_pack.rec_loss)
                reconstruction_results_eval['psnr'].append(reconstruction_pack.psnr)
                reconstruction_results_eval['ssim'].append(reconstruction_pack.ssim)
                reconstruction_results_eval['lpips'].append(reconstruction_pack.lpips)
                reconstruction_results_eval['rFID'].append(reconstruction_pack.rFID)
                reconstruction_results_eval['rIS'].append(reconstruction_pack.rIS)
                reconstruction_results_eval['rFD_convnext'].append(reconstruction_pack.rFD_convnext)
                reconstruction_results_eval['rFD_dino'].append(reconstruction_pack.rFD_dino)
                reconstruction_results_eval['rFD_mae'].append(reconstruction_pack.rFD_mae)
                reconstruction_results_eval['rFD_siglip'].append(reconstruction_pack.rFD_siglip)
                reconstruction_results_eval['rFD_clip'].append(reconstruction_pack.rFD_clip)

                print("\nReconstruction Evaluation Results:")
                print(f"  PSNR: {reconstruction_pack.psnr:.2f}")
                print(f"  SSIM: {reconstruction_pack.ssim:.4f}")
                print(f"  LPIPS: {reconstruction_pack.lpips:.4f}")
                print(f"  Rec Loss: {reconstruction_pack.rec_loss:.4f}")
                print(f"  rFID: {reconstruction_pack.rFID:.4f}")
                print(f"  rIS: {reconstruction_pack.rIS:.4f}")
                print(f"  rFD_convnext: {reconstruction_pack.rFD_convnext:.4f}")
                print(f"  rFD_dino: {reconstruction_pack.rFD_dino:.4f}")
                print(f"  rFD_mae: {reconstruction_pack.rFD_mae:.4f}")
                print(f"  rFD_siglip: {reconstruction_pack.rFD_siglip:.4f}")
                print(f"  rFD_clip: {reconstruction_pack.rFD_clip:.4f}")

                results_val_len = len(reconstruction_results_eval['epoch'])
                data_frame = pd.DataFrame(
                    data=reconstruction_results_eval,
                    index=range(1, results_val_len + 1),
                )
                data_frame.to_csv(
                    f'{args.results_dir}/eval_{args.saver_name_pre}_rec_results.csv',
                    index_label='index',
                )

            generator = Generator(
                generator_name=args.generator_name,
                generator_ckpt_path=args.generator_ckpt_path,
                cfg_omega=args.cfg_omega,
            ).to(device)
            with torch.no_grad():
                generation_pack = eval_generative_performance(
                    model,
                    generator,
                    fid_stats_dir=args.fid_stats_dir,
                    eval_num_samples=args.eval_num_samples,
                    batch_size=args.fid_batch_size,
                    seed=args.seed,
                    use_cfg=args.use_cfg,
                    generation_dtype=args.generation_dtype,
                )

            if is_main_process():
                generation_results_eval['epoch'].append(epoch)
                generation_results_eval['gFID'].append(generation_pack.gFID)
                generation_results_eval['gIS'].append(generation_pack.gIS)
                generation_results_eval['gFDr6'].append(generation_pack.gFDr6)
                generation_results_eval['gFD_convnext'].append(generation_pack.gFD_convnext)
                generation_results_eval['gFD_dino'].append(generation_pack.gFD_dino)
                generation_results_eval['gFD_mae'].append(generation_pack.gFD_mae)
                generation_results_eval['gFD_siglip'].append(generation_pack.gFD_siglip)
                generation_results_eval['gFD_clip'].append(generation_pack.gFD_clip)

                print("\nGeneration Evaluation Results:")
                print(f"  gFID: {generation_pack.gFID:.4f}")
                print(f"  gIS: {generation_pack.gIS:.4f}")
                print(f"  gFD_convnext: {generation_pack.gFD_convnext:.4f}")
                print(f"  gFD_dino: {generation_pack.gFD_dino:.4f}")
                print(f"  gFD_mae: {generation_pack.gFD_mae:.4f}")
                print(f"  gFD_siglip: {generation_pack.gFD_siglip:.4f}")
                print(f"  gFD_clip: {generation_pack.gFD_clip:.4f}")
                print(f"  gFDr6: {generation_pack.gFDr6:.4f}   "
                      f"(mean of FD/valFD over the six; 1.0 = as far as real held-out data)")

                data_frame = pd.DataFrame(
                    data=generation_results_eval,
                    index=range(1, len(generation_results_eval['epoch']) + 1),
                )
                data_frame.to_csv(
                    f'{args.results_dir}/eval_{args.saver_name_pre}_gen_results.csv',
                    index_label='index',
                )

            del generator

        torch.cuda.empty_cache()

    if is_main_process() and not eval_only:
        print("\n" + "=" * 70)
        print("Training complete! Saving final checkpoint...")
        print("=" * 70)

        checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint-{args.saver_name_pre}-final.pth.tar')
        save_checkpoint({
            'epoch': args.epochs,
            'model': model_module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'discriminator': loss_module.discriminator.state_dict() if loss_module.has_discriminator else None,
            'optimizer_disc': optimizer_disc.state_dict() if optimizer_disc is not None else None,
            'args': vars(args),
        }, is_best=False, filename=checkpoint_path)

    if getattr(args, "distributed", False):
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':
    os.environ['NCCL_TIMEOUT_IN_MS'] = '7200000'
    args = config.parse_arg()
    dict_args = vars(args)

    os.makedirs(args.saver_dir, exist_ok=True)
    sys.stdout = Logger(args.saver_dir, args.saver_name_pre)

    if is_main_process():
        print("Configuration:")
        for k, v in dict_args.items():
            print(f"  {k}: {v}")
        print()

    main_worker(args)
