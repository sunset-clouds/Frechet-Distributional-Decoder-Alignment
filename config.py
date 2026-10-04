import os
import random
import numpy as np
import torch
import argparse
import ruamel.yaml as yaml
from models.registries import model_names, resolve_model_paths

def parse_arg():
    parser = argparse.ArgumentParser(description='Fréchet Distributional Decoder Adaptation (FDDA).') 

    # dataset
    parser.add_argument('--dataset_dir', default="data/imagenet", type=str, help='dataset root directory')
    parser.add_argument('--dataset_name', default='ImageNet', help='dataset name', choices=['ImageNet'])
    parser.add_argument('--global_batch_size', type=int, default=64, help="batch size summed over all ranks")
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loader workers')
    parser.add_argument('--resolution', type=int, choices=[256], default=256, help='image resolution')
    parser.add_argument('--channels', default=3, type=int, metavar='N', help='image channels')

    # model
    parser.add_argument('--tokenizer_name', default='llamagen-vq16', help='tokenizer model name', choices=model_names('tokenizer'))
    parser.add_argument('--generator_name', default='llamagen-B_256', help='generator model name; the resolution is part of the name', choices=model_names('generator'))
    parser.add_argument('--extractor_name', default='Inception-v3', help="feature space(s) of the FD loss, comma-separated")
    parser.add_argument('--fd_extractor_weights', type=float, nargs='+', default=None, help='one weight per space in --extractor_name order; default 1.0')
    parser.add_argument('--fd_norm_eps', type=float, default=0.01, help="eps in normalized = raw / (raw.detach() + eps)")
    parser.add_argument('--fd_loss_type', default='ema', help='FD statistics mode', choices=['ema', 'queue'])
    parser.add_argument('--latent_pool', default="", type=str, help='npz of generated latents; paired index-wise with train images when a loss reads them')
    parser.add_argument('--warmup_pool', default="", type=str, help='npz of generated latents that primes the EMA moments')
    parser.add_argument('--fd_ema_beta', type=float, default=None, help='EMA decay of the generated-side moments; None = registry')
    parser.add_argument('--fd_queue_size', type=int, default=None, help='samples that initialise the moments; None = registry')
    parser.add_argument("--use_cfg", action="store_true", default=True, help="use classifier-free guidance during generation")

    # loss
    parser.add_argument('--rec_weight', type=float, default=0.0, help='reconstruction loss weight; 0 disables')
    parser.add_argument('--perceptual_weight', type=float, default=0.0, help='perceptual loss weight; 0 disables')
    parser.add_argument('--disc_weight', type=float, default=0.0, help='discriminator loss weight; 0 disables')
    parser.add_argument('--fd_weight', type=float, default=0.0, help='FD loss weight; 0 disables')
    parser.add_argument('--lecam_loss_weight', type=float, default=0.001,help='LeCAM regularization weight')
    parser.add_argument('--disc_cr_loss_weight', type=float, default=4.0, help='discriminator CR loss weight')

    # training
    parser.add_argument('--epochs', type=int, default=1, help="training epochs")
    parser.add_argument('--disc_start_epoch', type=int, default=0, help='epoch at which discriminator training starts')
    parser.add_argument('--eval_epochs', type=int, default=1, help="evaluate every N epochs")
    parser.add_argument('--lr', default=1e-5, type=float, metavar='LR', help='peak learning rate')
    parser.add_argument('--lr_schedule', default='constant', choices=['constant', 'cosine'], help="cosine = linear warmup to --lr, then cosine to --min_lr")
    parser.add_argument('--warmup_steps', type=int, default=0, help='linear lr warmup steps')
    parser.add_argument('--min_lr', type=float, default=1e-6, help="cosine floor; ignored for constant")
    parser.add_argument('--dropout', help='unused', type=float, default=0.0)
    parser.add_argument('--seed', help='random seed', type=int, default=3407)
    parser.add_argument('--weight_decay', help='weight decay for optimizer', type=float, default=0.0001)

    # paths and evaluation
    parser.add_argument('--checkpoint_dir', default="outputs/checkpoints", type=str, help='checkpoint directory')
    parser.add_argument('--results_dir', default="outputs/results", type=str, help='results directory')
    parser.add_argument('--saver_dir', default="outputs/samples", type=str, help='log and sample directory')
    parser.add_argument('--yaml_dir', default="outputs/configs", type=str, help='directory the resolved config yaml is written to')
    parser.add_argument('--fid_stats_dir', default="reference_stats", type=str, help='reference statistics directory')
    parser.add_argument('--cfg_omega', type=float, default=None, help='guidance scale at evaluation; None = registry value')
    # 'indexed': uint8 = trunc(127.5x + 128), per-image seeds; 'fd_loss': round(255(0.5x + 0.5)), contiguous shards
    parser.add_argument('--sampling_protocol', default='indexed', choices=['indexed', 'fd_loss'])
    parser.add_argument('--skip_npz', action='store_true', help='write the PNG folder but skip packing it into an npz')
    parser.add_argument('--compute_fid', action='store_true', help='run the OpenAI TensorFlow FID inline')
    parser.add_argument('--fid_reference_path', default='', help='OpenAI ImageNet reference npz')
    parser.add_argument('--inception_graph_path', default='', help='classify_image_graph_def.pb')
    parser.add_argument('--eval_name', default='eval', help="tag in the generated folder's name")
    parser.add_argument('--generation_dir', default='', help='eval_generation.py output dir; empty = <saver_dir>/generation')
    parser.add_argument('--eval_only', action='store_true', help='evaluate without training; no checkpoint = released decoder')
    parser.add_argument('--eval_checkpoint', default="", type=str, help='checkpoint to evaluate; no training runs')
    parser.add_argument('--eval_num_samples', type=int, default=50000, help='samples drawn for the generative metrics; images read by eval_reconstruction.py')
    parser.add_argument('--fid_batch_size', type=int, default=256, help='batch size of the generative eval')
    parser.add_argument('--amp_dtype', default='bf16', choices=['fp32', 'bf16'], help='autocast precision of the training forward')
    parser.add_argument('--generation_dtype', default='fp32', choices=['fp32', 'bf16'], help='autocast dtype for generation')
    parser.add_argument('--tokenizer_ckpt_path', default="", type=str, help='tokenizer checkpoint; empty = registry path')
    parser.add_argument('--generator_ckpt_path', default="", type=str, help='generator checkpoint; empty = registry path')
    parser.add_argument('--nnodes', default=-1, type=int, help='number of nodes')
    parser.add_argument('--node_rank', default=-1, type=int, help='node rank')
    parser.add_argument('--local-rank', default=-1, type=int, help='local rank')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    args = parser.parse_args()

    args = resolve_model_paths(args)
    
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.batch_size = round(args.global_batch_size/args.world_size)
    args.workers = min(max(0, args.workers), args.batch_size)
    # 'A,B' -> 'A+B'
    extractor_tag = args.extractor_name.replace(',', '+').replace(' ', '')
    # a generator checkpoint outside checkpoints/llamagen/generator is tagged into the run name
    gen_tag = args.generator_name
    if args.generator_ckpt_path and 'checkpoints/llamagen/generator' not in args.generator_ckpt_path:
        gen_tag += '@' + os.path.splitext(os.path.basename(args.generator_ckpt_path))[0]
    args.model_pre = '{}_{}_{}_{}'.format(args.tokenizer_name, gen_tag, extractor_tag, args.fd_loss_type)
    args.loss_pre = '{}_{}_{}_{}'.format(args.rec_weight, args.perceptual_weight, args.disc_weight, args.fd_weight)
    args.fd_pre = 'beta{}_q{}_lr{}'.format(args.fd_ema_beta, args.fd_queue_size, args.lr)
    if args.cfg_omega is not None:
        args.fd_pre += '_cfg{}'.format(args.cfg_omega)
    args.saver_name_pre =  args.model_pre + '_' + args.loss_pre + '_' + args.fd_pre

    args.reconstruction_dir = os.path.join(args.saver_dir, 'reconstruction')
    args.generation_dir = args.generation_dir or os.path.join(args.saver_dir, 'generation')

    for path in [
        args.checkpoint_dir,
        args.results_dir,
        args.saver_dir,
        args.reconstruction_dir,
        args.generation_dir,
        args.yaml_dir,
    ]:
        if path:
            os.makedirs(path, exist_ok=True)

    dict_args = vars(args)
    config_name = args.saver_name_pre+'.yaml'
    if args.yaml_dir:
        with open(os.path.join(args.yaml_dir, config_name), 'w', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            file_yaml.dump(dict_args, f)
    
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return args
