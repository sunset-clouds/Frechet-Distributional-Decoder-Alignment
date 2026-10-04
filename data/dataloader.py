import os
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .augmentation import random_crop_arr, center_crop_arr


def _latent_tensor(row):
    """Pool row -> tensor: int tokens as int64, continuous latents as float32."""
    if row.dtype.kind == "f":
        return torch.from_numpy(row.astype(np.float32))
    return torch.from_numpy(row.astype(np.int64))

DATASET_PATHS = {
    "ImageNet": "imagenet",
}

def build_train_transform(resolution=256):
    """Random crop + horizontal flip; PIL -> (3, R, R) in [-1, 1]."""
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    return transform


def build_eval_transform(resolution=256):
    """Center crop; PIL -> (3, R, R) in [-1, 1]."""
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    return transform




class PairedLatentDataset(torch.utils.data.Dataset):
    """(real image, generated latent, pool label) by index; NPZ arrays are loaded into RAM.

    Latents are int64 token indices or float32 continuous tensors.
    """

    def __init__(self, image_dataset, pool_path):
        self.image_dataset = image_dataset
        with np.load(pool_path) as pool:
            self.tokens = pool["tokens"]
            self.labels = pool["labels"]
        if len(self.tokens) < len(image_dataset):
            raise ValueError(
                f"pool {pool_path} has {len(self.tokens)} entries, fewer than the "
                f"{len(image_dataset)} images it is paired with"
            )

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, i):
        x, _ = self.image_dataset[i]
        return x, _latent_tensor(self.tokens[i]), int(self.labels[i])



class LatentPoolDataset(torch.utils.data.Dataset):
    """(generated latent, label) from NPZ arrays loaded into RAM.

    Latents are int64 token indices or float32 continuous tensors.
    """

    def __init__(self, pool_path):
        with np.load(pool_path) as pool:
            self.tokens = pool["tokens"]
            self.labels = pool["labels"]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return _latent_tensor(self.tokens[i]), int(self.labels[i])


def build_latent_pool_loader(args, pool_path):
    """-> (dataloader, sampler or None, pool size); shuffled."""
    dataset = LatentPoolDataset(pool_path)
    distributed = torch.distributed.is_initialized()
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset=dataset,
        num_workers=args.workers,
        pin_memory=True,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        persistent_workers=(args.workers > 0),
    )
    return loader, sampler, len(dataset)


def build_dataloader(args, split='train'):
    """-> (dataloader, sampler or None, dataset size) for split 'train' or 'val'."""
    if split == 'train':
        transform = build_train_transform(args.resolution)
    else:
        transform = build_eval_transform(args.resolution)
    
    if args.dataset_name == "ImageNet":
        # dataset_dir/{train,validation,val} or dataset_dir/imagenet/{train,validation,val}
        split_name = 'train' if split == 'train' else 'validation'

        data_path = os.path.join(args.dataset_dir, split_name)
        if not os.path.exists(data_path):
            data_path = os.path.join(args.dataset_dir, 'val' if split != 'train' else 'train')
        if not os.path.exists(data_path):
            data_path = os.path.join(args.dataset_dir, DATASET_PATHS[args.dataset_name], split_name)
        if not os.path.exists(data_path):
            data_path = os.path.join(args.dataset_dir, DATASET_PATHS[args.dataset_name], 'val' if split != 'train' else 'train')
        
        print(f"Loading ImageNet from: {data_path}")
        dataset = ImageFolder(root=data_path, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    # the pool is paired into the training split only
    pool_path = getattr(args, "latent_pool", "") if split == 'train' else ""
    if pool_path:
        dataset = PairedLatentDataset(dataset, pool_path)
        print(f"Paired with latent pool: {pool_path}")

    print(f"Dataset: {args.dataset_name}, split: {split}, size: {len(dataset)}")
    
    distributed = torch.distributed.is_initialized()

    num_workers = args.workers if not hasattr(args, 'use_jax') or not args.use_jax else 0
    
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=(split == 'train'))
        dataloader = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            pin_memory=True,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=(split == 'train'),
            persistent_workers=(num_workers > 0),
        )
    else:
        sampler = None
        dataloader = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            pin_memory=True,
            batch_size=args.batch_size,
            shuffle=(split == 'train'),
            drop_last=(split == 'train'),
            persistent_workers=(num_workers > 0),
        )
    
    return dataloader, sampler, len(dataset)


def load_dataset(args, batch_size=16, split='val'):
    """-> (dataloader, dataset size); non-distributed evaluation loader."""
    data_path = os.path.join(args.dataset_dir, DATASET_PATHS[args.dataset_name])
    transform = build_eval_transform(args.resolution)
    
    if args.dataset_name == "ImageNet":
        dataset = ImageFolder(
            root=os.path.join(data_path, 'val' if split == 'val' else 'train'),
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False
    )
    
    return dataloader, len(dataset)
