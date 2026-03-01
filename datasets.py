#!/usr/bin/env python3
"""
datasets.py — Unified dataset builder for ID and OOD splits.

ID datasets
-----------
cifar10, cifar100, svhn, mnist, fashionmnist
    Downloaded automatically via torchvision to --data_dir.

imagenet1k
    Expects an ImageNet-style folder layout.  Pass --imagenet_root pointing
    to the *train* directory (or its parent).  The val split is resolved
    automatically (looks for ``new_val`` then ``val`` as siblings of train).

OOD datasets
------------
Any dataset whose images live in an ImageFolder-compatible directory tree can
be used as OOD by passing ``--ood_dir /path/to/folder``.

For the small torchvision OOD datasets (svhn, cifar100, mnist, fashionmnist)
the ``--data_dir`` path is used instead and download=True is set.

Special ImageFolder OOD datasets (textures, places, sun, inaturalist,
imagenet_o, dtd, etc.) all just need ``--ood_dir`` pointing at the root of
the ImageFolder layout.
"""

import os
from typing import Optional, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ─────────────────────────────────────────────
# ImageNet path resolution
# ─────────────────────────────────────────────

def resolve_imagenet_roots(imagenet_root: str) -> Tuple[str, str]:
    """
    Given a path that may point to the *train* folder or its parent, return
    ``(train_root, val_root)``.  Val root is 'new_val' if present, else 'val'.
    """
    r    = imagenet_root.rstrip("/")
    base = os.path.basename(r)

    if base == "train":
        train_root = r
        parent     = os.path.dirname(r)
    elif os.path.isdir(os.path.join(r, "train")):
        train_root = os.path.join(r, "train")
        parent     = r
    else:
        train_root = r
        parent     = os.path.dirname(r)

    for cand in ("new_val", "val"):
        val_root = os.path.join(parent, cand)
        if os.path.isdir(val_root):
            return train_root, val_root

    raise FileNotFoundError(
        f"Could not find new_val or val under {parent!r}. "
        "Check --imagenet_root."
    )


# ─────────────────────────────────────────────
# Build datasets
# ─────────────────────────────────────────────

_GRAYSCALE_DATASETS = {"mnist", "fashionmnist"}
_TORCHVISION_OOD    = {"svhn", "cifar100", "mnist", "fashionmnist", "cifar10"}


def _maybe_grayscale(name: str, tfm) -> object:
    """Prepend a Grayscale(3) transform for single-channel datasets."""
    if name in _GRAYSCALE_DATASETS:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            tfm,
        ])
    return tfm


def build_id_dataset(
    name: str,
    split: str,
    tfm,
    *,
    data_dir: str = "./data",
    imagenet_root: Optional[str] = None,
):
    """
    Build an ID dataset.

    Parameters
    ----------
    name         : one of cifar10 | cifar100 | svhn | mnist | fashionmnist |
                   imagenet1k
    split        : 'train' or 'test' / 'val'
    tfm          : torchvision transform
    data_dir     : root for torchvision auto-download datasets
    imagenet_root: path to ImageNet (required when name='imagenet1k')
    """
    n = name.lower()

    if n == "cifar10":
        return datasets.CIFAR10(
            root=data_dir, train=(split == "train"),
            download=True, transform=_maybe_grayscale(n, tfm),
        )

    if n == "cifar100":
        return datasets.CIFAR100(
            root=data_dir, train=(split == "train"),
            download=True, transform=_maybe_grayscale(n, tfm),
        )

    if n == "svhn":
        return datasets.SVHN(
            root=data_dir, split=("train" if split == "train" else "test"),
            download=True, transform=tfm,
        )

    if n == "mnist":
        return datasets.MNIST(
            root=data_dir, train=(split == "train"),
            download=True, transform=_maybe_grayscale(n, tfm),
        )

    if n == "fashionmnist":
        return datasets.FashionMNIST(
            root=data_dir, train=(split == "train"),
            download=True, transform=_maybe_grayscale(n, tfm),
        )

    if n == "imagenet1k":
        if imagenet_root is None:
            raise ValueError("imagenet1k requires --imagenet_root.")
        train_root, val_root = resolve_imagenet_roots(imagenet_root)
        root = train_root if split == "train" else val_root
        return datasets.ImageFolder(root=root, transform=tfm)

    raise ValueError(
        f"Unknown ID dataset '{name}'. "
        "Supported: cifar10, cifar100, svhn, mnist, fashionmnist, imagenet1k"
    )


def build_ood_dataset(
    name: str,
    tfm,
    *,
    data_dir: str = "./data",
    ood_dir: Optional[str] = None,
):
    """
    Build an OOD dataset.

    For small torchvision datasets (svhn, cifar100, mnist, fashionmnist,
    cifar10) ``data_dir`` and auto-download are used.

    For everything else (textures, places, sun, dtd, inaturalist,
    imagenet_o, tiny-imagenet-200, or any custom path) pass ``ood_dir``
    pointing at the ImageFolder root.  If ``ood_dir`` already points directly
    at the class sub-folders, it is used as-is; if it contains a sub-folder
    named ``name`` that folder is preferred.
    """
    n = name.lower()

    # ── Torchvision OOD ─────────────────────────────────────────────────────
    if n == "svhn":
        return datasets.SVHN(
            root=data_dir, split="test", download=True,
            transform=tfm,
        )

    if n == "cifar100":
        return datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=tfm,
        )

    if n == "cifar10":
        return datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=tfm,
        )

    if n == "mnist":
        return datasets.MNIST(
            root=data_dir, train=False, download=True,
            transform=transforms.Compose([
                transforms.Grayscale(num_output_channels=3), tfm,
            ]),
        )

    if n == "fashionmnist":
        return datasets.FashionMNIST(
            root=data_dir, train=False, download=True,
            transform=transforms.Compose([
                transforms.Grayscale(num_output_channels=3), tfm,
            ]),
        )

    # ── ImageFolder OOD (requires ood_dir) ──────────────────────────────────
    if ood_dir is None:
        raise ValueError(
            f"OOD dataset '{name}' requires --ood_dir pointing at the "
            "ImageFolder root directory."
        )

    # Some datasets live under a sub-folder named after the dataset
    candidate = os.path.join(ood_dir, name)
    if os.path.isdir(candidate):
        root = candidate
    else:
        root = ood_dir

    # tiny-imagenet-200 uses a 'val' sub-split
    if n == "tiny-imagenet-200":
        val_path = os.path.join(root, "val")
        if os.path.isdir(val_path):
            root = val_path

    return datasets.ImageFolder(root=root, transform=tfm)


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────

def make_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )


def num_classes_from_dataset(ds) -> int:
    """Infer the number of classes from a torchvision dataset."""
    if hasattr(ds, "classes"):
        return len(ds.classes)
    if hasattr(ds, "class_to_idx"):
        return len(ds.class_to_idx)
    raise RuntimeError(
        "Cannot infer num_classes from dataset. "
        "Set --num_classes explicitly."
    )
