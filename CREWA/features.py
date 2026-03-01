#!/usr/bin/env python3
"""
features.py — Unified feature extraction and disk caching.

Two back-ends
-------------
``torch`` back-end  (.pt files)
    Simpler, uses torch.save/load.  Fine for datasets that fit comfortably
    in RAM (CIFAR-10/100, SVHN, MNIST …).

``memmap`` back-end  (.npy files via numpy memmap)
    Memory-mapped numpy arrays — recommended for ImageNet-scale datasets
    where the full feature matrix might be too large to hold in RAM.

The ``extract_or_load_*`` family of functions transparently chooses the cache
path from a consistent naming scheme and either loads existing data or runs
the model over the dataloader to produce it.
"""

import os
import time
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────
# Cache-path helpers
# ─────────────────────────────────────────────

def _safe(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._") else "_" for c in str(s))


def make_cache_paths(
    cache_dir: str,
    arch: str,
    id_name: str,
    ood_name: str,
    l2_normalize: bool,
    train_max_images: int = 0,
    seed: int = 0,
) -> dict:
    """
    Return a dict of canonical cache paths for a given experiment configuration.

    Keys: train_feats, train_labels, id_feats, ood_feats
    Paths use the .npy extension (memmap back-end).
    """
    sub      = "l2" if l2_normalize else "raw"
    root     = os.path.join(cache_dir, sub)
    os.makedirs(root, exist_ok=True)

    arch_k   = _safe(arch)
    id_k     = _safe(id_name)
    ood_k    = _safe(ood_name)

    train_tag = f"tr{int(train_max_images)}__seed{int(seed)}"

    return dict(
        train_feats  = os.path.join(root, f"train_feat_{id_k}__{arch_k}__{train_tag}.npy"),
        train_labels = os.path.join(root, f"train_lbl_{id_k}__{arch_k}__{train_tag}.npy"),
        id_feats     = os.path.join(root, f"id_val_{id_k}__{arch_k}.npy"),
        ood_feats    = os.path.join(root, f"ood_{ood_k}__{arch_k}.npy"),
    )


# ─────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────

def _unwrap_feats(out) -> torch.Tensor:
    """Accept (feats,) | (feats, logits) | feats and return a 2-D feature tensor."""
    feats = out[0] if isinstance(out, (tuple, list)) else out
    if feats.ndim == 3:        # (B, L, D) — pool tokens
        feats = feats.mean(dim=1)
    elif feats.ndim == 4:      # (B, D, H, W) — global avg pool
        feats = feats.mean(dim=(2, 3))
    elif feats.ndim != 2:
        raise ValueError(f"Unexpected feature shape {tuple(feats.shape)}.")
    return feats


def _is_valid_npy(path: str, dtype, ndim: int) -> bool:
    if not os.path.exists(path):
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.dtype == dtype and arr.ndim == ndim
    except Exception:
        return False


# ─────────────────────────────────────────────
# Core extraction routine (memmap back-end)
# ─────────────────────────────────────────────

@torch.inference_mode()
def extract_and_cache_memmap(
    *,
    dataset,
    out_feats_path: str,
    out_labels_path: Optional[str],
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    max_images: int = 0,
    seed: int = 0,
    l2_normalize: bool = False,
    eps: float = 1e-12,
) -> None:
    """
    Extract features from ``dataset`` using ``model`` and write them to disk
    as memory-mapped numpy arrays.

    If valid cache files already exist the function returns immediately
    (cache hit).  Partial writes use a ``.partial`` suffix and are atomically
    renamed on success so interrupted runs leave no corrupt files.
    """
    feats_ok  = _is_valid_npy(out_feats_path, np.float32, 2)
    labels_ok = (out_labels_path is None) or _is_valid_npy(out_labels_path, np.int64, 1)
    if feats_ok and labels_ok:
        print(f"[cache hit] {out_feats_path}")
        return

    from torch.utils.data import RandomSampler

    ds_len = len(dataset)
    if max_images <= 0 or max_images >= ds_len:
        N       = ds_len
        sampler = None
        shuffle = False
    else:
        N   = int(max_images)
        g   = torch.Generator()
        g.manual_seed(seed)
        sampler = RandomSampler(dataset, replacement=False, num_samples=N, generator=g)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    model  = model.to(device)
    model.eval()
    amp_ok = bool(use_amp and device.type == "cuda")

    # infer feature dimension from first batch
    x0, _ = next(iter(loader))
    with torch.cuda.amp.autocast(enabled=amp_ok):
        out0 = model(x0.to(device, non_blocking=True))
    D = int(_unwrap_feats(out0).shape[1])

    feats_tmp  = out_feats_path  + ".partial"
    labels_tmp = (out_labels_path + ".partial") if out_labels_path else None

    feats_mm  = open_memmap(feats_tmp,  mode="w+", dtype=np.float32, shape=(N, D))
    labels_mm = open_memmap(labels_tmp, mode="w+", dtype=np.int64,   shape=(N,)) \
                if labels_tmp else None

    idx = 0
    t0  = time.time()
    for images, labels in loader:
        bsz = int(labels.shape[0])
        if idx + bsz > N:
            bsz    = N - idx
            images = images[:bsz]
            labels = labels[:bsz]

        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_ok):
            out = model(images)
        feats = _unwrap_feats(out).float()

        if l2_normalize:
            feats = F.normalize(feats, p=2, dim=1, eps=eps)

        feats_mm[idx:idx + bsz] = feats.detach().cpu().numpy().astype(np.float32, copy=False)
        if labels_mm is not None:
            labels_mm[idx:idx + bsz] = labels.detach().cpu().long().numpy()[:bsz]

        idx += bsz
        if idx >= N:
            break

    feats_mm.flush()
    if labels_mm is not None:
        labels_mm.flush()

    os.replace(feats_tmp, out_feats_path)
    if labels_tmp:
        os.replace(labels_tmp, out_labels_path)

    dt = time.time() - t0
    print(f"[cache write] N={N} D={D} l2={l2_normalize} time={dt:.1f}s → {out_feats_path}")


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def load_memmap_as_torch(path: str) -> torch.Tensor:
    """Load a .npy memmap and return a Torch tensor (zero-copy on CPU)."""
    return torch.from_numpy(np.load(path, mmap_mode="r"))


def load_memmap_as_numpy(path: str) -> np.memmap:
    return np.load(path, mmap_mode="r")


def extract_all_features(
    *,
    model: nn.Module,
    train_dataset,
    id_dataset,
    ood_dataset,
    paths: dict,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    l2_normalize: bool,
    train_max_images: int = 0,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract (or load from cache) features for all three splits.

    Returns
    -------
    train_feats  : [N_train, D]
    train_labels : [N_train]
    id_feats     : [N_id, D]
    ood_feats    : [N_ood, D]

    All tensors are CPU float32 (memmap-backed, no data copy).
    """
    common = dict(
        model=model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        use_amp=use_amp,
        l2_normalize=l2_normalize,
        seed=seed,
    )

    print("\n[features] Extracting / loading cached features …")

    # Train (feats + labels)
    extract_and_cache_memmap(
        dataset=train_dataset,
        out_feats_path=paths["train_feats"],
        out_labels_path=paths["train_labels"],
        max_images=train_max_images,
        **common,
    )

    # ID test/val (feats only)
    extract_and_cache_memmap(
        dataset=id_dataset,
        out_feats_path=paths["id_feats"],
        out_labels_path=None,
        **common,
    )

    # OOD test (feats only)
    extract_and_cache_memmap(
        dataset=ood_dataset,
        out_feats_path=paths["ood_feats"],
        out_labels_path=None,
        **common,
    )

    train_feats  = load_memmap_as_torch(paths["train_feats"])
    train_labels = load_memmap_as_torch(paths["train_labels"])
    id_feats     = load_memmap_as_torch(paths["id_feats"])
    ood_feats    = load_memmap_as_torch(paths["ood_feats"])

    print(f"  train_feats  : {tuple(train_feats.shape)}")
    print(f"  train_labels : {tuple(train_labels.shape)}")
    print(f"  id_feats     : {tuple(id_feats.shape)}")
    print(f"  ood_feats    : {tuple(ood_feats.shape)}")

    return train_feats, train_labels, id_feats, ood_feats


# ─────────────────────────────────────────────
# Batched logit computation from cached features
# ─────────────────────────────────────────────

@torch.no_grad()
def logits_from_feats_batched(
    feats: Union[torch.Tensor, np.memmap],
    W: torch.Tensor,
    b: Optional[torch.Tensor],
    *,
    device: torch.device,
    batch_size: int = 65536,
) -> torch.Tensor:
    """
    Compute logits = feats @ W.T + b in batches (GPU-accelerated).
    Accepts both torch.Tensor and numpy memmap as feats.
    Returns a CPU float32 tensor of shape [N, C].
    """
    if isinstance(feats, np.memmap):
        feats = torch.from_numpy(feats)

    N  = feats.shape[0]
    C  = int(W.shape[0])
    Wd = W.to(device).float()
    bd = b.to(device).float() if b is not None else None
    out = torch.empty((N, C), dtype=torch.float32, device="cpu")

    for i in range(0, N, batch_size):
        x = feats[i:i + batch_size].to(device, non_blocking=True).float()
        y = x @ Wd.t()
        if bd is not None:
            y = y + bd
        out[i:i + y.shape[0]] = y.detach().cpu()

    return out


# ─────────────────────────────────────────────
# Gaussian feature extraction (for NCI tuning)
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_gaussian_feats_like_id(
    model: nn.Module,
    id_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> torch.Tensor:
    """
    Pass Gaussian noise through ``model`` using the same batch shapes as
    ``id_loader``.  Returns [N, D] CPU float32 tensor.

    Used for NCI's varpi tuning step.
    """
    model.eval()
    outs    = []
    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp)
    for x, _ in id_loader:
        x  = x.to(device, non_blocking=True)
        xg = torch.randn_like(x)
        with amp_ctx:
            out   = model(xg)
            feats = _unwrap_feats(out)
        outs.append(feats.detach().float().cpu())
    return torch.cat(outs, dim=0)
