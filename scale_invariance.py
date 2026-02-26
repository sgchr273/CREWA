#!/usr/bin/env python3
from __future__ import annotations
import os
import time
import argparse
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from numpy.lib.format import open_memmap
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

from torchvision.models import resnet50, ResNet50_Weights, swin_t, Swin_T_Weights

from typing import Optional, Union, Tuple, List, Any
import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from numpy.lib.format import open_memmap

from sklearn.metrics import roc_auc_score, roc_curve

from typing import Sequence, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_curve, roc_auc_score


# -------------------------
# Metrics
# -------------------------
def auroc_and_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> Tuple[float, float]:
    y = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)]).astype(np.int64)
    s = np.concatenate([id_scores, ood_scores]).astype(np.float64)
    auroc = float(roc_auc_score(y, s))
    fpr, tpr, _ = roc_curve(y, s)
    if (tpr >= 0.95).any():
        idx = int(np.argmax(tpr >= 0.95))
        fpr95 = float(fpr[idx])
    else:
        fpr95 = 1.0
    return auroc, fpr95

def fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray, tpr_level: float = 0.95) -> float:
    """
    Compute FPR at a given TPR level.
    Assumes larger score => more OOD.
    Labels: ID=0, OOD=1.
    """
    y = np.concatenate([np.zeros_like(id_scores, dtype=np.int32),
                        np.ones_like(ood_scores, dtype=np.int32)])
    s = np.concatenate([id_scores, ood_scores]).astype(np.float64)

    fpr, tpr, _ = roc_curve(y, s)
    # First index where TPR >= desired level
    idx = np.searchsorted(tpr, tpr_level, side="left")
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])

def report_metrics(name: str, id_scores: np.ndarray, ood_scores: np.ndarray) -> None:
    auroc, fpr95 = auroc_and_fpr95(id_scores, ood_scores)
    print(f"\n========== {name} ==========")
    print(f"AUROC (OOD positive):      {auroc:.4f}")
    print(f"FPR@95%TPR (OOD positive): {fpr95:.4f}")
    print(f"(score means) ID  mean={id_scores.mean():.6f} std={id_scores.std():.6f}")
    print(f"(score means) OOD mean={ood_scores.mean():.6f} std={ood_scores.std():.6f}")
    print("====================================\n")

# def report_metrics(method_name: str, id_scores: np.ndarray, ood_scores: np.ndarray):
#     """
#     Backward compatible: prints the same metrics for all methods.
#     For NCI only, also RETURNS AUROC (OOD positive) so you can tune varpi.
#     For other methods, returns None (no behavior change needed).
#     """
#     scores = np.concatenate([id_scores, ood_scores], axis=0)
#     y_true = np.concatenate(
#         [np.zeros_like(id_scores, dtype=np.int64),
#          np.ones_like(ood_scores, dtype=np.int64)],
#         axis=0,
#     )

#     auroc = roc_auc_score(y_true, scores)
#     fpr95 = fpr_at_tpr(y_true, scores, tpr_level=0.95)

#     print(f"\n========== {method_name.upper()} ==========")
#     print(f"AUROC (OOD positive):      {auroc:.4f}")
#     print(f"FPR@95%TPR (OOD positive): {fpr95:.4f}")
#     print(f"(score means) ID  mean={id_scores.mean():.6f} std={id_scores.std():.6f}")
#     print(f"(score means) OOD mean={ood_scores.mean():.6f} std={ood_scores.std():.6f}")
#     print("======================================")

#     # Only NCI returns AUROC; other methods unchanged (still print only).
#     if method_name.strip().lower() in {"nci", "nci-temp", "nci_score"}:
#         return float(auroc)
#     return None


def ensure_ood_higher(id_scores: np.ndarray, ood_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Enforce "OOD higher" convention
    if float(np.mean(ood_scores)) < float(np.mean(id_scores)):
        return -id_scores, -ood_scores
    return id_scores, ood_scores


# -------------------------
# Model wrappers
# -------------------------
class ResNetWithFeats(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor):
        bb = self.backbone
        x = bb.conv1(x)
        x = bb.bn1(x)
        x = bb.relu(x)
        x = bb.maxpool(x)
        x = bb.layer1(x)
        x = bb.layer2(x)
        x = bb.layer3(x)
        x = bb.layer4(x)
        x = bb.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = bb.fc(feats)
        return feats, logits


class SwinWithFeats(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor):
        bb = self.backbone

        # ---- timm-style Swin ----
        if hasattr(bb, "forward_features"):
            feats = bb.forward_features(x)
            if feats.ndim == 3:          # (B, L, C)
                feats = feats.mean(dim=1)
            elif feats.ndim == 4:        # (B, C, H, W)
                feats = feats.mean(dim=(2, 3))
            logits = bb.head(feats) if hasattr(bb, "head") and bb.head is not None else None
            return feats, logits

        # ---- torchvision-style Swin ----
        if hasattr(bb, "features") and hasattr(bb, "norm"):
            z = bb.features(x)  # could be BxHxWxC OR BxCxHxW OR BxHxCxW, etc.

            # Move channel dim (C=normalized_shape) to last so LayerNorm works.
            if z.ndim == 4:
                C = None
                if hasattr(bb.norm, "normalized_shape"):
                    # normalized_shape can be int or tuple like (768,)
                    ns = bb.norm.normalized_shape
                    C = int(ns[0]) if isinstance(ns, (tuple, list)) else int(ns)

                if C is not None and z.shape[-1] != C:
                    # find which dim is the channel dim among {1,2,3}
                    ch_dim = None
                    for d in (1, 2, 3):
                        if z.shape[d] == C:
                            ch_dim = d
                            break
                    if ch_dim is None:
                        raise ValueError("Cannot locate channel dim for Swin features shape {}".format(tuple(z.shape)))

                    # permute to (B, H, W, C)
                    if ch_dim == 1:      # (B, C, H, W) -> (B, H, W, C)
                        z = z.permute(0, 2, 3, 1)
                    elif ch_dim == 2:    # (B, H, C, W) -> (B, H, W, C)  <-- your case [B,7,768,7]
                        z = z.permute(0, 1, 3, 2)
                    else:
                        # ch_dim == 3 already (B, H, W, C)
                        pass

            # Now last dim is C, so LayerNorm is valid
            z = bb.norm(z)                 # (B, H, W, C)
            feats = z.mean(dim=(1, 2))     # global average pool -> (B, C)
            logits = bb.head(feats) if hasattr(bb, "head") and bb.head is not None else None
            return feats, logits

        raise ValueError("Unsupported Swin backbone: need timm forward_features or torchvision features+norm.")
def build_backbone(arch: str):
    a = arch.lower().replace("-", "_")
    if a == "resnet50":
        w = ResNet50_Weights.IMAGENET1K_V2
        backbone = resnet50(weights=w)
        return ResNetWithFeats(backbone), w.transforms()
    if a in ("swint", "swin_t"):
        w = Swin_T_Weights.IMAGENET1K_V1
        backbone = swin_t(weights=w)
        return SwinWithFeats(backbone), w.transforms()
    raise ValueError("arch must be one of: resnet50, swint (or swin_t)")


def get_classifier_Wb(model: nn.Module):
    bb = model.backbone
    if hasattr(bb, "fc") and isinstance(bb.fc, nn.Linear):
        W = bb.fc.weight.detach().to("cpu").float()
        b = bb.fc.bias.detach().to("cpu").float() if bb.fc.bias is not None else None
        return W, b
    if hasattr(bb, "head") and isinstance(bb.head, nn.Linear):
        W = bb.head.weight.detach().to("cpu").float()
        b = bb.head.bias.detach().to("cpu").float() if bb.head.bias is not None else None
        return W, b
    raise ValueError("Cannot find classifier Linear layer (expected fc or head).")


# -------------------------
# Caching helpers (only feats + optional labels)
# -------------------------
def _safe_name(x: str) -> str:
    x = str(x)
    return "".join([c if (c.isalnum() or c in "._") else "_" for c in x])


def _cache_root(cache_dir: str, l2_normalize: bool) -> str:
    sub = "l2" if l2_normalize else "raw"
    out = os.path.join(cache_dir, sub)
    os.makedirs(out, exist_ok=True)
    return out


def _paths_train(cache_dir: str, arch: str, l2_normalize: bool, train_max_images: int, seed: int) -> Tuple[str, str]:
    root = _cache_root(cache_dir, l2_normalize)
    arch = _safe_name(arch)
    tag = f"tr{int(train_max_images)}__seed{int(seed)}"
    feats = os.path.join(root, f"train_feat_{arch}__{tag}.npy")
    labels = os.path.join(root, f"train_lbl_{arch}__{tag}.npy")
    return feats, labels


def _path_feats(cache_dir: str, name: str, arch: str, l2_normalize: bool) -> str:
    root = _cache_root(cache_dir, l2_normalize)
    name = _safe_name(name)
    arch = _safe_name(arch)
    return os.path.join(root, f"{name}_{arch}.npy")


def _is_cache_hit(path: str, dtype, ndim: int) -> bool:
    if not os.path.exists(path):
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return (arr.dtype == dtype) and (arr.ndim == ndim)
    except Exception:
        return False




def _unwrap_feats(out: Union[torch.Tensor, Tuple[Any, ...], List[Any]]) -> torch.Tensor:
    """
    Accept either:
      - feats
      - (feats, logits)
    and coerce to BxD by pooling if needed.
    """
    feats = out[0] if isinstance(out, (tuple, list)) else out
    if isinstance(feats, dict):
        # handle rare dict outputs
        for k in ("feats", "features", "x", "penultimate"):
            if k in feats:
                feats = feats[k]
                break
        else:
            raise TypeError("Dict output from model is unsupported; add a key mapping in _unwrap_feats().")

    if not torch.is_tensor(feats):
        raise TypeError("Model output is not a tensor/tuple/list of tensors.")

    # Swin sometimes yields token-like or map-like tensors depending on wrapper
    if feats.ndim == 3:          # (B, L, D)
        feats = feats.mean(dim=1)
    elif feats.ndim == 4:        # (B, D, H, W)
        feats = feats.mean(dim=(2, 3))
    elif feats.ndim != 2:        # expected (B, D)
        raise ValueError("Unsupported feat shape {} (expected 2D/3D/4D).".format(tuple(feats.shape)))

    return feats


@torch.inference_mode()
def cache_feats_and_optional_labels_float32(
    *,
    ds,
    out_feats_path: str,
    out_labels_path: Optional[str],
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    workers: int,
    use_amp: bool,
    max_images: int,
    seed: int,
    l2_normalize: bool,
    eps: float = 1e-12,
) -> None:
    feats_ok = _is_cache_hit(out_feats_path, np.float32, 2)
    labels_ok = True if out_labels_path is None else _is_cache_hit(out_labels_path, np.int64, 1)
    if feats_ok and labels_ok:
        print("[cache] hit feats: {}".format(out_feats_path))
        if out_labels_path is not None:
            print("[cache] hit labels: {}".format(out_labels_path))
        return

    ds_len = len(ds)
    if max_images <= 0 or max_images > ds_len:
        N = ds_len
        sampler = None
        shuffle = False
    else:
        N = int(max_images)
        g = torch.Generator()
        g.manual_seed(int(seed))
        sampler = RandomSampler(ds, replacement=False, num_samples=N, generator=g)
        shuffle = False

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        drop_last=False,
    )

    model = model.to(device)
    model.eval()
    amp_ok = bool(use_amp and device.type == "cuda")

    # ---- infer D (robust for both ResNetWithFeats and SwinWithFeats) ----
    x0, _ = next(iter(loader))
    x0 = x0.to(device, non_blocking=True)
    with torch.cuda.amp.autocast(enabled=amp_ok):
        out0 = model(x0)
    f0 = _unwrap_feats(out0)
    D = int(f0.shape[1])

    feats_tmp = out_feats_path + ".partial"
    labels_tmp = (out_labels_path + ".partial") if out_labels_path is not None else None

    feats_mm = open_memmap(feats_tmp, mode="w+", dtype=np.float32, shape=(N, D))
    labels_mm = open_memmap(labels_tmp, mode="w+", dtype=np.int64, shape=(N,)) if labels_tmp else None

    idx = 0
    t0 = time.time()

    for images, labels in loader:
        bsz = int(labels.shape[0])
        if idx + bsz > N:
            bsz = N - idx
            images = images[:bsz]
            labels = labels[:bsz]

        images = images.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp_ok):
            out = model(images)
        feats = _unwrap_feats(out)

        feats = feats.float()
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
    if labels_tmp is not None:
        os.replace(labels_tmp, out_labels_path)

    dt = time.time() - t0
    print("[cache] wrote feats: N={} D={} l2={} time_s={:.1f} -> {}".format(
        N, D, bool(l2_normalize), dt, out_feats_path
    ))
    if out_labels_path is not None:
        print("[cache] wrote labels: N={} time_s={:.1f} -> {}".format(
            N, dt, out_labels_path
        ))

def load_memmap(path: str) -> np.memmap:
    return np.load(path, mmap_mode="r")


@torch.no_grad()
def logits_from_feats_batched(
    feats_mm: np.memmap,
    W: torch.Tensor,
    b: Optional[torch.Tensor],
    *,
    device: torch.device,
    batch_size: int = 65536,
) -> torch.Tensor:
    Wd = W.to(device, non_blocking=True).float()
    bd = b.to(device, non_blocking=True).float() if b is not None else None

    N = feats_mm.shape[0]
    C = int(Wd.shape[0])
    out = torch.empty((N, C), dtype=torch.float32, device="cpu")

    for i in range(0, N, batch_size):
        x = torch.from_numpy(feats_mm[i:i + batch_size]).to(device, non_blocking=True).float()
        y = x @ Wd.t()
        if bd is not None:
            y = y + bd
        out[i:i + y.shape[0]] = y.detach().cpu()

    return out


# -------------------------
# Dataset path logic
# -------------------------
def resolve_imagenet_train_val(imagenet_root: str) -> Tuple[str, str]:
    r = imagenet_root.rstrip("/")
    base = os.path.basename(r)

    # Try to locate train root
    if base == "train":
        train_root = r
        parent = os.path.dirname(r)
        cand_new = os.path.join(parent, "new_val")
        cand_val = os.path.join(parent, "val")
    elif os.path.isdir(os.path.join(r, "train")):
        train_root = os.path.join(r, "train")
        cand_new = os.path.join(r, "new_val")
        cand_val = os.path.join(r, "val")
    else:
        # if user points directly at train folder-like root
        train_root = r
        parent = os.path.dirname(r)
        cand_new = os.path.join(parent, "new_val")
        cand_val = os.path.join(parent, "val")

    if os.path.isdir(cand_new):
        val_root = cand_new
    elif os.path.isdir(cand_val):
        val_root = cand_val
    else:
        raise FileNotFoundError(f"Could not find new_val or val. Tried: {cand_new} and {cand_val}")

    return train_root, val_root


def build_ood_dataset(ood_name: str, ood_dir: Optional[str], tfm):
    if ood_dir is None:
        raise ValueError("--ood_dir is required (expects ImageFolder layout).")
    return datasets.ImageFolder(root=ood_dir, transform=tfm)

import os, re, glob
import numpy as np
import torch


# -------------------------
# memmap loader (yours can replace this)
# -------------------------
def load_memmap(path: str) -> np.memmap:
    """
    Loads a numpy file as memmap without copying.
    Works if the file was saved as .npy (including memmap saved via np.save/open_memmap).
    """
    return np.load(path, mmap_mode="r")


# -------------------------
# helpers: robust file finding
# -------------------------
def _norm_key(s: str) -> str:
    # normalize for matching: lower + remove non-alphanum
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _find_one(folder: str, include_tokens, *, must_endwith=".npy") -> str:
    """
    Find exactly one file in folder whose normalized name contains all include_tokens.
    include_tokens: list[str] tokens (already normalized)
    Returns best match (shortest filename) to avoid accidental superset matches.
    """
    files = glob.glob(os.path.join(folder, f"*{must_endwith}"))
    cand = []
    for f in files:
        base = os.path.basename(f)
        key = _norm_key(base)
        ok = True
        for t in include_tokens:
            if t not in key:
                ok = False
                break
        if ok:
            cand.append(f)

    if not cand:
        raise FileNotFoundError(
            f"No match in {folder} for tokens={include_tokens}. "
            f"Available examples: {[os.path.basename(x) for x in files[:10]]}"
        )

    # choose "best" match: shortest basename (usually the most specific canonical file)
    cand.sort(key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
    return cand[0]


def resolve_feature_paths(
    root_dir: str,
    *,
    arch: str,
    ood_name: str,
    id_split: str = "id_val",   # prefer id_val; fallback to id_test
    feature_subdir: str = "",   # if your files are directly in root_dir; else e.g., "test/l2"
) -> dict:
    """
    Returns dict with paths: train_feats_path, train_lbls_path, id_feats_path, ood_feats_path

    Expected (flexible) filenames in the folder, case-insensitive:
      - train_feat_{arch}... .npy
      - train_lbl_{arch}...  .npy
      - id_val_{arch}.npy or id_test_{arch}.npy
      - {ood_name}_{arch}.npy  (also handles openimage_o / imagenet_o, etc.)
    """
    folder = os.path.join(root_dir, feature_subdir) if feature_subdir else root_dir

    arch_k = _norm_key(arch)          # e.g., "resnet50", "swint"
    ood_k  = _norm_key(ood_name)      # e.g., "sun", "places", "inaturalist"

    # train feats/labels (these have extra tokens like tr0, seed0, etc.)
    train_feats_path = _find_one(folder, ["trainfeat", arch_k])
    train_lbls_path  = _find_one(folder, ["trainlbl",  arch_k])

    # ID split: prefer id_val; fallback to id_test
    id_split_k = _norm_key(id_split)  # "idval"
    try:
        id_feats_path = _find_one(folder, [id_split_k, arch_k])
    except FileNotFoundError:
        # fallback to id_test if id_val not found
        id_feats_path = _find_one(folder, [_norm_key("id_test"), arch_k])

    # OOD feats
    # Some datasets have variants, but you are passing the exact "ood_name" you want.
    ood_feats_path = _find_one(folder, [ood_k, arch_k])

    return dict(
        train_feats_path=train_feats_path,
        train_lbls_path=train_lbls_path,
        id_feats_path=id_feats_path,
        ood_feats_path=ood_feats_path,
    )


def load_cached_feats_as_torch(
    root_dir: str,
    *,
    arch: str,
    ood_name: str,
    id_split: str = "id_val",
    feature_subdir: str = "",
    device: torch.device | None = None,
):
    """
    Loads memmap-backed numpy arrays and returns torch tensors (views, no copy).

    Returns:
      train_feats, train_labels, id_feats, ood_feats, paths_dict
    """
    paths = resolve_feature_paths(
        root_dir,
        arch=arch,
        ood_name=ood_name,
        id_split=id_split,
        feature_subdir=feature_subdir,
    )

    train_feats_mm  = load_memmap(paths["train_feats_path"])
    train_labels_mm = load_memmap(paths["train_lbls_path"])
    id_feats_mm     = load_memmap(paths["id_feats_path"])
    ood_feats_mm    = load_memmap(paths["ood_feats_path"])

    # Torch views (memmap-backed)
    train_feats  = torch.from_numpy(train_feats_mm)
    train_labels = torch.from_numpy(train_labels_mm)
    id_feats     = torch.from_numpy(id_feats_mm)
    ood_feats    = torch.from_numpy(ood_feats_mm)

    # Optional: move to device (NOTE: this makes a copy if device is CUDA)
    if device is not None:
        train_feats  = train_feats.to(device)
        train_labels = train_labels.to(device)
        id_feats     = id_feats.to(device)
        ood_feats    = ood_feats.to(device)

    return train_feats, train_labels, id_feats, ood_feats, paths
def load_cached_feats_as_numpy(
    root_dir: str,
    *,
    arch: str,
    ood_name: str,
    id_split: str = "id_val",
    feature_subdir: str = "",
):
    paths = resolve_feature_paths(
        root_dir,
        arch=arch,
        ood_name=ood_name,
        id_split=id_split,
        feature_subdir=feature_subdir,
    )

    train_feats_mm  = load_memmap(paths["train_feats_path"])
    train_labels_mm = load_memmap(paths["train_lbls_path"])
    id_feats_mm     = load_memmap(paths["id_feats_path"])
    ood_feats_mm    = load_memmap(paths["ood_feats_path"])

    return train_feats_mm, train_labels_mm, id_feats_mm, ood_feats_mm, paths





# -------------------------
# Main
# -------------------------

ap = argparse.ArgumentParser()
##########places, sun, inaturalist,  dtd, imagenet_o
ap.add_argument("--imagenet_root", type=str, default="/cluster/pixstor/madrias-lab/Shreen/Angle_method/datasets/id_data/ILSVRC/Data/CLS-LOC/train")
ap.add_argument("--ood_name", type=str, default="sun")   #iNaturalist, imagenet-o
ap.add_argument("--ood_dir", type=str 
,default="/cluster/pixstor/madrias-lab/Shreen/Angle_method/datasets/ood_data/dtd/images")
# , default="/cluster/pixstor/madrias-lab/Shreen/Subspaces/imagenet-o")

ap.add_argument("--arch", type=str, default="swint", help="resnet50 | swint")
ap.add_argument("--device", type=str, default="cuda")
ap.add_argument("--batch_size", type=int, default=512)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--no_amp", action="store_true")
ap.add_argument("--seed", type=int, default=0)

ap.add_argument("--cache_dir", type=str, default="/home/sgchr/Documents/Subspaces/test/l2")
ap.add_argument("--train_max_images", type=int, default=0)
ap.add_argument("--no_l2", action="store_true", help="Disable L2 normalization for cached features")

# method runner
# ap.add_argument("--methods", nargs="+", required=True)

# your method params
ap.add_argument("--sub_energy_keep", type=float, default=0.3)
ap.add_argument("--sub_ipca_components", type=int, default=2048)
ap.add_argument("--sub_ipca_batch", type=int, default=4096)
ap.add_argument("--chunk", type=int, default=8192)

ap.add_argument("--energy_T", type=float, default=1.0)

ap.add_argument("--gate_energy_keep", type=float, default=0.95)
ap.add_argument("--gate_k_max", type=int, default=150)
ap.add_argument("--gate_eps", type=float, default=1e-6)
ap.add_argument("--gate_threshold", type=float, default=0.0)

ap.add_argument("--kpca_gamma", type=float, default=1.0)
ap.add_argument("--kpca_M", type=int, default=4096)
ap.add_argument("--kpca_exp_var_ratio", type=float, default=0.95)

ap.add_argument("--grad_n", type=int, default=4096)
ap.add_argument("--grad_exp_var_ratio", type=float, default=0.95)
ap.add_argument("--grad_center", action="store_true")
ap.add_argument("--grad_eps", type=float, default=1e-6)

ap.add_argument("--neco_dim", type=int, default=128)
ap.add_argument("--vim_dim", type=int, default=0)
ap.add_argument("--vim_fit_max", type=int, default=200000)
ap.add_argument("--vim_fit_device", type=str, default="cpu")
ap.add_argument("--vim_score_device", type=str, default="cpu")


ap.add_argument("--normalize_signals", action="store_true",
            help="Z-score s_res and alignment a using ID calibration stats before combining.")
ap.add_argument("--beta_auto", action="store_true",
                help="Calibrate beta on ID only (tail-match) to prevent alignment term domination.")
ap.add_argument("--beta_q", type=float, default=0.99,
                help="Quantile for ID-only beta calibration when --beta_auto is set (default: 0.99).")


args = ap.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

use_amp = (not args.no_amp) and torch.cuda.is_available()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
l2_normalize = (not args.no_l2)

model, tfm = build_backbone(args.arch)
model.eval().to(device)
num_classes = 1000


train_feats, train_labels, id_feats, ood_feats, paths = load_cached_feats_as_torch(
    root_dir="/home/sgchr/Documents/Subspaces/test/l2",
    arch=args.arch,
    ood_name=args.ood_name,
    id_split="id_val",   # will fallback to id_test automatically if id_val missing
)
train_feats_mm, train_labels_mm, id_feats_mm, ood_feats_mm, paths = load_cached_feats_as_numpy(
root_dir="/home/sgchr/Documents/Subspaces/test/l2",
arch=args.arch,
ood_name=args.ood_name,
)



# Classifier head
W, b = get_classifier_Wb(model)

# Load your methods
import new_methods as M

logits_id = logits_from_feats_batched(id_feats_mm, W, b, device=device, batch_size=65536)
logits_ood = logits_from_feats_batched(ood_feats_mm, W, b, device=device, batch_size=65536)

id_scores, ood_scores = M.run_method_subspaces_simple(
    train_feats=train_feats,
    train_labels=train_labels,
    id_feats=id_feats,
    id_logits=logits_id,
    ood_feats=ood_feats,
    ood_logits=logits_ood,
    num_classes=num_classes,
    energy_keep=args.sub_energy_keep,   # e.g. 0.8
    k_max=0,
    W=W, 
    beta=0,
    tune_beta=True,
    gamma=1,
    use_centered_for_cos=True,
)
report_metrics("subspaces_nc", id_scores, ood_scores)
# id_scores_np, ood_scores_np, beta_star, tau_star = M.run_subspaces_s1_id_tuned_beta(

#####places
####dtd
#####iNaturalist
####sun
###openimage-o