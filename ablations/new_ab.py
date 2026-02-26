#!/usr/bin/env python3
"""
Ablation script for pooled-within-class PCA subspace OOD method.

Produces 3 figures (dpi=300), each averaged over 4 OOD datasets:
  1) energy_keep sweep
  2) centering ablation: class vs global vs none
  3) scoring rule ablation: predicted-class vs min-over-classes

ID dataset: CIFAR-10 or CIFAR-100 (controlled by --id_dataset)

OOD suite (always evaluated, averaged):
  - tiny-imagenet-200
  - svhn
  - textures
  - places

What this script expects on disk:
  --ood_dir must contain folders:
    {ood_dir}/textures/...
    {ood_dir}/places/...
    {ood_dir}/tiny-imagenet-200/val/...

  Each ImageFolder folder must have class-subfolders (standard ImageFolder layout).
  SVHN is downloaded automatically to --data_dir.

Checkpoint expectation:
  --ckpt is a state_dict for a classifier with the same arch as --arch,
  with final layer sized to the ID classes (10 for CIFAR10, 100 for CIFAR100).
"""

import os
import copy
import random
import argparse
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    resnet18, resnet34,
    ResNet18_Weights, ResNet34_Weights,
    vit_b_16, ViT_B_16_Weights
)

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

plt.rcParams.update({
   "font.family": "DejaVu Serif",
})

# -------------------------
# Repro
# -------------------------
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Metrics helpers
# -------------------------
def orient_scores_higher_is_ood(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    flipped = False
    if float(ood_scores.mean()) < float(id_scores.mean()):
        id_scores = -id_scores
        ood_scores = -ood_scores
        flipped = True
    return id_scores, ood_scores, flipped


def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, tpr_level: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    mask = tpr >= tpr_level
    if not np.any(mask):
        return float("nan")
    return float(np.min(fpr[mask]))


def compute_auroc_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> Tuple[float, float]:
    id_s, ood_s, _ = orient_scores_higher_is_ood(id_scores.copy(), ood_scores.copy())
    y = np.concatenate(
        [np.zeros_like(id_s, dtype=np.int64), np.ones_like(ood_s, dtype=np.int64)],
        axis=0,
    )
    s = np.concatenate([id_s, ood_s], axis=0)
    auroc = float(roc_auc_score(y, s))
    fpr95 = float(fpr_at_tpr(y, s, tpr_level=0.95))
    return auroc, fpr95


def aggregate_over_ood_suite(
    id_scores: np.ndarray,
    ood_scores_dict: Dict[str, np.ndarray],
) -> Dict[str, object]:
    per = {}
    aurocs, fprs = [], []
    for name, ood_scores in ood_scores_dict.items():
        au, fp = compute_auroc_fpr95(id_scores, ood_scores)
        per[name] = {"auroc": au, "fpr95": fp}
        aurocs.append(au)
        fprs.append(fp)
    return {
        "per_ood": per,
        "mean_auroc": float(np.mean(aurocs)),
        "mean_fpr95": float(np.mean(fprs)),
        "std_auroc": float(np.std(aurocs)),
        "std_fpr95": float(np.std(fprs)),
    }


# -------------------------
# Small utils
# -------------------------
def safe_name(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ["_", "-", "."]:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


# -------------------------
# Data
# -------------------------
def build_dataset(
    name: str,
    split: str,
    data_dir: str,
    tfm,
    ood_dir: Optional[str] = None,
):
    name = name.lower()

    if name == "cifar10":
        train = (split == "train")
        return datasets.CIFAR10(root=data_dir, train=train, download=True, transform=tfm)

    if name == "cifar100":
        train = (split == "train")
        return datasets.CIFAR100(root=data_dir, train=train, download=True, transform=tfm)

    if name == "svhn":
        # split: "train" or "test"
        return datasets.SVHN(root=data_dir, split=split, download=True, transform=tfm)

    if name in ["textures", "places"]:
        if ood_dir is None:
            raise ValueError("For textures/places, pass --ood_dir with those folders inside.")
        return datasets.ImageFolder(os.path.join(ood_dir, name), transform=tfm)

    if name == "tiny-imagenet-200":
        if ood_dir is None:
            raise ValueError("For tiny-imagenet-200, pass --ood_dir with tiny-imagenet-200 folder inside.")
        return datasets.ImageFolder(os.path.join(ood_dir, name, "val"), transform=tfm)

    raise ValueError(f"Unknown dataset: {name}")


# -------------------------
# Model loading + feature extraction
# -------------------------
def load_state_dict_flex(ckpt_path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ["model_state", "model", "state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        is_state_dict = all(isinstance(v, torch.Tensor) for v in obj.values())
        if is_state_dict:
            return obj
    raise RuntimeError("Could not interpret checkpoint format as a state_dict.")


def build_classifier(
    arch: str,
    num_classes: int,
    ckpt_path: str,
    device: torch.device,
) -> nn.Module:
    arch = arch.lower()

    if arch == "resnet18":
        model = resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        sd = load_state_dict_flex(ckpt_path)
        model.load_state_dict(sd, strict=True)
        return model.to(device)

    if arch == "resnet34":
        model = resnet34(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        sd = load_state_dict_flex(ckpt_path)
        model.load_state_dict(sd, strict=True)
        return model.to(device)

    if arch in ["vit_b_16", "vitb16", "vit-b-16"]:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads = nn.Sequential(nn.Linear(in_features, num_classes))
        sd = load_state_dict_flex(ckpt_path)
        model.load_state_dict(sd, strict=True)
        return model.to(device)

    raise ValueError(f"Unknown --arch {arch}")


def make_penultimate_extractor(model: nn.Module, arch: str, l2_normalize: bool = False) -> nn.Module:
    arch = arch.lower()
    feat_model = copy.deepcopy(model).eval()

    if arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        feat_model.fc = nn.Identity()
    elif arch in ["vit_b_16", "vitb16", "vit-b-16", "vit_b_32", "vitb32", "vit-b-32"]:
        feat_model.heads = nn.Identity()
    else:
        raise ValueError(f"Unknown arch for penultimate extractor: {arch}")

    if not l2_normalize:
        return feat_model

    class L2Wrap(nn.Module):
        def __init__(self, m: nn.Module):
            super().__init__()
            self.m = m

        def forward(self, x):
            f = self.m(x)
            return F.normalize(f, p=2, dim=1)

    return L2Wrap(feat_model)


def get_classifier_linear(clf_model: nn.Module, arch: str) -> Tuple[torch.Tensor, torch.Tensor]:
    arch = arch.lower()
    if arch in ["resnet18", "resnet34"]:
        head = clf_model.fc
    elif arch in ["vit_b_16", "vitb16", "vit-b-16"]:
        head = clf_model.heads[0] if isinstance(clf_model.heads, nn.Sequential) else clf_model.heads
    else:
        raise ValueError(f"Unknown arch for classifier head extraction: {arch}")

    if not isinstance(head, nn.Linear):
        raise RuntimeError("Classifier head is not nn.Linear.")

    W = head.weight.detach().cpu().float()
    b = head.bias.detach().cpu().float()
    return W, b


@torch.no_grad()
def extract_features(
    feat_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feat_model.eval()
    all_feats, all_labels = [], []
    for imgs, labs in loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = feat_model(imgs)
        all_feats.append(feats.detach().cpu().float())
        all_labels.append(labs.detach().cpu().long())
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


# -------------------------
# Caching features
# -------------------------
def cache_paths(args) -> Tuple[str, str, str]:
    os.makedirs(args.cache_dir, exist_ok=True)

    ckpt_tag = safe_name(os.path.basename(args.ckpt))
    id_tag   = safe_name(args.id_dataset)
    arch_tag = safe_name(args.arch)
    l2_tag = "l2" if (not args.no_l2) else "nol2"

    id_base = f"{id_tag}__{arch_tag}__{l2_tag}__{ckpt_tag}"
    p_train = os.path.join(args.cache_dir, id_base + "__id_train_feats_labels.pt")
    p_idtst = os.path.join(args.cache_dir, id_base + "__id_test_feats.pt")

    ood_tag = safe_name(args.ood_dataset)
    extra = ""
    if args.ood_dir is not None and args.ood_dataset.lower() in ["textures", "places", "tiny-imagenet-200"]:
        extra = "__" + safe_name(os.path.basename(args.ood_dir.rstrip("/")))

    ood_base = f"{id_base}__{ood_tag}{extra}"
    p_ood = os.path.join(args.cache_dir, ood_base + "__ood_test_feats.pt")
    return p_train, p_idtst, p_ood


def load_or_extract_id_train(
    feat_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    path: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        return ckpt["feats"].float(), ckpt["labels"].long()
    feats, labels = extract_features(feat_model, loader, device)
    torch.save({"feats": feats.cpu(), "labels": labels.cpu()}, path)
    return feats, labels


def load_or_extract_feats(
    feat_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    path: str,
) -> torch.Tensor:
    if os.path.exists(path):
        return torch.load(path, map_location="cpu").float()
    feats, _ = extract_features(feat_model, loader, device)
    torch.save(feats.cpu(), path)
    return feats


def logits_from_feats(feats: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    X = feats.float()
    return X @ W.t() + b.unsqueeze(0)


# =========================
# Method: pooled-within-class PCA fit
# =========================
def fit_affine_pca_subspaces(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    energy_keep: float = 0.95,
    k_max: int = 150,
    eps: float = 1e-6,
    center_mode: str = "class",      # "class" | "global" | "none"
) -> Dict[int, Dict[str, torch.Tensor]]:
    feats = feats.float()
    labels = labels.long()
    _ = k_max

    if center_mode not in ["class", "global", "none"]:
        raise ValueError("center_mode must be one of: class|global|none")

    N, D = feats.shape
    if N < 5:
        raise RuntimeError(f"Too few samples ({N}).")
    if labels.numel() != N:
        raise RuntimeError(f"labels length mismatch: {labels.numel()} vs {N}")

    # class means (needed for class centering and also for scoring checks)
    mu_c = feats.new_zeros((num_classes, D))
    counts = torch.zeros((num_classes,), device=feats.device, dtype=torch.long)
    for c in range(num_classes):
        idx = (labels == c)
        nc = int(idx.sum().item())
        if nc == 0:
            raise RuntimeError(f"class {c} has zero samples")
        mu_c[c] = feats[idx].mean(dim=0)
        counts[c] = nc

    # global mean
    mu_global = feats.mean(dim=0)  # [D]

    # choose centering for PCA fit
    if center_mode == "class":
        X = feats - mu_c[labels]
    elif center_mode == "global":
        X = feats - mu_global.view(1, -1)
    else:
        X = feats

    # SVD
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh.transpose(0, 1)
    r = V.shape[1]
    lam = (S ** 2) / max(N - 1, 1)

    # energy keep
    energy = torch.cumsum(lam, dim=0) / (lam.sum() + eps)
    k_energy = int(torch.searchsorted(
        energy, torch.tensor(energy_keep, device=energy.device)
    ).item()) + 1
    k = min(k_energy, max(r - 1, 1))

    V_perp = V[:, k:]
    lam_perp = lam[k:]
    if V_perp.numel() == 0:
        V_perp = V[:, -1:].contiguous()
        lam_perp = lam[-1:].contiguous()

    return {
        0: {
            "mu_c": mu_c.contiguous(),
            "mu_global": mu_global.contiguous(),   # NEW
            "V_perp": V_perp.contiguous(),
            "lam_perp": lam_perp.contiguous(),
            "counts": counts.contiguous(),
            "center_mode_fit": center_mode,        # optional, for logging
        }
    }



# =========================
# Scoring (unwhitened residual) and suite loader
# =========================
@torch.no_grad()
def score_residual_subspace(
    feats: torch.Tensor,
    logits: torch.Tensor,
    mu_c: torch.Tensor,
    V_perp: torch.Tensor,
    mode: str = "pred",              # "pred" or "min"
    center_mode: str = "class",      # "class" | "global" | "none" (used only when mode="pred")
    mu_global: Optional[torch.Tensor] = None,
) -> np.ndarray:
    feats = feats.float()
    logits = logits.float()
    mu_c = mu_c.float()
    Vp = V_perp.float()

    N, D = feats.shape
    C = mu_c.shape[0]
    if logits.shape[0] != N:
        raise RuntimeError("N mismatch feats vs logits")
    if logits.shape[1] != C:
        raise RuntimeError("C mismatch logits vs mu_c")

    if mode == "pred":
        if center_mode not in ["class", "global", "none"]:
            raise ValueError("center_mode must be one of: class|global|none")

        pred = torch.argmax(logits, dim=1).long()
        if center_mode == "class":
            z = feats - mu_c[pred]
        elif center_mode == "global":
            if mu_global is None:
                raise ValueError("mu_global required when center_mode='global'")
            z = feats - mu_global.view(1, -1)
        else:
            z = feats

        coeff = z @ Vp
        s = (coeff ** 2).sum(dim=1)
        return s.detach().cpu().numpy().astype(np.float64)

    if mode == "min":
        best = None
        for c in range(C):
            zc = feats - mu_c[c].view(1, -1)
            coeff = zc @ Vp
            sc = (coeff ** 2).sum(dim=1)
            best = sc if best is None else torch.minimum(best, sc)
        return best.detach().cpu().numpy().astype(np.float64)

    raise ValueError("mode must be 'pred' or 'min'")


def load_ood_suite_feats_logits(
    args,
    feat_model: nn.Module,
    device: torch.device,
    tfm,
    W: torch.Tensor,
    b: torch.Tensor,
    ood_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    if ood_names is None:
        ood_names = ["tiny-imagenet-200", "svhn", "textures", "places"]

    out: Dict[str, Dict[str, torch.Tensor]] = {}

    for name in ood_names:
        a = copy.copy(args)
        a.ood_dataset = name

        _, _, p_ood = cache_paths(a)

        split = "test"
        ds = build_dataset(name=name, split=split, data_dir=a.data_dir, tfm=tfm, ood_dir=a.ood_dir)
        loader = DataLoader(
            ds,
            batch_size=a.batch_size,
            shuffle=False,
            num_workers=a.num_workers,
            pin_memory=True,
        )

        feats = load_or_extract_feats(feat_model, loader, device, p_ood)
        logits = logits_from_feats(feats, W, b)
        out[name] = {"feats": feats, "logits": logits}

    return out


# =========================
# Plot helper (dpi=300)
# =========================
# --- CHANGE 2: modify the 3 ablation functions to PRINT AUROC/FPR for each setting ---


# --- CHANGE 1: replace your plot_ablation_two_metrics with this version ---
def plot_ablation_two_metrics(
    x_vals: List,
    mean_auroc: List[float],
    mean_fpr95: List[float],
    xlabel: str,
    outpath: str,
    x_ticklabels: Optional[List[str]] = None,
    label_fontsize: int = 18,
    tick_fontsize: int = 14,
):
    # Single panel, bar representation, grouped by metric: AUROC group then FPR group
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    au_color = "#5fbf5f"  # green
    fp_color = "#ee8f8f"  # red
    edge_color = "#cfcfcf"
    edge_lw = 1.2

    n = len(x_vals)
    x = np.arange(n)

    gap = 1  # one-slot gap between AUROC block and FPR block
    x_au = x
    x_fp = x + n + gap

    bar_w = 1.0  # <-- makes adjacent bars touch

    ax.bar(x_au, mean_auroc, width=bar_w, color=au_color,
           edgecolor=edge_color, linewidth=edge_lw, label="AUROC(%)")
    ax.bar(x_fp, mean_fpr95, width=bar_w, color=fp_color,
           edgecolor=edge_color, linewidth=edge_lw, label="FPR95(%)")

    # Tick labels for each bar
    if x_ticklabels is None:
        x_ticklabels = [
            (f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v))
            for v in x_vals
        ]

    ax.set_xticks(list(x_au) + list(x_fp))
    ax.set_xticklabels(x_ticklabels + x_ticklabels, fontsize=tick_fontsize, rotation=0)

    # Add group labels centered under each group
    au_center = float(np.mean(x_au))
    fp_center = float(np.mean(x_fp))
    y_text = -0.18  # in axes coords; needs bottom margin
    ax.text(au_center, y_text, "AUROC(%)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=label_fontsize)
    ax.text(fp_center, y_text, "FPR95(%)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=label_fontsize)

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    # Legend: top-right, 2 columns, FPR before AUROC
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=fp_color, edgecolor=edge_color, linewidth=edge_lw, label="FPR95(%)"),
        Patch(facecolor=au_color, edgecolor=edge_color, linewidth=edge_lw, label="AUROC(%)"),
    ]
    ax.legend(handles=handles, fontsize=tick_fontsize, frameon=False, loc="upper right",
              ncol=2, columnspacing=1.2, handletextpad=0.6, borderaxespad=0.3)

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)  # more room for the group labels
    fig.savefig(outpath, dpi=300)
    plt.close(fig)



def _hex_to_rgb01(h: str):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0

def _rgb01_to_hex(rgb):
    rgb = np.clip(np.round(np.array(rgb) * 255.0), 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def make_shades_toward_white(base_hex: str, n: int, max_whiten: float = 0.65):
    """
    base_hex is the *brightest allowed* (most saturated) color.
    Generates n shades by blending base -> white, increasing whiteness gradually.
    """
    if n <= 1:
        return [base_hex]
    base = _hex_to_rgb01(base_hex)
    white = np.ones(3, dtype=np.float64)
    ts = np.linspace(0.0, max_whiten, n)  # 0: base, max_whiten: lightest
    return [_rgb01_to_hex((1 - t) * base + t * white) for t in ts]

import numpy as np

def make_value_shades_toward_white(base_hex: str, vals, *, higher_is_better: bool, max_whiten: float = 0.65):
    """
    Returns a list of hex colors where bars are more saturated (less white)
    when the metric is better at that point.

    - higher_is_better=True  : higher vals -> closer to base color
    - higher_is_better=False : lower  vals -> closer to base color
    """
    vals = np.asarray(vals, dtype=np.float64)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    denom = max(vmax - vmin, 1e-12)
    t = (vals - vmin) / denom  # 0..1

    if higher_is_better:
        s = t                 # better (higher) -> 1
    else:
        s = 1.0 - t           # better (lower)  -> 1

    # whiten amount: 0 means pure base, max_whiten means most washed-out
    w = max_whiten * (1.0 - s)

    base_hex = base_hex.lstrip("#")
    br = int(base_hex[0:2], 16)
    bg = int(base_hex[2:4], 16)
    bb = int(base_hex[4:6], 16)

    colors = []
    for wi in w:
        r = int(round((1.0 - wi) * br + wi * 255.0))
        g = int(round((1.0 - wi) * bg + wi * 255.0))
        b = int(round((1.0 - wi) * bb + wi * 255.0))
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors



def ablate_energy_keep_avg_over_oods(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    id_feats: torch.Tensor,
    id_logits: torch.Tensor,
    ood_suite: Dict[str, Dict[str, torch.Tensor]],
    num_classes: int,
    energy_grid: List[float],
    save_path: str,
    eps: float = 1e-6,
) -> None:
    mean_auroc, mean_fpr95 = [], []

    print("\n[ABLATION] energy_keep sweep (avg over OOD suite)")
    for ekeep in energy_grid:
        subspaces = fit_affine_pca_subspaces(
            feats=train_feats,
            labels=train_labels,
            num_classes=num_classes,
            energy_keep=ekeep,
            k_max=0,
            eps=eps,
        )
        m = subspaces[0]
        mu_c = m["mu_c"]
        Vp = m["V_perp"]

        id_scores = score_residual_subspace(
            feats=id_feats, logits=id_logits, mu_c=mu_c, V_perp=Vp,
            mode="pred", center_mode="class",
        )

        ood_scores_dict = {}
        for oname, pack in ood_suite.items():
            ood_scores_dict[oname] = score_residual_subspace(
                feats=pack["feats"], logits=pack["logits"], mu_c=mu_c, V_perp=Vp,
                mode="pred", center_mode="class",
            )

        agg = aggregate_over_ood_suite(id_scores, ood_scores_dict)
        mean_auroc.append(float(agg["mean_auroc"]))
        mean_fpr95.append(float(agg["mean_fpr95"]))

        print(f"  energy_keep={ekeep:.3f} | mean AUROC={agg['mean_auroc']:.4f} | mean FPR95={agg['mean_fpr95']:.4f}")

    # -------------------------
    # Plot: single panel, 2 metric blocks (AUROC then FPR), bars touch inside blocks
    # -------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    label_fontsize = 18
    tick_fontsize = 14

    au_color = "#5fbf5f"  # green
    fp_color = "#ee8f8f"  # red

    edge_color = "#cfcfcf"
    edge_lw = 1.2

    n = len(energy_grid)
    x = np.arange(n)

    gap = 1  # one-slot gap between AUROC block and FPR block
    x_au = x
    x_fp = x + n + gap

    bar_w = 1.0  # bars touch within each block

    # ax.bar(x_au, mean_auroc, width=bar_w, color=au_color,
    #        edgecolor=edge_color, linewidth=edge_lw, label="AUROC(%)")
    # ax.bar(x_fp, mean_fpr95, width=bar_w, color=fp_color,
    #        edgecolor=edge_color, linewidth=edge_lw, label="FPR95(%)")
    
    # inside your ablate_energy_keep_avg_over_oods plotting section:
    au_colors = make_value_shades_toward_white("#5fbf5f", mean_auroc, higher_is_better=True,  max_whiten=0.65)
    fp_colors = make_value_shades_toward_white("#ee8f8f", mean_fpr95, higher_is_better=True, max_whiten=0.65)


    ax.bar(x_au, mean_auroc, width=bar_w, color=au_colors, edgecolor=edge_color, linewidth=edge_lw, label="AUROC(%)")
    ax.bar(x_fp, mean_fpr95, width=bar_w, color=fp_colors, edgecolor=edge_color, linewidth=edge_lw, label="FPR95(%)")

    # x tick labels (repeat energy_grid for each block)
    ticklabs = [f"{v:.1f}" for v in energy_grid]
    ax.set_xticks(list(x_au) + list(x_fp))
    ax.set_xticklabels(ticklabs + ticklabs, fontsize=tick_fontsize)

    # group labels
    au_center = float(np.mean(x_au))
    fp_center = float(np.mean(x_fp))
    y_text = -0.18
    ax.text(au_center, y_text, "AUROC(%)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=label_fontsize)
    ax.text(fp_center, y_text, "FPR95(%)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=label_fontsize)

    # ax.set_xlabel("energy_keep", fontsize=label_fontsize)
    ax.set_xlabel(r"$\rho$", fontsize=label_fontsize)
    ax.xaxis.set_label_coords(0.5, y_text - 0.14)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.yaxis.set_ticks_position("both")
    ax.tick_params(axis="y", which="both", right=True, labelright=True)


    # Legend: top-right, 2 columns, FPR before AUROC
    from matplotlib.patches import Patch
    handles = [ 
        Patch(facecolor=au_color, edgecolor=edge_color, linewidth=edge_lw, label="AUROC(%)"),
        Patch(facecolor=fp_color, edgecolor=edge_color, linewidth=edge_lw, label="FPR95(%)"),
       
    ]
    ax.legend(handles=handles, fontsize=tick_fontsize, frameon=False, loc="upper right",
              ncol=2, columnspacing=1.2, handletextpad=0.6, borderaxespad=0.3)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def ablate_centering_avg_over_oods(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    id_feats: torch.Tensor,
    id_logits: torch.Tensor,
    ood_suite: Dict[str, Dict[str, torch.Tensor]],
    num_classes: int,
    energy_keep: float,
    save_path: str,
    eps: float = 1e-6,
) -> None:
    variants = ["class", "global", "none"]
    mean_auroc, mean_fpr95 = [], []

    print(f"\n[ABLATION] centering modes (true fit+test) (energy_keep={energy_keep}) (avg over OOD suite)")
    for v in variants:
        subspaces = fit_affine_pca_subspaces(
            feats=train_feats,
            labels=train_labels,
            num_classes=num_classes,
            energy_keep=energy_keep,
            k_max=0,
            eps=eps,
            center_mode=v,   # NEW: fit changes with v
        )
        m = subspaces[0]
        mu_c = m["mu_c"]
        Vp = m["V_perp"]
        mu_global = m["mu_global"]

        # score with the SAME centering mode
        id_scores = score_residual_subspace(
            feats=id_feats, logits=id_logits, mu_c=mu_c, V_perp=Vp,
            mode="pred", center_mode=v, mu_global=mu_global,
        )

        ood_scores_dict = {}
        for oname, pack in ood_suite.items():
            ood_scores_dict[oname] = score_residual_subspace(
                feats=pack["feats"], logits=pack["logits"], mu_c=mu_c, V_perp=Vp,
                mode="pred", center_mode=v, mu_global=mu_global,
            )

        agg = aggregate_over_ood_suite(id_scores, ood_scores_dict)
        mean_auroc.append(float(agg["mean_auroc"]))
        mean_fpr95.append(float(agg["mean_fpr95"]))

        print(f"  centering={v:<6} | mean AUROC={agg['mean_auroc']:.4f} | mean FPR95={agg['mean_fpr95']:.4f}")
    # -------------------------
    # Plot: single panel, 2 metric blocks (AUROC then FPR), bars touch inside blocks
    # 3 shades per metric based on relative values (darker = higher)
    # -------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    edge_color = "#cfcfcf"
    edge_lw = 1.2
    label_fontsize = 18
    tick_fontsize = 14

    # 3-level palettes (dark/med/light)
    greens = [ "#5fbf5f", "#9bd39b", "#d9f2d9"]  # dark -> light
    reds   = [ "#ee8f8f", "#f6b1b1",  "#fde2e2"]  # dark -> light

    def _three_shades_by_rank(vals, palette):
        """
        Assign 3 shades based on relative rank:
          highest -> palette[0] (dark), middle -> palette[1], lowest -> palette[2] (light).
        If ties, stable ordering is fine; tied values may get different shades (acceptable for visualization).
        """
        order = np.argsort(vals)  # ascending: low..high
        colors = [None] * len(vals)
        if len(vals) != 3:
            # fallback: quantize by percentiles if you ever change variants length
            q1, q2 = np.quantile(vals, [1/3, 2/3])
            for i, v in enumerate(vals):
                if v >= q2:
                    colors[i] = palette[0]
                elif v >= q1:
                    colors[i] = palette[1]
                else:
                    colors[i] = palette[2]
            return colors

        colors[order[0]] = palette[2]  # lowest -> light
        colors[order[1]] = palette[1]  # middle -> medium
        colors[order[2]] = palette[0]  # highest -> dark
        return colors

    au_cols = _three_shades_by_rank(np.array(mean_auroc, dtype=np.float64), greens)
    fp_cols = _three_shades_by_rank(np.array(mean_fpr95, dtype=np.float64), reds)

    n = len(variants)
    x = np.arange(n)
    gap = 1  # one-slot gap between AUROC block and FPR block
    x_au = x
    x_fp = x + n + gap
    bar_w = 1.0  # bars touch within each block

    # AUROC block
    ax.bar(x_au, mean_auroc, width=bar_w, color=au_cols,
           edgecolor=edge_color, linewidth=edge_lw)

    # FPR block
    ax.bar(x_fp, mean_fpr95, width=bar_w, color=fp_cols,
           edgecolor=edge_color, linewidth=edge_lw)

    # Tick labels (repeat variants for each block)
    ax.set_xticks(list(x_au) + list(x_fp))
    ax.set_xticklabels(variants + variants, fontsize=tick_fontsize)

    # Group labels centered under each block
    au_center = float(np.mean(x_au))
    fp_center = float(np.mean(x_fp))
    y_text = -0.18
    ax.text(au_center, y_text, "AUROC(%)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=label_fontsize)
    ax.text(fp_center, y_text, "FPR95(%)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=label_fontsize)

    ax.set_xlabel("Centering Mode", fontsize=label_fontsize)
    ax.xaxis.set_label_coords(0.5, y_text - 0.14)

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.yaxis.set_ticks_position("both")
    ax.tick_params(axis="y", which="both", right=True, labelright=True)

    # Legend: top-right, 2 columns, FPR in front of AUROC
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=greens[0], edgecolor=edge_color, linewidth=edge_lw, label="AUROC(%)"),
        Patch(facecolor=reds[0],   edgecolor=edge_color, linewidth=edge_lw, label="FPR95(%)"),
        
    ]
    ax.legend(handles=handles, fontsize=tick_fontsize, frameon=False, loc="upper right",
              ncol=2, columnspacing=1.2, handletextpad=0.6, borderaxespad=0.3)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def ablate_pred_vs_min_avg_over_oods(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    id_feats: torch.Tensor,
    id_logits: torch.Tensor,
    ood_suite: Dict[str, Dict[str, torch.Tensor]],
    num_classes: int,
    energy_keep: float,
    save_path: str,
    eps: float = 1e-6,
) -> None:
    subspaces = fit_affine_pca_subspaces(
        feats=train_feats,
        labels=train_labels,
        num_classes=num_classes,
        energy_keep=energy_keep,
        k_max=0,
        eps=eps,
    )
    m = subspaces[0]
    mu_c = m["mu_c"]
    Vp = m["V_perp"]

    modes = ["pred", "min"]
    mean_auroc, mean_fpr95 = [], []

    print(f"\n[ABLATION] scoring rule (energy_keep={energy_keep}) (avg over OOD suite)")
    for mode in modes:
        id_scores = score_residual_subspace(
            feats=id_feats, logits=id_logits, mu_c=mu_c, V_perp=Vp,
            mode=mode, center_mode="class",
        )

        ood_scores_dict = {}
        for oname, pack in ood_suite.items():
            ood_scores_dict[oname] = score_residual_subspace(
                feats=pack["feats"], logits=pack["logits"], mu_c=mu_c, V_perp=Vp,
                mode=mode, center_mode="class",
            )

        agg = aggregate_over_ood_suite(id_scores, ood_scores_dict)
        mean_auroc.append(agg["mean_auroc"])
        mean_fpr95.append(agg["mean_fpr95"])

        print(f"  rule={mode:<4} | mean AUROC={agg['mean_auroc']:.4f} | mean FPR95={agg['mean_fpr95']:.4f}")

    x = list(range(len(modes)))
    plot_ablation_two_metrics(
        x_vals=x,
        mean_auroc=mean_auroc,
        mean_fpr95=mean_fpr95,
        xlabel="scoring rule",
        outpath=save_path,
        x_ticklabels=modes,
    )


# =========================
# NEW ablation: use KEPT PCA directions vs DISCARDED (perp) directions
# Drop-in: add these two functions to your existing script.
# =========================

@torch.no_grad()
def score_subspace_kept_or_perp(
    feats: torch.Tensor,
    logits: torch.Tensor,
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    use_kept: bool = False,   # False=your current method (perp), True=use kept basis
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Same scoring pipeline as your current method, but switches which PCA basis is used:

    Current method (discarded directions):
      score = || V_perp^T (x - mu_{pred}) ||^2

    Ablation (kept directions):
      score = || V_keep^T (x - mu_{pred}) ||^2

    Notes:
    - Uses predicted class mean centering exactly like score_subspaces_ratio_min.
    - Returns 1D numpy scores.
    """
    feats = feats.float()
    logits = logits.float()

    m = subspaces[0] if 0 in subspaces else subspaces[next(iter(subspaces.keys()))]
    mu_c = m["mu_c"]  # [C,D]

    # both bases are expected to be present
    V_keep = m.get("V_keep", None)
    V_perp = m.get("V_perp", None)
    if V_keep is None or V_perp is None:
        raise KeyError("subspaces must contain both 'V_keep' and 'V_perp' (see fit_ablation_pca_bases).")

    pred = torch.argmax(logits, dim=1).long()  # [N]
    z = feats - mu_c[pred]                     # [N,D]

    V = V_keep if use_kept else V_perp         # [D,k] or [D,p]
    coeff = z @ V                               # [N,dim]
    s = (coeff ** 2).sum(dim=1)                 # [N]
    return s.detach().cpu().numpy().astype(np.float64)


def fit_ablation_pca_bases(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    energy_keep: float = 0.95,
    eps: float = 1e-6,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Same as your fit_affine_pca_subspaces, but also stores V_keep explicitly.
    - V_keep: first k PCA directions
    - V_perp: complement (discarded directions) = V[:,k:]
    """
    feats = feats.float()
    labels = labels.long()

    N, D = feats.shape
    if labels.numel() != N:
        raise RuntimeError("labels length mismatch")

    # class means
    mu_c = feats.new_zeros((num_classes, D))
    counts = torch.zeros((num_classes,), device=feats.device, dtype=torch.long)
    for c in range(num_classes):
        idx = (labels == c)
        nc = int(idx.sum().item())
        if nc == 0:
            raise RuntimeError(f"class {c} has zero samples")
        mu_c[c] = feats[idx].mean(dim=0)
        counts[c] = nc

    # pooled within-class residuals
    X = feats - mu_c[labels]

    # SVD
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh.transpose(0, 1)  # [D,r]
    r = V.shape[1]
    lam = (S ** 2) / max(N - 1, 1)

    energy = torch.cumsum(lam, dim=0) / (lam.sum() + eps)
    k_energy = int(torch.searchsorted(energy, torch.tensor(energy_keep, device=energy.device)).item()) + 1
    k = min(k_energy, max(r - 1, 1))  # ensure at least 1 perp direction

    V_keep = V[:, :k].contiguous()
    V_perp = V[:, k:].contiguous()
    if V_perp.numel() == 0:
        V_perp = V[:, -1:].contiguous()

    print(f"[pca bases] k={k} (energy_keep={energy_keep}, rank={r}, kept_dim={k}, perp_dim={V_perp.shape[1]})")

    return {
        0: {
            "mu_c": mu_c.contiguous(),
            "V_keep": V_keep,
            "V_perp": V_perp,
            "counts": counts.contiguous(),
        }
    }
# =========================
# NEW: runner for kept-vs-perp ablation + plot (same style as your original)
# Drop-in: add this function too.
# =========================

# =========================
# Drop-in: kept (V_keep) vs discarded (V_perp) ablation OVER an ood_suite
# Uses the same ood_suite dict you already have:
#   ood_suite[name]["feats"], ood_suite[name]["logits"]
# Produces ONE figure (dpi=300) with two panels (AUROC, FPR95) averaged over OODs.
# =========================

def ablate_kept_vs_perp_over_oods(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    id_feats: torch.Tensor,
    id_logits: torch.Tensor,
    ood_suite: Dict[str, Dict[str, torch.Tensor]],
    num_classes: int,
    energy_keep: float,
    eps: float = 1e-6,
    fig_path: str = "./ablation_kept_vs_perp_over_oods.png",
) -> None:
    """
    Fits pooled within-class PCA once, then compares:
      - DISCARDED directions (V_perp): your current method
      - KEPT directions (V_keep): ablation

    For each OOD dataset in ood_suite:
      compute AUROC/FPR95 for perp and kept
    Then report and plot the mean over OOD datasets.
    """

    # fit once
    subspaces = fit_ablation_pca_bases(
        feats=train_feats,
        labels=train_labels,
        num_classes=num_classes,
        energy_keep=energy_keep,
        eps=eps,
    )

    # ID scores (same for all OOD datasets)
    id_perp = score_subspace_kept_or_perp(id_feats, id_logits, subspaces, use_kept=False, eps=eps)
    id_keep = score_subspace_kept_or_perp(id_feats, id_logits, subspaces, use_kept=True,  eps=eps)

    # per-OOD metrics
    ood_names = list(ood_suite.keys())
    au_perp_list, fp_perp_list = [], []
    au_keep_list, fp_keep_list = [], []

    print("\n[ABLATION] discarded (V_perp) vs kept (V_keep) over OOD suite")
    for name in ood_names:
        ood_feats = ood_suite[name]["feats"]
        ood_logits = ood_suite[name]["logits"]

        ood_perp = score_subspace_kept_or_perp(ood_feats, ood_logits, subspaces, use_kept=False, eps=eps)
        ood_keep = score_subspace_kept_or_perp(ood_feats, ood_logits, subspaces, use_kept=True,  eps=eps)

        au_perp, fp_perp = compute_auroc_fpr95(id_perp, ood_perp)
        au_keep, fp_keep = compute_auroc_fpr95(id_keep, ood_keep)

        au_perp_list.append(au_perp); fp_perp_list.append(fp_perp)
        au_keep_list.append(au_keep); fp_keep_list.append(fp_keep)

        print(f"  {name:<14} | PERP: AUROC={au_perp:.4f} FPR95={fp_perp:.4f}  ||  "
              f"KEPT: AUROC={au_keep:.4f} FPR95={fp_keep:.4f}")

    # mean over OOD datasets
    au_perp_m = float(np.mean(au_perp_list))
    fp_perp_m = float(np.mean(fp_perp_list))
    au_keep_m = float(np.mean(au_keep_list))
    fp_keep_m = float(np.mean(fp_keep_list))

    print("\n  ---- mean over OOD suite ----")
    print(f"  PERP (discarded): AUROC={au_perp_m:.4f}  FPR95={fp_perp_m:.4f}")
    print(f"  KEPT           : AUROC={au_keep_m:.4f}  FPR95={fp_keep_m:.4f}")

    # plot (same simple style as your current ablation plots)
    # plot: group by metric (AUROC together, FPR95 together) with higher value -> darker shade
    labels = ["discarded", "kept"]
    au_vals = [au_perp_m, au_keep_m]
    fp_vals = [fp_perp_m, fp_keep_m]

    label_fs = 18
    tick_fs  = 14

    edge_color = "#cfcfcf"
    edge_lw = 1.2

    # Base shades (dark/light) for each metric
    dark_green  = "#5fbf5f"
    light_green = "#9bd39b"
    dark_red    = "#ee8f8f"
    light_red   = "#f6b1b1"

    # Enforce: higher value -> darker shade (separately for AUROC and FPR95)
    if au_vals[0] >= au_vals[1]:
        au_discarded_col, au_kept_col = dark_green, light_green
    else:
        au_discarded_col, au_kept_col = light_green, dark_green

    if fp_vals[0] >= fp_vals[1]:
        fp_discarded_col, fp_kept_col = dark_red, light_red
    else:
        fp_discarded_col, fp_kept_col = light_red, dark_red

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    groups = ["AUROC(%)", "FPR95(%)"]
    xg = np.arange(len(groups))
    bar_w = 0.32

    # AUROC group: left=discarded, right=kept
    ax.bar(xg[0] - bar_w/2, au_vals[0], width=bar_w, color=au_discarded_col,
           edgecolor=edge_color, linewidth=edge_lw)
    ax.bar(xg[0] + bar_w/2, au_vals[1], width=bar_w, color=au_kept_col,
           edgecolor=edge_color, linewidth=edge_lw)

    # FPR95 group: left=discarded, right=kept
    ax.bar(xg[1] - bar_w/2, fp_vals[0], width=bar_w, color=fp_discarded_col,
           edgecolor=edge_color, linewidth=edge_lw)
    ax.bar(xg[1] + bar_w/2, fp_vals[1], width=bar_w, color=fp_kept_col,
           edgecolor=edge_color, linewidth=edge_lw)

    ax.set_xticks(xg)
    ax.set_xticklabels(groups, fontsize=tick_fs)
    ax.set_xlabel("Space Considered", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.yaxis.set_ticks_position("both")
    ax.tick_params(axis="y", which="both", right=True, labelright=True)

    # Legend that explicitly describes BOTH metrics + discarded/kept
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=au_discarded_col, edgecolor=edge_color, linewidth=edge_lw, label="AUROC discarded"),
        Patch(facecolor=au_kept_col,      edgecolor=edge_color, linewidth=edge_lw, label="AUROC retained"),
        Patch(facecolor=fp_discarded_col, edgecolor=edge_color, linewidth=edge_lw, label="FPR95 discarded"),
        Patch(facecolor=fp_kept_col,      edgecolor=edge_color, linewidth=edge_lw, label="FPR95 retained"),
    ]
    # ax.legend(handles=handles, fontsize=tick_fs, frameon=False, loc="best")
    ax.legend(
        handles=handles,
        fontsize=tick_fs,
        frameon=False,
        loc="upper right",
        ncol=2,                 # <-- puts AUROC/FPR side-by-side
        columnspacing=1.2,
        handletextpad=0.6,
        borderaxespad=0.3,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



# =========================
# Drop-in: Avg (over OOD suite) Pred vs Min scoring rule ablation
# Mirrors the SAME structure as your kept-vs-perp averaging code:
#   - loops over ood_suite
#   - computes metrics per dataset
#   - averages AUROC/FPR over datasets
#   - plots two bars (pred vs min) with your simple 2-panel bar figure
#
# Requires you already have:
#   - fit_affine_pca_subspaces(...)
#   - score_residual_subspace(...)
#   - compute_auroc_fpr95(...)
# =========================

def ablate_avg_pred_vs_min_over_oods(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    id_feats: torch.Tensor,
    id_logits: torch.Tensor,
    ood_suite: Dict[str, Dict[str, torch.Tensor]],
    num_classes: int,
    energy_keep: float,
    eps: float = 1e-6,
    fig_path: str = "./ablation_avg_pred_vs_min.png",
) -> None:
    """
    Compares scoring rules:
      - pred: score using predicted class only
      - min : score = min over all classes

    Prints per-dataset AUROC/FPR95 for both, then prints mean over OOD suite,
    and saves one 2-panel bar figure (AUROC, FPR95) with two bars (pred/min).
    """

    # Fit once (same PCA fit you use everywhere)
    subspaces = fit_affine_pca_subspaces(
        feats=train_feats,
        labels=train_labels,
        num_classes=num_classes,
        energy_keep=energy_keep,
        k_max=0,
        eps=eps,
    )
    m = subspaces[0]
    mu_c = m["mu_c"]
    Vp = m["V_perp"]

    # ID scores for both rules (reused for every OOD dataset)
    id_pred = score_residual_subspace(
        feats=id_feats, logits=id_logits,
        mu_c=mu_c, V_perp=Vp,
        mode="pred", center_mode="class",
    )
    id_min = score_residual_subspace(
        feats=id_feats, logits=id_logits,
        mu_c=mu_c, V_perp=Vp,
        mode="min", center_mode="class",
    )

    au_pred_list, fp_pred_list = [], []
    au_min_list,  fp_min_list  = [], []

    print(f"\n[ABLATION] avg over OOD suite: pred vs min (energy_keep={energy_keep})")
    for name in ood_suite.keys():
        ood_feats  = ood_suite[name]["feats"]
        ood_logits = ood_suite[name]["logits"]

        # OOD scores for both rules
        ood_pred = score_residual_subspace(
            feats=ood_feats, logits=ood_logits,
            mu_c=mu_c, V_perp=Vp,
            mode="pred", center_mode="class",
        )
        ood_min = score_residual_subspace(
            feats=ood_feats, logits=ood_logits,
            mu_c=mu_c, V_perp=Vp,
            mode="min", center_mode="class",
        )

        # Metrics per dataset
        au_pred, fp_pred = compute_auroc_fpr95(id_pred, ood_pred)
        au_min,  fp_min  = compute_auroc_fpr95(id_min,  ood_min)

        au_pred_list.append(au_pred); fp_pred_list.append(fp_pred)
        au_min_list.append(au_min);   fp_min_list.append(fp_min)

        print(f"  {name:<14} | PRED: AUROC={au_pred:.4f} FPR95={fp_pred:.4f}  ||  "
              f"MIN: AUROC={au_min:.4f} FPR95={fp_min:.4f}")

    # mean over OOD datasets
    au_pred_m = float(np.mean(au_pred_list))
    fp_pred_m = float(np.mean(fp_pred_list))
    au_min_m  = float(np.mean(au_min_list))
    fp_min_m  = float(np.mean(fp_min_list))

    print("\n  ---- mean over OOD suite ----")
    print(f"  PRED: AUROC={au_pred_m:.4f}  FPR95={fp_pred_m:.4f}")
    print(f"  MIN : AUROC={au_min_m:.4f}  FPR95={fp_min_m:.4f}")

    # plot (thinner bars + light red + light border boxes)
        # plot (green for AUROC, red for FPR, thinner bars)
    # --- single-panel plot (AUROC bars in green, FPR95 bars in red) ---
    # --- single-panel plot: group by metric (AUROC together, FPR95 together) ---
    # --- single-panel plot: group by metric (AUROC together, FPR95 together) ---
    labels = ["pred", "min"]
    au_vals = [au_pred_m, au_min_m]
    fp_vals = [fp_pred_m, fp_min_m]

    label_fs = 18
    tick_fs  = 14

    edge_color = "#cfcfcf"
    edge_lw = 1.2

    # Base shades (dark/light) for each metric
    dark_green  = "#5fbf5f"
    light_green = "#9bd39b"
    dark_red    = "#ee8f8f"
    light_red   = "#f6b1b1"

    # Enforce: higher value -> darker shade (done separately for AUROC and FPR95)
    if au_vals[0] >= au_vals[1]:
        au_pred_col, au_min_col = dark_green, light_green
    else:
        au_pred_col, au_min_col = light_green, dark_green

    if fp_vals[0] >= fp_vals[1]:
        fp_pred_col, fp_min_col = dark_red, light_red
    else:
        fp_pred_col, fp_min_col = light_red, dark_red

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    groups = ["AUROC(%)", "FPR95(%)"]
    xg = np.arange(len(groups))
    bar_w = 0.32

    # AUROC group: left=pred, right=min
    ax.bar(xg[0] - bar_w/2, au_vals[0], width=bar_w, color=au_pred_col,
           edgecolor=edge_color, linewidth=edge_lw)
    ax.bar(xg[0] + bar_w/2, au_vals[1], width=bar_w, color=au_min_col,
           edgecolor=edge_color, linewidth=edge_lw)

    # FPR95 group: left=pred, right=min
    ax.bar(xg[1] - bar_w/2, fp_vals[0], width=bar_w, color=fp_pred_col,
           edgecolor=edge_color, linewidth=edge_lw)
    ax.bar(xg[1] + bar_w/2, fp_vals[1], width=bar_w, color=fp_min_col,
           edgecolor=edge_color, linewidth=edge_lw)

    ax.set_xticks(xg)
    ax.set_xticklabels(groups, fontsize=tick_fs)
    ax.set_xlabel("Scoring Rule", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.yaxis.set_ticks_position("both")
    ax.tick_params(axis="y", which="both", right=True, labelright=True)

    # Legend that explicitly describes BOTH metrics + pred/min
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=au_pred_col, edgecolor=edge_color, linewidth=edge_lw, label="AUROC pred"),
        Patch(facecolor=au_min_col,  edgecolor=edge_color, linewidth=edge_lw, label="AUROC min"),
        Patch(facecolor=fp_pred_col, edgecolor=edge_color, linewidth=edge_lw, label="FPR95 pred"),
        Patch(facecolor=fp_min_col,  edgecolor=edge_color, linewidth=edge_lw, label="FPR95 min"),
    ]
    # ax.legend(handles=handles, fontsize=tick_fs, frameon=False, loc="best")
    ax.legend(
        handles=handles,
        fontsize=tick_fs,
        frameon=False,
        loc="upper right",
        ncol=2,                 # <-- puts AUROC/FPR side-by-side
        columnspacing=1.2,
        handletextpad=0.6,
        borderaxespad=0.3,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



# "#9bd39b"



# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id_dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--data_dir", type=str, default="/home/sgchr/Documents/Subspaces/data")
    ap.add_argument("--ood_dir", type=str, default="/home/sgchr/Documents/Multiple_spaces/Imagenet/ood_data/")
    ap.add_argument("--ood_dataset", type=str, default="textures",
                        help="svhn, cifar100, mnist, fashionmnist, textures, places, tiny-imagenet-200")
    ap.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet34", "vit_b_16"])
    ap.add_argument("--ckpt", type=str, default="/home/sgchr/Documents/Subspaces/checkpoints/resnet18_imagenet_finetuned_nodropout_cifar10.pth")

    ap.add_argument("--cache_dir", type=str, default="./cache_feats")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_l2", action="store_true")

    ap.add_argument("--energy_keep", type=float, default=0.30)
    ap.add_argument("--eps", type=float, default=1e-6)

    ap.add_argument("--out_dir", type=str, default="./ablation_figs")
    ap.add_argument("--energy_grid", type=float, nargs="+", default=[  0.2, 0.30, 0.40, 0.50, 0.70, 0.80, 0.90])

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms: ImageNet preprocessing matching backbone family
    if args.arch == "resnet18":
        tfm = ResNet18_Weights.DEFAULT.transforms()
    elif args.arch == "resnet34":
        tfm = ResNet34_Weights.DEFAULT.transforms()
    else:
        tfm = ViT_B_16_Weights.DEFAULT.transforms()

    num_classes = 10 if args.id_dataset.lower() == "cifar10" else 100

    clf_model = build_classifier(args.arch, num_classes, args.ckpt, device).eval()
    feat_model = make_penultimate_extractor(clf_model, args.arch, l2_normalize=(not args.no_l2)).to(device).eval()
    W, b = get_classifier_linear(clf_model, args.arch)

    # ID loaders
    id_train = build_dataset(args.id_dataset, "train", args.data_dir, tfm, ood_dir=args.ood_dir)
    id_test  = build_dataset(args.id_dataset, "test",  args.data_dir, tfm, ood_dir=args.ood_dir)

    id_train_loader = DataLoader(
        id_train, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    id_test_loader = DataLoader(
        id_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Cache/load ID train feats+labels and ID test feats
    p_train, p_idtst, _ = cache_paths(args)
    train_feats, train_labels = load_or_extract_id_train(feat_model, id_train_loader, device, p_train)
    id_feats = load_or_extract_feats(feat_model, id_test_loader, device, p_idtst)
    id_logits = logits_from_feats(id_feats, W, b)

    # Load 4 OOD datasets (cached)
    ood_suite = load_ood_suite_feats_logits(
        args=args,
        feat_model=feat_model,
        device=device,
        tfm=tfm,
        W=W, b=b,
        ood_names=["tiny-imagenet-200", "svhn", "textures", "places","cifar100"],
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Study 1: energy_keep sweep
    ablate_energy_keep_avg_over_oods(
        train_feats=train_feats,
        train_labels=train_labels,
        id_feats=id_feats,
        id_logits=id_logits,
        ood_suite=ood_suite,
        num_classes=num_classes,
        energy_grid=list(args.energy_grid),
        save_path=os.path.join(args.out_dir, "ablation_energy_keep.png"),
        eps=args.eps,
    )

    # Study 2: centering ablation
    ablate_centering_avg_over_oods(
        train_feats=train_feats,
        train_labels=train_labels,
        id_feats=id_feats,
        id_logits=id_logits,
        ood_suite=ood_suite,
        num_classes=num_classes,
        energy_keep=args.energy_keep,
        save_path=os.path.join(args.out_dir, "ablation_centering.png"),
        eps=args.eps,
    )

    # Study 3: predicted vs min-over-classes
    ablate_pred_vs_min_avg_over_oods(
        train_feats=train_feats,
        train_labels=train_labels,
        id_feats=id_feats,
        id_logits=id_logits,
        ood_suite=ood_suite,
        num_classes=num_classes,
        energy_keep=args.energy_keep,
        save_path=os.path.join(args.out_dir, "ablation_pred_vs_min.png"),
        eps=args.eps,
    )

    ablate_kept_vs_perp_over_oods(
    train_feats=train_feats,
    train_labels=train_labels,
    id_feats=id_feats,
    id_logits=id_logits,
    ood_suite=ood_suite,
    num_classes=num_classes,
    energy_keep=args.energy_keep,
    eps=args.eps,
    fig_path=os.path.join(args.out_dir, "ablation_kept_vs_perp.png"),
    )

    ablate_avg_pred_vs_min_over_oods(
    train_feats=train_feats,
    train_labels=train_labels,
    id_feats=id_feats,
    id_logits=id_logits,
    ood_suite=ood_suite,
    num_classes=num_classes,
    energy_keep=args.energy_keep,
    eps=args.eps,
    fig_path=os.path.join(args.out_dir, "ablation_avg_pred_vs_min.png"),
)



    print("Saved figures to:", args.out_dir)


if __name__ == "__main__":
    main()
