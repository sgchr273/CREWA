#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multi_method_ood_cached.py

Methods included (knn/odin removed):
  - subspaces
  - mahalanobis
  - energy
  - msp
  - logit_gate
  - kpca_rff
  - gradsubspace  (ID subspace from penultimate feats; score uses KL-to-uniform mixed target)

Caching:
  Saves and reuses penultimate features under --cache_dir.
  ID caches are independent of the OOD dataset.
  OOD cache stores feats only (no labels).
"""

import os
import copy
import random
import argparse
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import (
    resnet18, resnet34, ResNet34_Weights,
    ResNet18_Weights,
    vit_b_16, ViT_B_16_Weights,
)
from sklearn.metrics import roc_auc_score, roc_curve

from new_methods import (
    logits_from_feats,
    scores_energy_from_logits,
    scores_msp_from_logits,
    fit_global_pca_basis,   
    # fit_affine_pca_subspaces,
    fit_id_feature_subspace,
    fit_kpca_rff,
    scores_logit_gate,
    scores_kpca_rff,
    scores_gradsubspace_pseudo_resid,
    run_method_subspaces,
    run_method_subspaces_simple,
    run_method_mahalanobis,
    run_method_neco,
    run_method_vim,
    compute_mu_global,
    extract_gaussian_feats_like_id,
    nci_scores_batched,
    run_method_mahalanobis_plus_align)

from new_methods import (
    # get_all_labels,
    # compute_class_means,
    # get_classifier_weight_matrix,
    # infer_pseudo_labels_by_cosine,
    # nc_centered_alignment_scores,
    compute_nc_subtracted_scores
)

# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def orient_scores_higher_is_ood(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Ensure higher scores => more OOD.
    If OOD mean < ID mean, flip sign for both.
    Returns possibly flipped scores and a flag.
    """
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


def report_metrics(method_name: str, id_scores: np.ndarray, ood_scores: np.ndarray):
    """
    Backward compatible:
      - Prints AUROC + FPR95 + score stats for all methods.
      - For NCI (and aliases), returns AUROC only (for varpi tuning).
      - For all other methods, returns (auroc, fpr95) so you can average across seeds.
        (If you never use the return value, behavior is unchanged.)
    """
    scores = np.concatenate([id_scores, ood_scores], axis=0)
    y_true = np.concatenate(
        [np.zeros_like(id_scores, dtype=np.int64),
         np.ones_like(ood_scores, dtype=np.int64)],
        axis=0,
    )

    auroc = roc_auc_score(y_true, scores)
    fpr95 = fpr_at_tpr(y_true, scores, tpr_level=0.95)

    print(f"\n========== {method_name.upper()} ==========")
    print(f"AUROC (OOD positive):      {auroc:.4f}")
    print(f"FPR@95%TPR (OOD positive): {fpr95:.4f}")
    print(f"(score means) ID  mean={id_scores.mean():.6f} std={id_scores.std():.6f}")
    print(f"(score means) OOD mean={ood_scores.mean():.6f} std={ood_scores.std():.6f}")
    print("======================================")

    name = method_name.strip().lower()
    if name in {"nci", "nci-temp", "nci_score"}:
        return float(auroc)                 # keep your tuning behavior
    return float(auroc), float(fpr95)       # for seed averaging




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
        return datasets.SVHN(root=data_dir, split=split, download=True, transform=tfm)

    if name in ["mnist", "fashionmnist"]:
        train = (split == "train")
        base = tfm
        tfm_3ch = transforms.Compose([transforms.Grayscale(num_output_channels=3), base])
        if name == "mnist":
            return datasets.MNIST(root=data_dir, train=train, download=True, transform=tfm_3ch)
        else:
            return datasets.FashionMNIST(root=data_dir, train=train, download=True, transform=tfm_3ch)

    if name in ["textures","places"]:
        if ood_dir is None:
            raise ValueError("For textures, pass ood_dir pointing to the Textures folder.")
        return datasets.ImageFolder(os.path.join(ood_dir, name), transform=tfm)
    if name == "tiny-imagenet-200":
        if ood_dir is None:
            raise ValueError("For tiny-imagenet-200, pass ood_dir pointing to the Tiny-ImageNet-200 folder.")
    
        return datasets.ImageFolder(os.path.join(ood_dir, name, 'val'), transform=tfm)
    # if name == "places":
    #     if ood_dir is None:
    #         raise ValueError("For places, pass ood_dir pointing to the Places folder.")
    #     return datasets.ImageFolder(ood_dir, transform=tfm)

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
        raise RuntimeError("Classifier head is not nn.Linear; logits-based methods assume a linear head.")

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

    # IMPORTANT: include no_l2 in cache key (see next section)
    l2_tag = "l2" if (not args.no_l2) else "nol2"

    id_base = f"{id_tag}__{arch_tag}__{l2_tag}__{ckpt_tag}"

    p_train = os.path.join(args.cache_dir, id_base + "__id_train_feats_labels.pt")
    p_idtst = os.path.join(args.cache_dir, id_base + "__id_test_feats.pt")

    ood_tag = safe_name(args.ood_dataset)

    extra = ""
    if args.ood_dir is not None and args.ood_dataset.lower() in ["textures", "places", "tiny-imagenet-200", "imagefolder"]:
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




# -------------------------
# CLI parsing
# -------------------------
def parse_methods(args) -> List[str]:
    methods = []
    if args.methods is not None and len(args.methods) > 0:
        methods = [m.lower() for m in args.methods]
    else:
        if args.subspaces: methods.append("subspaces")
        if args.mahalanobis: methods.append("mahalanobis")
        if args.energy: methods.append("energy")
        if args.msp: methods.append("msp")
        if args.logit_gate: methods.append("logit_gate")
        if args.kpca_rff: methods.append("kpca_rff")
        if args.gradsubspace: methods.append("gradsubspace")
        if args.neco: methods.append("neco")


    # if len(methods) == 0:
    #     raise RuntimeError("No methods selected. Use --methods ... or flags.")

    allowed = {"subspaces", "mahalanobis", "energy", "msp", "logit_gate", "kpca_rff", "gradsubspace", "neco", "vim", "nci", "maha_align"}
    out = []
    for m in methods:
        if m not in allowed:
            raise ValueError(f"Unknown/disabled method: {m}. Allowed: {sorted(list(allowed))}")
        if m not in out:
            out.append(m)
    return out



# ---------- helpers ----------
@torch.no_grad()
def _logits_from_feats(feats: torch.Tensor, W: torch.Tensor, b: torch.Tensor = None) -> torch.Tensor:
    """
    feats: [N, D] on some device
    W:    [C, D] (may be on cpu/gpu)
    b:    [C] or [1,C] (optional, may be on cpu/gpu)

    Returns logits on feats.device
    """
    dev = feats.device
    Wd = W.to(dev)
    logits = feats @ Wd.t()

    if b is not None:
        bd = b.to(dev)
        logits = logits + bd.view(1, -1)

    return logits

@torch.no_grad()
def _l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)

@torch.no_grad()
def score_parts_residual_and_align(
    feats: torch.Tensor,            # [N,D]
    logits: torch.Tensor,           # [N,C]
    subspace_dict: dict,            # output of fit_affine_pca_subspaces(...)[0]
    W: torch.Tensor,                # [C,D]
    eps: float = 1e-12,
):
    dev = feats.device
    feats = feats.float()

    mu_c   = subspace_dict["mu_c"].to(dev).float()      # [C,D]
    V_perp = subspace_dict["V_perp"].to(dev).float()    # [D,p]
    W      = W.to(dev).float()                          # [C,D]
    logits = logits.to(dev)                             # [N,C]

    pred = logits.argmax(dim=1)                         # [N]
    z = feats - mu_c[pred]                              # [N,D]

    # complement residual energy
    proj = z @ V_perp                                   # [N,p]
    s_res = (proj * proj).sum(dim=1)                    # [N]

    # alignment penalty with predicted class weight
    w_pred = W[pred]                                    # [N,D]

    z_hat = z / (z.norm(dim=1, keepdim=True) + eps)
    w_hat = w_pred / (w_pred.norm(dim=1, keepdim=True) + eps)

    cos = (z_hat * w_hat).sum(dim=1).clamp(-1.0, 1.0)   # [N]
    a = 1.0 - cos

    return s_res, a

def calibrate_beta_id_only(
    s_res_id: torch.Tensor,
    a_id: torch.Tensor,
    mode: str = "median_ratio",   # "median_ratio", "std_ratio", or "fixed"
    fixed_beta: float = 1.0,
    eps: float = 1e-12,
) -> float:
    if mode == "fixed":
        return float(fixed_beta)

    s_res_np = s_res_id.detach().float().cpu().numpy()
    a_np     = a_id.detach().float().cpu().numpy()

    if mode == "median_ratio":
        med_res = float(np.median(s_res_np))
        med_a   = float(np.median(a_np))
        if med_a < eps:
            return 0.0
        return med_res / med_a

    if mode == "std_ratio":
        std_res = float(np.std(s_res_np)) + eps
        std_a   = float(np.std(a_np))     + eps
        return std_res / std_a

    raise ValueError(f"Unknown beta calibration mode: {mode}")

@torch.no_grad()
def summarize_scores(x: torch.Tensor, qs=(0.95, 0.99, 0.999)) -> dict:
    x_np = x.detach().float().cpu().numpy()
    out = {
        "mean": float(x_np.mean()),
        "std": float(x_np.std()),
    }
    for q in qs:
        out[f"q{int(q*1000):03d}"] = float(np.quantile(x_np, q))
    return out

from typing import Dict, Optional
import torch


# =========================
def fit_affine_pca_subspaces(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    energy_keep: float = 0.95,
    alpha: float = 0.30,                 # stable rank scaling factor
    k_mode: str = "stable",              # "stable", "energy", "min", "max"
    min_perp_dim: int = 64,              # enforce at least this many discarded dims when possible
    k_max: int = 150,                    # kept for signature compatibility, NOT used
    eps: float = 1e-6,
    verbose: bool = True,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Pooled within class PCA (tied covariance):
      mu_c: class means
      X = feats - mu_c[labels]
      SVD on X to get a shared PCA basis
      choose k (kept dims) via either:
        - energy threshold (energy_keep)
        - stable rank rule: k = floor(alpha * stable_rank)
        - combine both with k_mode in {"min","max"}
      store V_perp and lam_perp for residual scoring

    Returns a dict with:
      mu_c:     [C, D]
      V_keep:   [D, k]
      V_perp:   [D, r-k]
      lam_perp: [r-k]
      counts:   [C]
    """
    feats = feats.float()
    labels = labels.long()
    _ = k_max

    N, D = feats.shape
    if N < 5:
        raise RuntimeError(f"Too few samples ({N}).")
    if labels.numel() != N:
        raise RuntimeError(f"labels length mismatch: {labels.numel()} vs {N}")

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

    # pooled within class residuals
    X = feats - mu_c[labels]  # [N, D]

    # SVD
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh.transpose(0, 1)  # [D, r]
    r = V.shape[1]
    lam = (S ** 2) / max(N - 1, 1)

    # energy based k
    energy = torch.cumsum(lam, dim=0) / (lam.sum() + eps)
    k_energy = int(torch.searchsorted(
        energy, torch.tensor(energy_keep, device=energy.device)
    ).item()) + 1

    # stable rank (effective rank proxy) and stable k
    stable_r = float(((S ** 2).sum() ** 2) / ((S ** 4).sum() + eps))
    k_stable = int(alpha * stable_r)

    # clamp helper
    def clamp_k(k_raw: int) -> int:
        # ensure at least 1 kept and at least 1 discarded
        k_clamped = max(1, min(int(k_raw), r - 1))

        # if you want at least min_perp_dim discarded directions, enforce k <= r - min_perp_dim
        # only enforce when it is feasible
        if min_perp_dim is not None and min_perp_dim > 0:
            if r - 1 >= 1 and r - min_perp_dim >= 1:
                k_clamped = min(k_clamped, r - min_perp_dim)

        # final safety: still ensure at least 1 discarded
        k_clamped = max(1, min(k_clamped, r - 1))
        return k_clamped

    k_energy = clamp_k(k_energy)
    k_stable = clamp_k(k_stable)

    if k_mode == "stable":
        k = k_stable
    elif k_mode == "energy":
        k = k_energy
    elif k_mode == "min":
        k = min(k_energy, k_stable)
    elif k_mode == "max":
        k = max(k_energy, k_stable)
    else:
        raise ValueError(f"Unknown k_mode={k_mode}. Use: stable, energy, min, max.")

    if verbose:
        retained_energy = float(energy[k - 1]) if k - 1 < energy.numel() else float("nan")
        print(
            f"[dim summary] r={r} | stable_rank={stable_r:.1f} | "
            f"k_energy={k_energy} (energy_keep={energy_keep}) | "
            f"k_stable={k_stable} (alpha={alpha}) | "
            f"k_used={k} (k_mode={k_mode}) | "
            f"perp_dim={r - k} | retained_energy={retained_energy:.4f}"
        )

    V_keep = V[:, :k].contiguous()
    V_perp = V[:, k:].contiguous()
    lam_perp = lam[k:].contiguous()

    # ensure at least 1 perp direction (should already hold, but keep as guard)
    if V_perp.numel() == 0:
        V_perp = V[:, -1:].contiguous()
        lam_perp = lam[-1:].contiguous()
        V_keep = V[:, :-1].contiguous() if r > 1 else V[:, :1].contiguous()

    return {
        0: {
            "mu_c": mu_c.contiguous(),          # [C, D]
            "V_keep": V_keep,                   # [D, k]
            "V_perp": V_perp,                   # [D, r-k]
            "lam_perp": lam_perp,               # [r-k]
            "counts": counts.contiguous(),      # [C]
        }
    }

# ---------- main sweep ----------
def sweep_alpha_id_only(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    id_feats: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    num_classes: int,
    alphas=np.arange(0.1, 1.0, 0.1),
    k_mode: str = "stable",
    min_perp_dim: int = 64,
    beta_mode: str = "median_ratio",   # "median_ratio", "std_ratio", "fixed"
    fixed_beta: float = 1.0,
    select_quantile: float = 0.99,     # choose alpha minimizing ID tail at this quantile
    energy_keep: float = 0.95,
    eps: float = 1e-6,
):
    """
    Uses ONLY ID data:
      - Fit subspace on (train_feats, train_labels)
      - Calibrate beta from ID (train set scores)
      - Evaluate tail of s_aug on id_feats (ID test)
      - Pick alpha minimizing quantile(s_aug, select_quantile)

    Returns:
      rows: list of dicts (one per alpha)
      best: dict for the selected alpha
    """
    device = train_feats.device
    W = W.to(device)
    b = None if b is None else b.to(device)

    # logits for train and id-test (ID only)
    train_logits = _logits_from_feats(train_feats, W, b)
    id_logits    = _logits_from_feats(id_feats,    W, b)

    rows = []
    for alpha in alphas:
        # fit pooled within-class PCA subspace (ID train only)
        subspaces = fit_affine_pca_subspaces(
            train_feats, train_labels, num_classes,
            energy_keep=energy_keep,
            alpha=float(alpha),
            k_mode=k_mode,
            min_perp_dim=min_perp_dim,
            eps=eps,
            verbose=False,
        )[0]

        # score parts on ID train for beta calibration
        s_res_tr, a_tr = score_parts_residual_and_align(train_feats, train_logits, subspaces, W)
        beta = calibrate_beta_id_only(s_res_tr, a_tr, mode=beta_mode, fixed_beta=fixed_beta)

        # score on ID test
        s_res_id, a_id = score_parts_residual_and_align(id_feats, id_logits, subspaces, W)
        s_aug_id = s_res_id + beta * a_id

        summ_aug = summarize_scores(s_aug_id, qs=(0.95, 0.99, 0.999))
        q_key = f"q{int(select_quantile*1000):03d}"
        crit = summ_aug[q_key]

        rows.append({
            "alpha": float(alpha),
            "beta": float(beta),
            "criterion": float(crit),
            "aug_mean": summ_aug["mean"],
            "aug_std": summ_aug["std"],
            "aug_q095": summ_aug["q950"],
            "aug_q099": summ_aug["q990"],
            "aug_q0999": summ_aug["q999"],
            "k_used": int(subspaces["V_keep"].shape[1]),
            "perp_dim": int(subspaces["V_perp"].shape[1]),
        })

    # pick best alpha by smallest ID tail quantile
    rows_sorted = sorted(rows, key=lambda d: d["criterion"])
    best = rows_sorted[0]

    print("\n[alpha sweep | ID-only selection]")
    print(f"Selection criterion: minimize quantile_{select_quantile:.3f}(s_aug on ID test)")
    print(f"Best alpha = {best['alpha']:.1f} | beta={best['beta']:.4g} | "
          f"k={best['k_used']} | perp_dim={best['perp_dim']} | "
          f"criterion={best['criterion']:.6g}")

    # optional: print a small table
    for r in rows_sorted:
        print(f"alpha={r['alpha']:.1f} | beta={r['beta']:.3g} | "
              f"k={r['k_used']:4d} | perp={r['perp_dim']:4d} | "
              f"q{int(select_quantile*1000):03d}={r['criterion']:.6g} | "
              f"mean={r['aug_mean']:.6g} std={r['aug_std']:.6g}")

    return rows, best



def select_alpha_by_perp_participation_ratio(
    train_feats, train_labels, num_classes,
    alphas=[0.3, 0.4, 0.5, 0.6, 0.7],
    eps=1e-6
):
    feats = train_feats.float()
    labels = train_labels.long()
    mu_c = feats.new_zeros((num_classes, feats.shape[1]))
    for c in range(num_classes):
        mu_c[c] = feats[labels == c].mean(dim=0)
    X = feats - mu_c[labels]
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh.transpose(0, 1)
    r = V.shape[1]
    lam = (S ** 2) / max(len(feats) - 1, 1)
    stable_r = float(((S**2).sum()**2) / ((S**4).sum() + eps))

    print(f"\n{'Alpha':<8} {'k':<8} {'perp_dim':<12} {'mean_lam_perp':<18} {'mean_lam_keep':<18} {'ratio perp/keep':<15}")
    print("-" * 79)

    results = {}
    for alpha in alphas:
        k = max(1, min(int(alpha * stable_r), r - 64))
        lam_perp = lam[k:]
        lam_keep = lam[:k]

        mean_lam_perp = float(lam_perp.mean())
        mean_lam_keep = float(lam_keep.mean())
        # ratio: how large perp eigenvalues are relative to kept ones
        # we want this as large as possible — penalizes cutting too deep
        ratio = mean_lam_perp / (mean_lam_keep + eps)

        print(f"{alpha:<8} {k:<8} {r-k:<12} {mean_lam_perp:<18.6f} {mean_lam_keep:<18.6f} {ratio:<15.6f}")
        results[alpha] = {
            "k": k,
            "perp_dim": r - k,
            "mean_lam_perp": mean_lam_perp,
            "mean_lam_keep": mean_lam_keep,
            "ratio": ratio,
        }

    # pick alpha where perp eigenvalues are largest relative to kept
    # this penalizes large alpha (which pushes perp into noise dimensions)
    best_alpha = max(results, key=lambda a: results[a]["mean_lam_perp"])
    print(f"\n=> selected alpha={best_alpha} (largest mean perp eigenvalue)")
    return best_alpha, results

parser = argparse.ArgumentParser()

parser.add_argument("--id_dataset", type=str, default="cifar10",
                    help="cifar10, cifar100, svhn, mnist, fashionmnist")
parser.add_argument("--ood_dataset", type=str, default="cifar100",
                    help="svhn, cifar100, mnist, fashionmnist, textures, places, tiny-imagenet-200")
parser.add_argument("--ood_dir", type=str, default="/home/sgchr/Documents/Multiple_spaces/Imagenet/ood_data/",
                    help="For textures/places: pass folder path to ImageFolder root")
parser.add_argument("--data_dir", type=str, default="./data")

parser.add_argument("--arch", type=str, default="vit_b_16",
                    help="resnet18, resnet34, vit_b_16")
parser.add_argument("--ckpt", type=str, default="/home/sgchr/Documents/Subspaces/checkpoints/vit_b16_cifar10_best_finetuning.pt",
                    help="Path to checkpoint state_dict (or dict containing model_state/model/state_dict)")
parser.add_argument("--no_l2", action="store_true")

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)

# caching
parser.add_argument("--cache_dir", type=str, default="./cache_feats",
                    help="Directory to save/load cached features")

# methods
parser.add_argument("--methods", nargs="+", default=None,
                    help="List: subspaces mahalanobis energy msp logit_gate kpca_rff gradsubspace neco")
parser.add_argument("--subspaces", action="store_true")
parser.add_argument("--mahalanobis", action="store_true")
parser.add_argument("--energy", action="store_true")
parser.add_argument("--msp", action="store_true")
parser.add_argument("--logit_gate", action="store_true")
parser.add_argument("--kpca_rff", action="store_true")
parser.add_argument("--gradsubspace", action="store_true")
parser.add_argument("--neco", action="store_true")
parser.add_argument("--vim", action="store_true")
parser.add_argument("--NCI", action="store_true")
parser.add_argument("--maha_align", action="store_true")

# subspaces params
parser.add_argument("--energy_keep", type=float, default=0.20)  ##0.30 for resnet18, 0.20 for vitb16,  0.90 for cifar100,resnet34 
parser.add_argument("--k_max", type=int, default=64)

# energy param
parser.add_argument("--energy_T", type=float, default=1.0)

# logit_gate params
parser.add_argument("--gate_threshold", type=float, default=10.0)
parser.add_argument("--gate_energy_keep", type=float, default=0.95)
parser.add_argument("--gate_k_max", type=int, default=64)
parser.add_argument("--gate_eps", type=float, default=1e-6)

# kpca_rff params
parser.add_argument("--kpca_gamma", type=float, default=1.0)
parser.add_argument("--kpca_M", type=int, default=2048)
parser.add_argument("--kpca_exp_var_ratio", type=float, default=0.95)

# gradsubspace params
parser.add_argument("--grad_n", type=int, default=512,
                    help="Mini-batch size sampled from ID train feats to compute S")
parser.add_argument("--grad_exp_var_ratio", type=float, default=0.95,
                    help="Explained variance ratio for selecting k")
parser.add_argument("--grad_center", action="store_true",
                    help="Center sampled ID features before SVD")
parser.add_argument("--grad_eps", type=float, default=1e-8)

# KL-to-uniform mixing controls (OOD)
parser.add_argument("--grad_kl_thr", type=float, default=1.0,
                    help="Threshold in sigmoid for mixing using KL(U||p)")
parser.add_argument("--grad_kl_temp", type=float, default=1.0,
                    help="Temperature for sigmoid mixing; larger => smoother")
parser.add_argument("--neco_dim", type=int, default=128,
help="NECO: number of PCA dimensions to keep (1..feature_dim)."),
parser.add_argument("--vim_dim", type=int, default=0,
help="ViM principal space dimension D. 0 picks a default based on feat_dim and num_classes."),
parser.add_argument("--vim_fit_max", type=int, default=50000,
                help="Max number of ID train features to sample for ViM PCA and alpha."),




args = parser.parse_args()

set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [13, 37, 101, 202, 999]

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


print("Device:", device)

methods = parse_methods(args)
print("Methods:", methods)

# transforms: match your script (ImageNet preprocessing)
weights = ViT_B_16_Weights.DEFAULT
# weights = ResNet18_Weights.DEFAULT
# weights = ResNet34_Weights.DEFAULT
base_tf = weights.transforms()
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
test_tfm = base_tf

# datasets
id_train = build_dataset(args.id_dataset, "train", args.data_dir, test_tfm, ood_dir=None)
id_test  = build_dataset(args.id_dataset, "test",  args.data_dir, test_tfm,  ood_dir=None)

ood_split = "test"
if args.ood_dataset.lower() == "svhn":
    ood_split = "test"

# For textures/places, user passes ood_dir pointing to that dataset root
ood_dir = None
if args.ood_dataset.lower() in ["textures", "places", "tiny-imagenet-200"]:
    ood_dir = args.ood_dir

ood_test = build_dataset(args.ood_dataset, ood_split, args.data_dir, test_tfm, ood_dir=ood_dir)

# loaders (only used if cache miss)
train_loader = DataLoader(id_train, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
id_loader = DataLoader(id_test, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
ood_loader = DataLoader(ood_test, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

# num classes
num_classes = 100
if hasattr(id_train, "classes"):
    num_classes = len(id_train.classes)

# load model once
clf_model = build_classifier(args.arch, num_classes=num_classes, ckpt_path=args.ckpt, device=device).eval()
feat_model = make_penultimate_extractor(clf_model, args.arch, l2_normalize=(not args.no_l2)).to(device).eval()

# cache paths
p_train, p_idtst, p_ood = cache_paths(args)
print("\nCache paths:")
print("  ID train feats+labels:", p_train)
print("  ID test feats:", p_idtst)
print("  OOD test feats:", p_ood)

# load or extract features
print("\nLoading/extracting cached features...")
train_feats, train_labels = load_or_extract_id_train(feat_model, train_loader, device, p_train)
id_feats = load_or_extract_feats(feat_model, id_loader, device, p_idtst)
ood_feats = load_or_extract_feats(feat_model, ood_loader, device, p_ood)
print("  train_feats:", tuple(train_feats.shape), "train_labels:", tuple(train_labels.shape))
print("  id_feats   :", tuple(id_feats.shape))
print("  ood_feats  :", tuple(ood_feats.shape), args.ood_dataset)

W, b = get_classifier_linear(clf_model, args.arch)

# alphas = np.arange(0.1, 1.0, 0.1)

# rows, best = sweep_alpha_id_only(
#     train_feats=train_feats,
#     train_labels=train_labels,
#     id_feats=id_feats,
#     W=W,
#     b=b,
#     num_classes=num_classes,
#     alphas=alphas,
#     k_mode="stable",
#     min_perp_dim=64,
#     beta_mode="median_ratio",     # or "std_ratio" or "fixed"
#     fixed_beta=1.0,
#     select_quantile=0.99,         # minimize 99th percentile of ID scores
# )
# best_alpha = best["alpha"]
# best_beta  = best["beta"]


# subspaces = fit_affine_pca_subspaces(
#     train_feats, train_labels, num_classes,
#     alpha=float(best_alpha),
#     k_mode="stable",
#     min_perp_dim=64,
#     verbose=True,
# )[0]
# id_logits  = _logits_from_feats(id_feats,  W.to(device), None if b is None else b.to(device))
# ood_logits = _logits_from_feats(ood_feats, W.to(device), None if b is None else b.to(device))

# s_res_id, a_id   = score_parts_residual_and_align(id_feats,  id_logits,  subspaces, W.to(device))
# s_res_ood, a_ood = score_parts_residual_and_align(ood_feats, ood_logits, subspaces, W.to(device))

# s_aug_id  = s_res_id  + best_beta * a_id
# s_aug_ood = s_res_ood + best_beta * a_ood
# # feed s_aug_id / s_aug_ood into report_metrics(...)

best_alpha, results = select_alpha_by_perp_participation_ratio(
    train_feats, train_labels, num_classes,
    alphas=[0.2,0.3, 0.4, 0.5, 0.6, 0.7],
    eps=1e-6
)
print(best_alpha, results)