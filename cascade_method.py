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
    Backward compatible: prints the same metrics for all methods.
    For NCI only, also RETURNS AUROC (OOD positive) so you can tune varpi.
    For other methods, returns None (no behavior change needed).
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

    # Only NCI returns AUROC; other methods unchanged (still print only).
    if method_name.strip().lower() in {"nci", "nci-temp", "nci_score"}:
        return float(auroc)
    return None



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
        # if args.mahalanobis: methods.append("mahalanobis")
        # if args.energy: methods.append("energy")
        # if args.msp: methods.append("msp")
        # if args.logit_gate: methods.append("logit_gate")
        # if args.kpca_rff: methods.append("kpca_rff")
        # if args.gradsubspace: methods.append("gradsubspace")
        # if args.neco: methods.append("neco")

    if len(methods) == 0:
        raise RuntimeError("No methods selected. Use --methods ... or flags.")

    allowed = {"subspaces", "mahalanobis", "energy", "msp", "logit_gate", "kpca_rff", "gradsubspace", "neco", "vim", "nci"}
    out = []
    for m in methods:
        if m not in allowed:
            raise ValueError(f"Unknown/disabled method: {m}. Allowed: {sorted(list(allowed))}")
        if m not in out:
            out.append(m)
    return out


import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def _as_torch(x, *, dtype=None, device=None) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x
    else:
        t = torch.from_numpy(np.asarray(x))
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device=device)
    return t


def fit_affine_pca_subspaces(
    feats,
    labels,
    num_classes: int,
    energy_keep: float = 0.70,
    eps: float = 1e-6,
) -> Dict[int, Dict[str, torch.Tensor]]:
    feats  = _as_torch(feats,  dtype=torch.float32)
    labels = _as_torch(labels, dtype=torch.long, device=feats.device)

    N, D = feats.shape
    if labels.numel() != N:
        raise RuntimeError(f"labels length mismatch: {labels.numel()} vs {N}")
    if N < 5:
        raise RuntimeError(f"Too few samples ({N}).")

    mu_c = feats.new_zeros((num_classes, D))
    for c in range(num_classes):
        idx = (labels == c)
        nc = int(idx.sum().item())
        if nc == 0:
            raise RuntimeError(f"class {c} has zero samples")
        mu_c[c] = feats[idx].mean(dim=0)

    X = feats - mu_c[labels]  # [N, D]
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh.transpose(0, 1)    # [D, r]
    r = V.shape[1]
    lam = (S ** 2) / max(N - 1, 1)  # [r]

    energy = torch.cumsum(lam, dim=0) / (lam.sum() + eps)
    k_energy = int(torch.searchsorted(energy, torch.tensor(energy_keep, device=energy.device)).item()) + 1

    if r == 1:
        k = 0
    else:
        k = min(k_energy, r - 1)

    V_keep = V[:, :k].contiguous()  # [D, k]
    V_perp = V[:, k:].contiguous()  # [D, p]
    lam_perp = lam[k:].contiguous() # [p]

    if V_perp.numel() == 0:
        V_perp = V[:, -1:].contiguous()
        lam_perp = lam[-1:].contiguous()

    print(f"[pooled within class pca] k={k} (energy_keep={energy_keep}, rank={r}, perp_dim={V_perp.shape[1]})")

    return {
        0: {
            "mu_c": mu_c.contiguous(),
            "V_keep": V_keep,
            "V_perp": V_perp,
            "lam_perp": lam_perp,
        }
    }


@torch.no_grad()
def fit_kept_stats_per_class(
    train_feats,
    train_labels,
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    num_classes: int,
    *,
    var_floor: float = 1e-6,
) -> None:
    """
    Adds kept-space stats to subspaces[0]:
      mu_q_c: [C, k]
      var_q_c: [C, k] (diagonal variance, floored)
    """
    m = subspaces[0]
    feats  = _as_torch(train_feats,  dtype=torch.float32)
    labels = _as_torch(train_labels, dtype=torch.long, device=feats.device)

    mu_c   = m["mu_c"].to(device=feats.device)
    Vk     = m["V_keep"].to(device=feats.device)  # [D, k]

    C = num_classes
    k = Vk.shape[1]

    if k == 0:
        # no kept dims; keep empty tensors for consistency
        m["mu_q_c"]  = feats.new_zeros((C, 0))
        m["var_q_c"] = feats.new_zeros((C, 0))
        return

    mu_q_c  = feats.new_zeros((C, k))
    var_q_c = feats.new_zeros((C, k))

    for c in range(C):
        idx = (labels == c)
        if int(idx.sum().item()) < 2:
            raise RuntimeError(f"class {c} has too few samples for variance")
        z = feats[idx] - mu_c[c]          # [Nc, D]
        q = z @ Vk                        # [Nc, k]
        mu_q_c[c] = q.mean(dim=0)
        var_q_c[c] = q.var(dim=0, unbiased=True).clamp_min(float(var_floor))

    m["mu_q_c"]  = mu_q_c.contiguous()
    m["var_q_c"] = var_q_c.contiguous()

@torch.no_grad()
def calibrate_cascade_thresholds_id_only(
    id_feats,
    id_labels,
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    *,
    tau: float = 1e-2,
    eps: float = 1e-6,
    q_mu: float = 0.99,
    q_res: float = 0.99,
    q_keep: float = 0.99,
) -> Dict[str, float]:
    """
    Uses ID labels ONLY for calibration to get stable thresholds.
    """
    m = subspaces[0]
    feats  = _as_torch(id_feats,  dtype=torch.float32)
    labels = _as_torch(id_labels, dtype=torch.long, device=feats.device)

    mu_c     = m["mu_c"].to(device=feats.device)
    Vp       = m["V_perp"].to(device=feats.device)
    lam_perp = m["lam_perp"].to(device=feats.device)
    Vk       = m["V_keep"].to(device=feats.device)
    mu_q_c   = m["mu_q_c"].to(device=feats.device)
    var_q_c  = m["var_q_c"].to(device=feats.device)

    z = feats - mu_c[labels]             # [N, D]
    d2 = (z ** 2).sum(dim=1)             # [N]

    coeff = z @ Vp                       # [N, p]
    r = (coeff ** 2 / (lam_perp + float(tau)).unsqueeze(0)).sum(dim=1)

    if Vk.shape[1] == 0:
        # no kept dims; kept typicality can't be used
        mkeep = torch.zeros_like(d2)
    else:
        q = z @ Vk                       # [N, k]
        dq = q - mu_q_c[labels]          # [N, k]
        mkeep = (dq ** 2 / (var_q_c[labels] + float(eps))).sum(dim=1)

    T_mu   = float(torch.quantile(d2,    q_mu).item())
    T_res  = float(torch.quantile(r,     q_res).item())
    T_keep = float(torch.quantile(mkeep, q_keep).item())

    return {"T_mu": T_mu, "T_res": T_res, "T_keep": T_keep}

@torch.no_grad()
def score_topk_cascade_min(
    feats,
    logits,
    W,
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    *,
    topk: int = 5,
    tau: float = 1e-2,
    eps: float = 1e-6,
    T_mu: float,
    T_res: float,
    T_keep: float,
) -> np.ndarray:
    feats  = _as_torch(feats,  dtype=torch.float32)
    logits = _as_torch(logits, dtype=torch.float32, device=feats.device)
    W      = _as_torch(W,      dtype=torch.float32, device=feats.device)

    m = subspaces[0]
    mu_c     = m["mu_c"].to(device=feats.device)          # [C, D]
    Vp       = m["V_perp"].to(device=feats.device)        # [D, P]
    lam_perp = m["lam_perp"].to(device=feats.device)      # [P]
    Vk       = m["V_keep"].to(device=feats.device)        # [D, H]
    mu_q_c   = m["mu_q_c"].to(device=feats.device)        # [C, H]
    var_q_c  = m["var_q_c"].to(device=feats.device)       # [C, H]

    N, D = feats.shape
    K = min(int(topk), logits.shape[1])

    topk_idx = torch.topk(logits, k=K, dim=1).indices     # [N, K]

    # Expand features and class means to [N, K, D]
    f  = feats[:, None, :].expand(N, K, D)                # [N, K, D]
    mu = mu_c[topk_idx]                                   # [N, K, D]
    z  = f - mu                                            # [N, K, D]

    # Signal A: distance d2
    d2 = (z ** 2).sum(dim=2)                              # [N, K]

    # Signal B: discarded residual r
    coeff = torch.einsum("njd,dp->njp", z, Vp)            # [N, K, P]
    denom = (lam_perp + float(tau)).view(1, 1, -1)        # [1, 1, P]
    r = (coeff ** 2 / denom).sum(dim=2)                   # [N, K]

    # Kept part vector z_keep = z - proj_perp(z)
    z_perp = torch.einsum("njp,dp->njd", coeff, Vp)       # [N, K, D]
    z_keep = z - z_perp                                   # [N, K, D]

    # Signal D: kept typicality mkeep
    H = Vk.shape[1]
    if H == 0:
        mkeep = torch.zeros_like(d2)
    else:
        q    = torch.einsum("njd,dh->njh", z, Vk)         # [N, K, H]
        mu_q = mu_q_c[topk_idx]                           # [N, K, H]
        var_q = var_q_c[topk_idx]                          # [N, K, H]
        dq = q - mu_q
        mkeep = (dq ** 2 / (var_q + float(eps))).sum(dim=2)  # [N, K]

    # Signal C: alignment a in kept space, projecting weights into kept space too
    w_raw   = W[topk_idx]                                  # [N, K, D]
    w_coeff = torch.einsum("njd,dp->njp", w_raw, Vp)        # [N, K, P]
    w_perp  = torch.einsum("njp,dp->njd", w_coeff, Vp)      # [N, K, D]
    w_keep  = w_raw - w_perp                                # [N, K, D]

    u_n = F.normalize(z_keep, p=2, dim=2, eps=eps)
    w_n = F.normalize(w_keep, p=2, dim=2, eps=eps)
    cos = (u_n * w_n).sum(dim=2).clamp(-1.0, 1.0)          # [N, K]
    a = 1.0 - cos                                          # [N, K]

    # Cascade inside each class score
    score = torch.empty_like(d2)

    m1 = d2 > float(T_mu)
    score[m1] = d2[m1]

    m2 = (~m1) & (r > float(T_res))
    score[m2] = r[m2]

    m3 = (~m1) & (~m2) & (mkeep > float(T_keep))
    score[m3] = mkeep[m3]

    m4 = (~m1) & (~m2) & (~m3)
    score[m4] = a[m4]

    # Min across TopK candidates
    s = score.min(dim=1).values
    return s.detach().cpu().numpy().astype(np.float64)



@torch.no_grad()
def cascade_hit_rates_id_only(
    id_feats,
    id_labels,
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    *,
    tau: float = 1e-2,
    eps: float = 1e-6,
    q_mu: float = 0.99,
    q_res: float = 0.99,
    q_keep: float = 0.99,
) -> Dict[str, float]:
    """
    ID-only sanity check for cascade thresholds.

    Returns:
      T_mu, T_res, T_keep and stage hit rates:
        p1 = mean(d2 > T_mu)
        p2 = mean((d2<=T_mu) & (r > T_res))
        p3 = mean((d2<=T_mu) & (r<=T_res) & (mkeep > T_keep))
        p4 = remaining fraction that reaches alignment stage
    """
    m = subspaces[0]
    feats  = _as_torch(id_feats,  dtype=torch.float32)
    labels = _as_torch(id_labels, dtype=torch.long, device=feats.device)

    mu_c     = m["mu_c"].to(device=feats.device)
    Vp       = m["V_perp"].to(device=feats.device)
    lam_perp = m["lam_perp"].to(device=feats.device)
    Vk       = m["V_keep"].to(device=feats.device)
    mu_q_c   = m["mu_q_c"].to(device=feats.device)
    var_q_c  = m["var_q_c"].to(device=feats.device)

    # center using TRUE ID labels (calibration only)
    z = feats - mu_c[labels]                  # [N, D]

    # Signal A: distance
    d2 = (z ** 2).sum(dim=1)                  # [N]

    # Signal B: discarded residual
    coeff = z @ Vp                            # [N, p]
    r = (coeff ** 2 / (lam_perp + float(tau)).unsqueeze(0)).sum(dim=1)

    # Signal D: kept typicality
    if Vk.shape[1] == 0:
        mkeep = torch.zeros_like(d2)
    else:
        q = z @ Vk                            # [N, k]
        dq = q - mu_q_c[labels]               # [N, k]
        mkeep = (dq ** 2 / (var_q_c[labels] + float(eps))).sum(dim=1)

    # thresholds from ID quantiles
    T_mu   = float(torch.quantile(d2,    q_mu).item())
    T_res  = float(torch.quantile(r,     q_res).item())
    T_keep = float(torch.quantile(mkeep, q_keep).item())

    # stage hit rates
    m1 = d2 > T_mu
    m2 = (~m1) & (r > T_res)
    m3 = (~m1) & (~m2) & (mkeep > T_keep)
    m4 = (~m1) & (~m2) & (~m3)

    p1 = float(m1.float().mean().item())
    p2 = float(m2.float().mean().item())
    p3 = float(m3.float().mean().item())
    p4 = float(m4.float().mean().item())

    return {
        "T_mu": T_mu,
        "T_res": T_res,
        "T_keep": T_keep,
        "p_stage1_d2": p1,
        "p_stage2_res": p2,
        "p_stage3_keep": p3,
        "p_stage4_align": p4,
    }

@torch.no_grad()
def stage_usage_topk(
    feats, logits, W, subspaces,
    *, topk=5, tau=1e-2, eps=1e-6, T_mu=1.0, T_res=10.0, T_keep=10.0
):
    feats  = _as_torch(feats,  dtype=torch.float32)
    logits = _as_torch(logits, dtype=torch.float32, device=feats.device)
    W      = _as_torch(W,      dtype=torch.float32, device=feats.device)

    m = subspaces[0]
    mu_c     = m["mu_c"].to(device=feats.device)
    Vp       = m["V_perp"].to(device=feats.device)
    lam_perp = m["lam_perp"].to(device=feats.device)
    Vk       = m["V_keep"].to(device=feats.device)
    mu_q_c   = m["mu_q_c"].to(device=feats.device)
    var_q_c  = m["var_q_c"].to(device=feats.device)

    N, D = feats.shape
    K = min(int(topk), logits.shape[1])
    topk_idx = torch.topk(logits, k=K, dim=1).indices

    f  = feats[:, None, :].expand(N, K, D)
    z  = f - mu_c[topk_idx]

    d2 = (z ** 2).sum(dim=2)
    coeff = torch.einsum("njd,dp->njp", z, Vp)
    r = (coeff ** 2 / (lam_perp + float(tau)).view(1, 1, -1)).sum(dim=2)

    z_perp = torch.einsum("njp,dp->njd", coeff, Vp)
    z_keep = z - z_perp

    H = Vk.shape[1]
    if H == 0:
        mkeep = torch.zeros_like(d2)
    else:
        q = torch.einsum("njd,dh->njh", z, Vk)
        dq = q - mu_q_c[topk_idx]
        mkeep = (dq ** 2 / (var_q_c[topk_idx] + float(eps))).sum(dim=2)

    w_raw   = W[topk_idx]
    w_coeff = torch.einsum("njd,dp->njp", w_raw, Vp)
    w_perp  = torch.einsum("njp,dp->njd", w_coeff, Vp)
    w_keep  = w_raw - w_perp

    a = 1.0 - (F.normalize(z_keep, p=2, dim=2, eps=eps) *
               F.normalize(w_keep, p=2, dim=2, eps=eps)).sum(dim=2).clamp(-1, 1)

    # cascade per candidate
    m1 = d2 > float(T_mu)
    m2 = (~m1) & (r > float(T_res))
    m3 = (~m1) & (~m2) & (mkeep > float(T_keep))
    m4 = (~m1) & (~m2) & (~m3)

    # which stage wins after min over topK: approximate by taking argmin of score candidates
    score = torch.empty_like(d2)
    score[m1] = d2[m1]
    score[m2] = r[m2]
    score[m3] = mkeep[m3]
    score[m4] = a[m4]
    argmin = score.argmin(dim=1)  # which candidate class chosen by min

    # stage usage for the chosen candidate class
    chosen_m1 = m1[torch.arange(N, device=feats.device), argmin]
    chosen_m2 = m2[torch.arange(N, device=feats.device), argmin]
    chosen_m3 = m3[torch.arange(N, device=feats.device), argmin]
    chosen_m4 = m4[torch.arange(N, device=feats.device), argmin]

    return {
        "stage1": float(chosen_m1.float().mean().item()),
        "stage2": float(chosen_m2.float().mean().item()),
        "stage3": float(chosen_m3.float().mean().item()),
        "stage4": float(chosen_m4.float().mean().item()),
    }


@torch.no_grad()
def calibrate_cascade_thresholds_topkmin_id_only(
    id_feats,
    id_logits,
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    *,
    topk: int = 5,
    tau: float = 1e-2,
    eps: float = 1e-6,
    q_mu: float = 0.99,
    q_res: float = 0.99,
    q_keep: float = 0.99,
) -> Dict[str, float]:
    """
    ID-only calibration WITHOUT true labels:
      - pick pseudo class c* = argmin d2_c over TopK candidates
      - compute d2, residual, kept-typicality using that c*
      - set thresholds by quantiles

    This matches the topK-based inference geometry much better.
    """
    feats  = _as_torch(id_feats,  dtype=torch.float32)
    logits = _as_torch(id_logits, dtype=torch.float32, device=feats.device)

    m = subspaces[0]
    mu_c     = m["mu_c"].to(device=feats.device)       # [C, D]
    Vp       = m["V_perp"].to(device=feats.device)     # [D, P]
    lam_perp = m["lam_perp"].to(device=feats.device)   # [P]
    Vk       = m["V_keep"].to(device=feats.device)     # [D, H]
    mu_q_c   = m["mu_q_c"].to(device=feats.device)     # [C, H]
    var_q_c  = m["var_q_c"].to(device=feats.device)    # [C, H]

    N, D = feats.shape
    K = min(int(topk), logits.shape[1])
    topk_idx = torch.topk(logits, k=K, dim=1).indices  # [N, K]

    # compute d2 for each candidate class
    f  = feats[:, None, :].expand(N, K, D)             # [N, K, D]
    z  = f - mu_c[topk_idx]                             # [N, K, D]
    d2 = (z ** 2).sum(dim=2)                           # [N, K]

    # choose pseudo class = closest mean among topK
    j_star = d2.argmin(dim=1)                          # [N]
    c_star = topk_idx[torch.arange(N, device=feats.device), j_star]  # [N]

    # now compute signals w.r.t. chosen pseudo class
    z0 = feats - mu_c[c_star]                          # [N, D]
    d2_0 = (z0 ** 2).sum(dim=1)                        # [N]

    coeff0 = z0 @ Vp                                    # [N, P]
    r0 = (coeff0 ** 2 / (lam_perp + float(tau)).unsqueeze(0)).sum(dim=1)

    H = Vk.shape[1]
    if H == 0:
        mkeep0 = torch.zeros_like(d2_0)
    else:
        q0 = z0 @ Vk                                    # [N, H]
        dq0 = q0 - mu_q_c[c_star]                        # [N, H]
        mkeep0 = (dq0 ** 2 / (var_q_c[c_star] + float(eps))).sum(dim=1)

    T_mu   = float(torch.quantile(d2_0,    q_mu).item())
    T_res  = float(torch.quantile(r0,      q_res).item())
    T_keep = float(torch.quantile(mkeep0,  q_keep).item())
    return {"T_mu": T_mu, "T_res": T_res, "T_keep": T_keep}

def run_method_topk_cascade(
    train_feats,
    train_labels,
    id_feats,
    id_logits,
    id_labels,      # used ONLY for threshold calibration
    ood_feats,
    ood_logits,
    num_classes: int,
    energy_keep: float,
    *,
    W,
    topk: int = 5,
    tau: float = 1e-2,
    eps: float = 1e-6,
    q_mu: float = 0.99,
    q_res: float = 0.99,
    q_keep: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    subspaces = fit_affine_pca_subspaces(
        feats=train_feats,
        labels=train_labels,
        num_classes=num_classes,
        energy_keep=energy_keep,
        eps=eps,
    )


    fit_kept_stats_per_class(
        train_feats, train_labels, subspaces, num_classes,
        var_floor=1e-6,
    )

    th = calibrate_cascade_thresholds_topkmin_id_only(
        id_feats=id_feats,
        id_logits=id_logits,
        subspaces=subspaces,
        topk=5,
        tau=1e-2,
        eps=1e-6,
        q_mu=0.99,
        q_res=0.99,
        q_keep=0.99,
    )
    print("[cascade thresholds topk-min] ", th)

    stats = cascade_hit_rates_id_only(
            id_feats=id_feats,
            id_labels=id_labels,
            subspaces=subspaces,
            tau=1e-2,
            q_mu=0.99, q_res=0.99, q_keep=0.99,
        )
    print(stats)
    

    s_id = score_topk_cascade_min(
        id_feats, id_logits, W, subspaces,
        topk=topk, tau=tau, eps=eps,
        T_mu=th["T_mu"], T_res=th["T_res"], T_keep=th["T_keep"],
    )
    s_ood = score_topk_cascade_min(
        ood_feats, ood_logits, W, subspaces,
        topk=topk, tau=tau, eps=eps,
        T_mu=th["T_mu"], T_res=th["T_res"], T_keep=th["T_keep"],
    )

    id_usage = stage_usage_topk(
            feats=id_feats,
            logits=id_logits,
            W=W,
            subspaces=subspaces,
            topk=5,
            tau=1e-2,
            eps=1e-6,
            T_mu=th["T_mu"],
            T_res=th["T_res"],
            T_keep=th["T_keep"],
        )
    print("ID stage usage:", id_usage)

    ood_usage = stage_usage_topk(
            feats=ood_feats,
            logits=ood_logits,
            W=W,
            subspaces=subspaces,
            topk=5,
            tau=1e-2,
            eps=1e-6,
            T_mu=th["T_mu"],
            T_res=th["T_res"],
            T_keep=th["T_keep"],
        )
    print("OOD stage usage:", ood_usage)
    return s_id, s_ood, th









# -------------------------
# Main
# -------------------------
def main():
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

    # subspaces params
    parser.add_argument("--energy_keep", type=float, default=0.20)  ##0.30 for resnet18, 0.20 for vitb16,  0.90 for cifar100,resnet34 
    parser.add_argument("--k_max", type=int, default=64)


    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    if hasattr(id_test, "targets"):
        id_labels = torch.as_tensor(id_test.targets, dtype=torch.long)
    elif hasattr(id_test, "labels"):
        id_labels = torch.as_tensor(id_test.labels, dtype=torch.long)
    else:
        raise RuntimeError("Could not find labels on id_test. Use Option 1.")



    # head params for logits-based methods
    W, b = get_classifier_linear(clf_model, args.arch)

    # run methods
    for m in methods:
        if m == "subspaces":

            logits_id = logits_from_feats(id_feats, W, b)
            logits_ood = logits_from_feats(ood_feats, W, b)
            # s_id, s_ood, used = run_method_subspaces_rich(
            #     train_feats, train_labels,
            #     id_feats, logits_id,
            #     ood_feats, logits_ood,
            #     num_classes=num_classes,
            #     energy_keep=0.30,
            #     k_max=150,
            #     W=W,
            #     corr_thr=0.95,
            #     clip_val=6.0,
            #     drop_names={"z_norm", "feat_norm"},     # optional: fast redundancy removal
            #     use_diag_scorer=False,                  # try True if full-cov is unstable
            #     do_forward_select=True,
            #     forward_k=8,
            # )
            s_id, s_ood, th=run_method_topk_cascade(
                train_feats,
                train_labels,
                id_feats,
                logits_id,
                id_labels,      # used ONLY for threshold calibration
                ood_feats,
                logits_ood,
                num_classes=num_classes,
                energy_keep=args.energy_keep,
            
                W=W,
                topk=5,
                tau=1e-2,
                eps=1e-6,
                q_mu=0.99,
                q_res=0.99,
                q_keep=0.99,
            )

           
            report_metrics("subspaces (s1)", s_id, s_ood)






if __name__ == "__main__":
    main()

