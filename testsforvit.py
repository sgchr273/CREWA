#!/usr/bin/env python3
"""
Subspace diagnostics suite (single-run script)

What it does:
- Builds ID train, ID test, and OOD test datasets
- Loads a finetuned classifier checkpoint (ResNet18/34 or ViT-B/16)
- Builds a penultimate feature extractor
- Caches extracted features (and labels where applicable)
- Fits per-class affine PCA subspaces (your method)
- Runs diagnostics tests:
  1) Subspace-based label prediction accuracy + avg margin (s2 - s1)
  2) True-class rank stats (top1/top2/top3, mean/median rank)
  3) Subspace overlap summary (mean off-diagonal overlap)
  4) Spearman rank correlation of scores when k_max changes (k1 vs k2)
  5) Nearest class mean baseline accuracy
  6) Optional: ID vs OOD score summary (s1 and ratio)

Key fixes vs your pasted script:
- One argparse parser (no ap/parser mix)
- Removed undefined functions (parse_methods, load_pt) and implemented cache_paths
- Fixed load_or_extract_feats to return (feats, labels) consistently
- Use deterministic transforms for feature extraction (no random aug during caching)
- Device handling consistent with --device
- num_classes handling consistent (infer from dataset unless overridden)
- ViT head replacement consistent (use model.heads.head)
- Pin-memory conditioned on CUDA
"""

import os
import copy
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    resnet18, resnet34,
    vit_b_16,
    ViT_B_16_Weights,
    ResNet18_Weights,
)
from sklearn.metrics import roc_auc_score, roc_curve

def fpr_at_95_tpr(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    # first index where tpr >= 0.95
    idx = np.searchsorted(tpr, 0.95, side="left")
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx]), float(thr[idx])

def eval_ood(id_scores, ood_scores, name="score"):
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])  # OOD=1
    y_score = np.concatenate([id_scores, ood_scores])

    auc = roc_auc_score(y_true, y_score)
    fpr95, thr95 = fpr_at_95_tpr(y_true, y_score)

    # also check flipped sign to detect convention errors
    auc_flip = roc_auc_score(y_true, -y_score)
    fpr95_flip, thr95_flip = fpr_at_95_tpr(y_true, -y_score)

    print(f"\n[OOD eval: {name}]")
    print({"auroc": float(auc), "fpr@95tpr": float(fpr95), "thr@95tpr": float(thr95)})
    print({"auroc_flipped": float(auc_flip), "fpr@95tpr_flipped": float(fpr95_flip), "thr@95tpr_flipped": float(thr95_flip)})

# ----------------------------
# Utils
# ----------------------------
def safe_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)

def pick_device(device_str: str) -> torch.device:
    s = device_str.lower()
    if s.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARN] --device=cuda requested but CUDA not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")

def cache_paths(args) -> Tuple[str, str, str, str]:
    """
    Cache is independent of OOD choice for ID train/test.
    OOD cache depends on ood_dataset (and ood_dir basename if ImageFolder-like).
    """
    os.makedirs(args.cache_dir, exist_ok=True)

    ckpt_tag = safe_name(os.path.basename(args.ckpt))
    arch_tag = safe_name(args.arch)
    id_tag = safe_name(args.id_dataset)

    base_id = f"{id_tag}__{arch_tag}__{ckpt_tag}"
    p_train = os.path.join(args.cache_dir, base_id + "__id_train_feats_labels.pt")
    p_idtst = os.path.join(args.cache_dir, base_id + "__id_test_feats_labels.pt")

    ood_tag = safe_name(args.ood_dataset)
    extra = ""
    if args.ood_dataset.lower() in ["textures", "places", "tiny-imagenet-200"] and args.ood_dir:
        extra = "__" + safe_name(os.path.basename(args.ood_dir.rstrip("/")))
    base_ood = f"{id_tag}__{ood_tag}{extra}__{arch_tag}__{ckpt_tag}"
    p_ood = os.path.join(args.cache_dir, base_ood + "__ood_test_feats.pt")

    return p_train, p_idtst, p_ood, base_id

# ----------------------------
# Data
# ----------------------------
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
        # torchvision SVHN split must be "train" or "test"
        if split not in ["train", "test"]:
            raise ValueError("SVHN split must be 'train' or 'test'")
        return datasets.SVHN(root=data_dir, split=split, download=True, transform=tfm)

    if name in ["mnist", "fashionmnist"]:
        train = (split == "train")
        tfm_3ch = transforms.Compose([transforms.Grayscale(num_output_channels=3), tfm])
        if name == "mnist":
            return datasets.MNIST(root=data_dir, train=train, download=True, transform=tfm_3ch)
        return datasets.FashionMNIST(root=data_dir, train=train, download=True, transform=tfm_3ch)

    if name in ["textures", "places"]:
        if ood_dir is None:
            raise ValueError("For textures/places, pass --ood_dir pointing to the parent folder containing those subfolders.")
        return datasets.ImageFolder(os.path.join(ood_dir, name), transform=tfm)

    if name == "tiny-imagenet-200":
        if ood_dir is None:
            raise ValueError("For tiny-imagenet-200, pass --ood_dir pointing to the parent folder containing tiny-imagenet-200/val.")
        return datasets.ImageFolder(os.path.join(ood_dir, name, "val"), transform=tfm)

    raise ValueError(f"Unknown dataset: {name}")

# ----------------------------
# Model loading + feature extraction
# ----------------------------
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

@torch.no_grad()
def extract_features(feat_model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    feat_model.eval()
    all_feats, all_labels = [], []
    for imgs, labs in loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = feat_model(imgs)
        all_feats.append(feats.detach().cpu().float())
        all_labels.append(labs.detach().cpu().long())
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)

def load_or_extract_feats_labels(
    feat_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    path: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "feats" in ckpt and "labels" in ckpt:
            return ckpt["feats"].float(), ckpt["labels"].long()
        raise RuntimeError(f"Cache at {path} exists but is not a dict with keys 'feats' and 'labels'.")
    feats, labels = extract_features(feat_model, loader, device)
    torch.save({"feats": feats.cpu(), "labels": labels.cpu()}, path)
    return feats, labels

def load_or_extract_feats_only(
    feat_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    path: str,
) -> torch.Tensor:
    if os.path.exists(path):
        t = torch.load(path, map_location="cpu")
        if torch.is_tensor(t):
            return t.float()
        if isinstance(t, dict) and "feats" in t:
            return t["feats"].float()
        raise RuntimeError(f"Cache at {path} exists but is not a tensor (or dict with 'feats').")
    feats, _ = extract_features(feat_model, loader, device)
    torch.save(feats.cpu(), path)
    return feats

# ----------------------------
# Your subspace method (unchanged)
# ----------------------------
def fit_affine_pca_subspaces(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    energy_keep: float = 0.95,
    k_max: int = 150,
    eps: float = 1e-6,
) -> Dict[int, Dict[str, torch.Tensor]]:
    feats = feats.float()
    labels = labels.long()
    models: Dict[int, Dict[str, torch.Tensor]] = {}

    for c in range(num_classes):
        Xc = feats[labels == c]
        if Xc.shape[0] < 5:
            raise RuntimeError(f"Class {c} has too few samples ({Xc.shape[0]}).")
        mu = Xc.mean(dim=0)
        X = Xc - mu

        _, S, Vh = torch.linalg.svd(X, full_matrices=False)
        V = Vh.transpose(0, 1)
        n = X.shape[0]
        lam = (S**2) / max(n - 1, 1)

        energy = torch.cumsum(lam, dim=0) / (lam.sum() + eps)
        k = int(torch.searchsorted(energy, torch.tensor(energy_keep, device=energy.device)).item()) + 1
        r = V.shape[1]
        k = min(k, k_max, max(r - 1, 1))

        V_perp = V[:, k:]
        lam_perp = lam[k:]

        if V_perp.numel() == 0:
            k = max(k - 1, 0)
            V_perp = V[:, k:]
            lam_perp = lam[k:]
            if V_perp.numel() == 0:
                V_perp = V[:, -1:].contiguous()
                lam_perp = lam[-1:].contiguous()

        models[c] = {"mu": mu.contiguous(), "V_perp": V_perp.contiguous(), "lam_perp": lam_perp.contiguous()}
    return models

@torch.no_grad()
def score_subspaces_ratio_min(
    feats: torch.Tensor,
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    eps: float = 1e-2,
    normalize_by_perp_dim: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    feats = feats.float()
    classes = sorted(subspaces.keys())

    scores_per_class = []
    for c in classes:
        mu = subspaces[c]["mu"]
        Vp = subspaces[c]["V_perp"]
        lam = subspaces[c]["lam_perp"]

        z = feats - mu.unsqueeze(0)
        coeff = z @ Vp
        s = (coeff**2) / (lam.unsqueeze(0) + eps)
        s = s.sum(dim=1)
        if normalize_by_perp_dim:
            s = s / float(max(Vp.shape[1], 1))
        scores_per_class.append(s)

    scores = torch.stack(scores_per_class, dim=1)
    vals, _ = torch.topk(scores, k=2, dim=1, largest=False)
    s1, s2 = vals[:, 0], vals[:, 1]
    ratio = s1 / (s2 + eps)
    return s1.cpu().numpy().astype(np.float64), ratio.cpu().numpy().astype(np.float64)

# ----------------------------
# Diagnostics / tests
# ----------------------------
@torch.no_grad()
def predict_labels_from_subspaces(
    feats: torch.Tensor,
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    eps: float = 1e-6,
    normalize_by_perp_dim: bool = False,
    return_scores: bool = False,
):
    feats = feats.float()
    classes = sorted(subspaces.keys())

    scores_per_class = []
    for c in classes:
        mu = subspaces[c]["mu"]
        Vp = subspaces[c]["V_perp"]
        lam = subspaces[c]["lam_perp"]

        z = feats - mu.unsqueeze(0)
        coeff = z @ Vp
        s = (coeff * coeff) / (lam.unsqueeze(0) + eps)
        s = s.sum(dim=1)
        if normalize_by_perp_dim:
            s = s / float(max(Vp.shape[1], 1))
        scores_per_class.append(s)

    scores = torch.stack(scores_per_class, dim=1)  # (N,C)
    vals, idxs = torch.topk(scores, k=2, dim=1, largest=False)
    s1, s2 = vals[:, 0], vals[:, 1]
    y_hat = torch.tensor(classes, device=scores.device)[idxs[:, 0]]

    if return_scores:
        return y_hat.cpu().numpy(), s1.cpu().numpy(), s2.cpu().numpy(), scores.cpu().numpy()
    return y_hat.cpu().numpy(), s1.cpu().numpy(), s2.cpu().numpy()

def eval_subspace_classifier(id_feats, id_labels, subspaces, eps=1e-6, normalize_by_perp_dim=False):
    y_hat, s1, s2 = predict_labels_from_subspaces(id_feats, subspaces, eps=eps, normalize_by_perp_dim=normalize_by_perp_dim)
    y_true = id_labels.long().cpu().numpy()
    return {"acc": float((y_hat == y_true).mean()), "avg_margin_s2_minus_s1": float(np.mean(s2 - s1))}

@torch.no_grad()
def true_class_rank_stats(id_feats, id_labels, subspaces, eps=1e-6, normalize_by_perp_dim=False):
    y_true = id_labels.long().cpu().numpy()
    classes = sorted(subspaces.keys())
    class_to_col = {c: i for i, c in enumerate(classes)}

    _, _, _, scores = predict_labels_from_subspaces(
        id_feats, subspaces, eps=eps, normalize_by_perp_dim=normalize_by_perp_dim, return_scores=True
    )
    scores_t = torch.from_numpy(scores)
    true_cols = torch.tensor([class_to_col[int(y)] for y in y_true], dtype=torch.long)

    ranks = torch.argsort(scores_t, dim=1)  # ascending, best=0
    inv_rank = torch.empty_like(ranks)
    inv_rank.scatter_(1, ranks, torch.arange(ranks.size(1)).unsqueeze(0).expand_as(ranks))
    true_rank = inv_rank[torch.arange(scores_t.size(0)), true_cols]

    return {
        "mean_true_rank": float(true_rank.float().mean().item()),
        "median_true_rank": float(true_rank.float().median().item()),
        "pct_true_in_top1": float((true_rank == 0).float().mean().item()),
        "pct_true_in_top2": float((true_rank <= 1).float().mean().item()),
        "pct_true_in_top3": float((true_rank <= 2).float().mean().item()),
    }

@torch.no_grad()
def subspace_overlap_summary(subspaces: Dict[int, Dict[str, torch.Tensor]]):
    classes = sorted(subspaces.keys())
    C = len(classes)
    M = torch.zeros((C, C), dtype=torch.float64)
    for i, ci in enumerate(classes):
        Vi = subspaces[ci]["V_perp"]
        for j, cj in enumerate(classes):
            Vj = subspaces[cj]["V_perp"]
            A = Vi.t() @ Vj
            denom = float(max(min(Vi.shape[1], Vj.shape[1]), 1))
            M[i, j] = (A.pow(2).sum() / denom).item()
    off = M[~torch.eye(C, dtype=torch.bool)]
    return {
        "mean_offdiag_overlap": float(off.mean().item()),
        "median_offdiag_overlap": float(off.median().item()),
        "min_offdiag_overlap": float(off.min().item()),
        "max_offdiag_overlap": float(off.max().item()),
    }

def spearman_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float(np.mean(ra * rb))

@torch.no_grad()
def nearest_class_mean_classifier(train_feats, train_labels, test_feats, num_classes):
    mus = []
    for c in range(num_classes):
        Xc = train_feats[train_labels == c]
        mus.append(Xc.mean(dim=0))
    mu = torch.stack(mus, dim=0)  # (C,d)

    x2 = (test_feats**2).sum(dim=1, keepdim=True)
    m2 = (mu**2).sum(dim=1).unsqueeze(0)
    xm = test_feats @ mu.t()
    d2 = x2 + m2 - 2 * xm
    y_hat = torch.argmin(d2, dim=1)
    return y_hat.cpu().numpy()

def score_summary(name: str, s: np.ndarray):
    return {
        f"{name}_mean": float(np.mean(s)),
        f"{name}_std": float(np.std(s)),
        f"{name}_min": float(np.min(s)),
        f"{name}_max": float(np.max(s)),
    }

def pick_weights_transforms(arch: str):
    """
    Use deterministic ImageNet-style preprocessing.
    For simplicity, use the appropriate weights' transforms for the backbone family.
    """
    arch = arch.lower()
    if arch.startswith("vit"):
        return ViT_B_16_Weights.DEFAULT.transforms()
    # ResNet transforms are also ImageNet-style; ResNet18 weights include transforms().
    return ResNet18_Weights.DEFAULT.transforms()



# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--id_dataset", type=str, default="cifar10",
                        help="cifar10, cifar100, svhn, mnist, fashionmnist")
    parser.add_argument("--ood_dataset", type=str, default="cifar100",
                        help="svhn, cifar100, mnist, fashionmnist, textures, places, tiny-imagenet-200")
    parser.add_argument("--ood_dir", type=str,
                        default="/home/sgchr/Documents/Multiple_spaces/Imagenet/ood_data/")
    parser.add_argument("--data_dir", type=str, default="./data")

    # model
    parser.add_argument("--arch", type=str, default="vit_b_16",
                        help="resnet18, resnet34, vit_b_16")
    parser.add_argument("--ckpt", type=str, default="/home/sgchr/Documents/Subspaces/checkpoints/vit_b16_cifar10_best_finetuning.pt",
                        help="Path to checkpoint (state_dict or dict containing model_state/model/state_dict)")
    parser.add_argument("--no_l2", action="store_true",
                        help="Disable L2 normalization on extracted penultimate features")

    # loader
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    # caching
    parser.add_argument("--cache_dir", type=str, default="./cache_feats",
                        help="Directory to save/load cached features")

    # subspace params
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Override class count. If omitted, inferred from ID dataset.")
    parser.add_argument("--energy_keep", type=float, default=0.95)
    parser.add_argument("--k_max", type=int, default=350)
    parser.add_argument("--k1", type=int, default=20)
    parser.add_argument("--k2", type=int, default=100)
    parser.add_argument("--score_eps", type=float, default=1e-2)
    parser.add_argument("--pred_eps", type=float, default=1e-6)
    parser.add_argument("--normalize_by_perp_dim", action="store_true")

    # misc
    parser.add_argument("--device", type=str, default="cuda",
                        help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    print("Device:", device)

    tfm = pick_weights_transforms(args.arch)  # deterministic transforms

    # datasets
    id_train = build_dataset(args.id_dataset, "train", args.data_dir, tfm, ood_dir=None)
    id_test  = build_dataset(args.id_dataset, "test",  args.data_dir, tfm, ood_dir=None)

    ood_split = "test"
    if args.ood_dataset.lower() == "svhn":
        ood_split = "test"
    ood_dir = args.ood_dir if args.ood_dataset.lower() in ["textures", "places", "tiny-imagenet-200"] else None
    ood_test = build_dataset(args.ood_dataset, ood_split, args.data_dir, tfm, ood_dir=ood_dir)

    # infer num_classes
    inferred_num_classes = None
    if hasattr(id_train, "classes"):
        inferred_num_classes = len(id_train.classes)
    else:
        # SVHN/MNIST/FashionMNIST also effectively have 10, but keep simple
        inferred_num_classes = 10

    num_classes = args.num_classes if args.num_classes is not None else inferred_num_classes
    if args.num_classes is not None and args.num_classes != inferred_num_classes:
        print(f"[WARN] --num_classes={args.num_classes} differs from inferred={inferred_num_classes}. Using --num_classes.")

    # loaders (no shuffle for deterministic feature caching)
    pin = (device.type == "cuda")
    train_loader = DataLoader(id_train, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)
    id_loader = DataLoader(id_test, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=pin)
    ood_loader = DataLoader(ood_test, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    # model
    clf_model = build_classifier(args.arch, num_classes=num_classes, ckpt_path=args.ckpt, device=device).eval()
    feat_model = make_penultimate_extractor(clf_model, args.arch, l2_normalize=(not args.no_l2)).to(device).eval()

    # cache paths
    p_train, p_idtst, p_ood, base_id = cache_paths(args)
    print("\nCache paths:")
    print("  ID train feats+labels:", p_train)
    print("  ID test feats+labels :", p_idtst)
    print("  OOD test feats       :", p_ood)

    # load/extract caches
    print("\nLoading/extracting cached features...")
    train_feats, train_labels = load_or_extract_feats_labels(feat_model, train_loader, device, p_train)
    id_feats, id_labels = load_or_extract_feats_labels(feat_model, id_loader, device, p_idtst)
    ood_feats = load_or_extract_feats_only(feat_model, ood_loader, device, p_ood)

    # sanity checks
    print("\n[Data]")
    print(" train_feats:", tuple(train_feats.shape), train_feats.dtype)
    print(" train_labels:", tuple(train_labels.shape), train_labels.dtype, "unique:", int(train_labels.unique().numel()))
    print(" id_feats:", tuple(id_feats.shape), id_feats.dtype)
    print(" id_labels:", tuple(id_labels.shape), id_labels.dtype, "unique:", int(id_labels.unique().numel()))
    print(" ood_feats:", tuple(ood_feats.shape), ood_feats.dtype)
    assert train_feats.shape[1] == id_feats.shape[1] == ood_feats.shape[1], "Feature dim mismatch between splits."

    # fit main subspaces
    print(f"\n[Fit subspaces] energy_keep={args.energy_keep} k_max={args.k_max}")
    subspaces = fit_affine_pca_subspaces(
        feats=train_feats,
        labels=train_labels,
        num_classes=num_classes,
        energy_keep=args.energy_keep,
        k_max=args.k_max,
    )

    # test 1
    out1 = eval_subspace_classifier(id_feats, id_labels, subspaces, eps=args.pred_eps,
                                   normalize_by_perp_dim=args.normalize_by_perp_dim)
    print("\n[Test 1] Subspace label prediction (argmin residual)")
    print(out1)

    # test 2
    out2 = true_class_rank_stats(id_feats, id_labels, subspaces, eps=args.pred_eps,
                                 normalize_by_perp_dim=args.normalize_by_perp_dim)
    print("\n[Test 2] True-class rank stats")
    print(out2)

    # test 3
    out3 = subspace_overlap_summary(subspaces)
    print("\n[Test 3] Subspace overlap summary (higher off-diag => class subspaces look similar)")
    print(out3)

    # test 4
    print(f"\n[Test 4] Rank invariance across k_max: k1={args.k1} vs k2={args.k2}")
    sub_k1 = fit_affine_pca_subspaces(train_feats, train_labels, num_classes, args.energy_keep, args.k1)
    sub_k2 = fit_affine_pca_subspaces(train_feats, train_labels, num_classes, args.energy_keep, args.k2)
    s1_k1, ratio_k1 = score_subspaces_ratio_min(id_feats, sub_k1, eps=args.score_eps, normalize_by_perp_dim=args.normalize_by_perp_dim)
    s1_k2, ratio_k2 = score_subspaces_ratio_min(id_feats, sub_k2, eps=args.score_eps, normalize_by_perp_dim=args.normalize_by_perp_dim)
    print({"spearman_s1": spearman_rank_corr(s1_k1, s1_k2),
           "spearman_ratio": spearman_rank_corr(ratio_k1, ratio_k2)})

    # test 5
    print("\n[Test 5] Nearest class mean baseline (feature separability sanity check)")
    y_hat_ncm = nearest_class_mean_classifier(train_feats, train_labels, id_feats, num_classes)
    acc_ncm = float((y_hat_ncm == id_labels.cpu().numpy()).mean())
    print({"ncm_acc": acc_ncm})

    # test 6 (ID vs OOD summary)
    print("\n[Test 6] ID vs OOD score summary (s1 and ratio) using main subspaces")
    s1_id, ratio_id = score_subspaces_ratio_min(id_feats, subspaces, eps=args.score_eps, normalize_by_perp_dim=args.normalize_by_perp_dim)
    s1_ood, ratio_ood = score_subspaces_ratio_min(ood_feats, subspaces, eps=args.score_eps, normalize_by_perp_dim=args.normalize_by_perp_dim)
    out6 = {}
    out6.update(score_summary("s1_id", s1_id))
    out6.update(score_summary("s1_ood", s1_ood))
    out6.update(score_summary("ratio_id", ratio_id))
    out6.update(score_summary("ratio_ood", ratio_ood))
    print(out6)
    eval_ood(s1_id, s1_ood, name="s1")
    eval_ood(ratio_id, ratio_ood, name="ratio")

    print("\n[Done]")

if __name__ == "__main__":
    main()
