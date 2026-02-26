#!/usr/bin/env python3
"""
OOD detection with GLOBAL affine PCA on penultimate features.

ID  = CIFAR-10
OOD = SVHN (default) or ImageFolder (Textures/Places/iSUN/LSUN/etc.)

Method:
1) Train / load an ImageNet pretrained backbone finetuned on CIFAR-10.
2) Extract penultimate features for:
   - CIFAR-10 train (fit global PCA)
   - CIFAR-10 test  (ID eval)
   - OOD test       (OOD eval)
3) Fit ONE global affine PCA on CIFAR-10 train features:
      mu, V, lam
   Choose k by cumulative energy (clamped by k_max).
4) Compute per sample whitened residual in complement:
      s(x) = sum_{i>k} (v_i^T (x-mu))^2 / (lam_i + eps)
5) Final score is the ratio of top-2 scores.
   Global PCA has a single score per sample, so we compute the ratio by
   splitting the complement directions into two halves and treating them as
   two "partial residual" scores:
      sA(x), sB(x)
      ratio(x) = min(sA,sB) / (max(sA,sB) + eps)

Metrics:
  AUROC and FPR@95%TPR with OOD as positive class, higher score => more OOD.

Notes:
- Because global PCA provides only one subspace score per sample, the ratio is
  not defined across classes. This script defines a consistent "top-2" ratio
  by splitting the complement space into two parts and taking the ratio of the
  smaller to the larger partial residual.

Example:
  python global_pca_ratio.py \
    --arch resnet18 \
    --ckpt /path/to/resnet18_cifar10_ckpt.pt \
    --ood svhn
"""

import random
import argparse
import copy
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, roc_curve


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_state_dict_flexible(ckpt_path: str, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location=map_location)
    if isinstance(obj, dict):
        for k in ["model_state", "model", "state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj
    raise ValueError(f"Unrecognized checkpoint format at: {ckpt_path}")


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_feats, all_labels = [], []
    for imgs, labs in loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = model(imgs)
        all_feats.append(feats.detach().cpu().float())
        all_labels.append(labs.detach().cpu().long())
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


def make_classifier(arch: str, num_classes: int, imagenet_weights: bool = True) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT if imagenet_weights else None
        model = torchvision.models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if arch == "resnet34":
        weights = torchvision.models.ResNet34_Weights.DEFAULT if imagenet_weights else None
        model = torchvision.models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if arch == "resnet50":
        weights = torchvision.models.ResNet50_Weights.DEFAULT if imagenet_weights else None
        model = torchvision.models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if arch in ["vit_b16", "vit_b_16", "vitb16"]:
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT if imagenet_weights else None
        model = torchvision.models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads = nn.Sequential(nn.Linear(in_features, num_classes))
        return model
    raise ValueError(f"Unknown arch: {arch}")


def make_penultimate_extractor(trained_classifier: nn.Module, arch: str) -> nn.Module:
    arch = arch.lower()
    feat_model = copy.deepcopy(trained_classifier)
    if "resnet" in arch:
        feat_model.fc = nn.Identity()
        return feat_model
    if "vit" in arch:
        feat_model.heads = nn.Identity()
        return feat_model
    raise ValueError(f"Unsupported arch for feature extractor: {arch}")


# -------------------------
# Metrics
# -------------------------
def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, tpr_level: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    mask = tpr >= tpr_level
    if not np.any(mask):
        return float("nan")
    return float(np.min(fpr[mask]))


def report_metrics(name: str, id_scores: np.ndarray, ood_scores: np.ndarray) -> None:
    y_true = np.concatenate([np.zeros_like(id_scores, dtype=np.int64),
                             np.ones_like(ood_scores, dtype=np.int64)], axis=0)
    scores = np.concatenate([id_scores, ood_scores], axis=0)

    auroc = roc_auc_score(y_true, scores)
    fpr95 = fpr_at_tpr(y_true, scores, tpr_level=0.95)

    print(f"\n==== {name} ====")
    print(f"AUROC (OOD positive):      {auroc:.4f}")
    print(f"FPR@95%TPR (OOD positive): {fpr95:.4f}")
    print(f"(score stats) ID  mean={id_scores.mean():.6f} std={id_scores.std():.6f}")
    print(f"(score stats) OOD mean={ood_scores.mean():.6f} std={ood_scores.std():.6f}")


# -------------------------
# PCA fitting / scoring
# -------------------------
@torch.no_grad()
def fit_global_pca(
    feats: torch.Tensor,  # (N, D) CPU
    energy_keep: float = 0.95,
    k_max: int = 50,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    feats = feats.float()
    mu = feats.mean(dim=0)
    X = feats - mu

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh.transpose(0, 1)  # (D, r)
    n = X.shape[0]
    lam = (S ** 2) / max(n - 1, 1)  # (r,)

    energy = torch.cumsum(lam, dim=0) / (lam.sum() + eps)
    k = int(torch.searchsorted(energy, torch.tensor(energy_keep)).item()) + 1
    r = V.shape[1]
    k = min(k, k_max, max(r - 1, 1))

    V_perp = V[:, k:].contiguous()
    lam_perp = lam[k:].contiguous()

    if V_perp.numel() == 0:
        V_perp = V[:, -1:].contiguous()
        lam_perp = lam[-1:].contiguous()

    return {"mu": mu.contiguous(), "V_perp": V_perp, "lam_perp": lam_perp, "k": torch.tensor(k)}


@torch.no_grad()
def score_global_pca_ratio(
    feats: torch.Tensor,     # (N, D) CPU
    model: Dict[str, torch.Tensor],
    eps: float = 1e-6,
    split: str = "half",
) -> np.ndarray:
    """
    Global PCA yields ONE residual per sample, so "top-2 ratio" isn't defined across classes.
    We define a reproducible two-part ratio by splitting complement directions:

      - split="half": first half of complement dims vs second half
      - split="even_odd": even vs odd complement indices

    Compute two partial whitened residuals sA, sB and return:
      ratio = min(sA,sB) / (max(sA,sB) + eps)

    This ratio is in (0,1] and is what you feed to AUROC/FPR95.
    """
    feats = feats.float()
    mu = model["mu"]
    Vp = model["V_perp"]          # (D, m)
    lam = model["lam_perp"]       # (m,)

    z = feats - mu.unsqueeze(0)   # (N, D)
    coeff = z @ Vp                # (N, m)

    w = (coeff ** 2) / (lam.unsqueeze(0) + eps)  # (N, m)

    m = w.shape[1]
    if m < 2:
        # cannot form a ratio; return the single residual (still valid, but not ratio)
        s = w.sum(dim=1)
        return s.cpu().numpy().astype(np.float64)

    if split == "half":
        mid = m // 2
        # ensure both parts non-empty
        if mid == 0:
            mid = 1
        sA = w[:, :mid].sum(dim=1)
        sB = w[:, mid:].sum(dim=1)
    elif split == "even_odd":
        sA = w[:, 0::2].sum(dim=1)
        sB = w[:, 1::2].sum(dim=1)
    else:
        raise ValueError(f"unknown split: {split}")

    s1 = torch.minimum(sA, sB)
    s2 = torch.maximum(sA, sB)
    ratio = s1 / (s2 + eps)
    return ratio.cpu().numpy().astype(np.float64)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/sgchr/Documents/Subspaces/data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "vit_b16"])
    parser.add_argument("--ckpt", type=str, default="/home/sgchr/Documents/Subspaces/checkpoints/resnet18_imagenet_finetuned_nodropout_cifar10.pth")
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--energy_keep", type=float, default=0.95)
    parser.add_argument("--k_max", type=int, default=50)
    parser.add_argument("--eps", type=float, default=1e-6)

    parser.add_argument("--ood", type=str, default="svhn", choices=["svhn", "folder"])
    parser.add_argument("--ood_dir", type=str, default="", help="Required if --ood folder")

    parser.add_argument("--fit_transform", type=str, default="aug", choices=["aug", "det"],
                        help="Transform for PCA fit features: aug uses random crop/flip; det uses pretrained eval tfm.")
    parser.add_argument("--ratio_split", type=str, default="half", choices=["half", "even_odd"],
                        help="How to create two partial residual scores for the ratio.")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"Arch={args.arch} | GLOBAL PCA | ood={args.ood} | ratio_split={args.ratio_split}")

    # Transforms
    if args.arch.startswith("resnet"):
        base_tf = torchvision.models.ResNet18_Weights.DEFAULT.transforms()
    else:
        base_tf = torchvision.models.ViT_B_16_Weights.DEFAULT.transforms()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_tfm = base_tf

    fit_tfm = train_tfm if args.fit_transform == "aug" else test_tfm

    # Datasets / loaders
    cifar_train = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=fit_tfm)
    cifar_test  = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_tfm)

    if args.ood == "svhn":
        ood_set = datasets.SVHN(root=args.data_dir, split="test", download=True, transform=test_tfm)
    else:
        if not args.ood_dir:
            raise ValueError("When --ood folder, you must pass --ood_dir /path/to/ImageFolder")
        ood_set = datasets.ImageFolder(args.ood_dir, transform=test_tfm)

    fit_loader  = DataLoader(cifar_train, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(cifar_test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    ood_loader  = DataLoader(ood_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Build classifier + load ckpt
    clf = make_classifier(args.arch, num_classes=args.num_classes, imagenet_weights=True)
    sd = load_state_dict_flexible(args.ckpt, map_location="cpu")
    clf.load_state_dict(sd, strict=True)
    clf = clf.to(device).eval()

    # Feature extractor
    feat_model = make_penultimate_extractor(clf, args.arch).to(device).eval()

    # Extract features
    print("\n==> Extracting features (ID train, fit global PCA)")
    id_train_feats, _ = extract_features(feat_model, fit_loader, device)
    print("ID train feats:", tuple(id_train_feats.shape))

    print("==> Extracting features (ID test)")
    id_test_feats, _ = extract_features(feat_model, test_loader, device)
    print("ID test feats:", tuple(id_test_feats.shape))

    print("==> Extracting features (OOD test)")
    ood_feats, _ = extract_features(feat_model, ood_loader, device)
    print("OOD test feats:", tuple(ood_feats.shape))

    # Fit global PCA
    print("\n==> Fitting GLOBAL PCA (ignore labels)")
    pca_model = fit_global_pca(
        feats=id_train_feats,
        energy_keep=args.energy_keep,
        k_max=args.k_max,
        eps=args.eps,
    )
    print(f"Chosen k = {int(pca_model['k'].item())} | complement dims = {pca_model['V_perp'].shape[1]}")

    # Score using ratio (main)
    print("==> Scoring (GLOBAL PCA ratio)")
    id_scores = score_global_pca_ratio(id_test_feats, pca_model, eps=args.eps, split=args.ratio_split)
    ood_scores = score_global_pca_ratio(ood_feats, pca_model, eps=args.eps, split=args.ratio_split)

    report_metrics("GLOBAL PCA: ratio (two-part complement)", -id_scores, -ood_scores)

    print("\nDone.")


if __name__ == "__main__":
    main()


# ==== GLOBAL PCA: ratio (two-part complement) ====
# AUROC (OOD positive):      0.8966
# FPR@95%TPR (OOD positive): 0.5978
# (score stats) ID  mean=-0.879372 std=0.085711
# (score stats) OOD mean=-0.677725 std=0.122428