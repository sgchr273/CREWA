#!/usr/bin/env python3
"""
Class specific affine PCA OOD detection ablations for scoring:

You can choose ONE scoring mode via CLI:

Complement space (unkept directions):
  1) perp_raw   : sum_{i in perp} (v_i^T (x-mu_c))^2
  2) perp_white : sum_{i in perp} (v_i^T (x-mu_c))^2 / (lam_i + eps)

Kept space (retained directions):
  3) kept_raw   : - max_c sum_{i <= k} (v_i^T (x-mu_c))^2
  4) kept_white : - max_c sum_{i <= k} (v_i^T (x-mu_c))^2 / (lam_i + eps)

Notes:
- For perp_* we use min over classes (closest subspace in residual sense).
- For kept_* higher kept energy means more ID-like, so we negate max energy to form an OOD score.
- Metrics assume: higher score => more OOD (OOD positive).

OOD dataset options:
  --ood svhn | cifar100 | folder
  if folder, pass --ood_dir

Example:
  python affine_pca_ablate.py \
    --arch resnet18 \
    --ckpt /path/to/resnet18_cifar10_ckpt.pt \
    --mode perp_white \
    --ood svhn

"""

import os
import copy
import random
import argparse
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, roc_curve


# -------------------------
# Repro
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    y_true = np.concatenate(
        [np.zeros_like(id_scores, dtype=np.int64), np.ones_like(ood_scores, dtype=np.int64)],
        axis=0,
    )
    scores = np.concatenate([id_scores, ood_scores], axis=0)
    auroc = roc_auc_score(y_true, scores)
    fpr95 = fpr_at_tpr(y_true, scores, tpr_level=0.95)

    print(f"\n==== {name} ====")
    print(f"AUROC (OOD positive):      {auroc:.4f}")
    print(f"FPR@95%TPR (OOD positive): {fpr95:.4f}")
    print(f"(score stats) ID  mean={id_scores.mean():.4f} std={id_scores.std():.4f}")
    print(f"(score stats) OOD mean={ood_scores.mean():.4f} std={ood_scores.std():.4f}")


# -------------------------
# Checkpoint load
# -------------------------
def load_state_dict_flexible(ckpt_path: str, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location=map_location)
    if isinstance(obj, dict):
        for k in ["model_state", "model", "state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # maybe already a raw state_dict
        return obj
    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")


# -------------------------
# Models / features
# -------------------------
def build_resnet18_cifar10_classifier(imagenet_weights: bool = True, num_classes: int = 10) -> nn.Module:
    weights = torchvision.models.ResNet18_Weights.DEFAULT if imagenet_weights else None
    model = torchvision.models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def make_penultimate_extractor(trained_resnet: nn.Module) -> nn.Module:
    feat_model = copy.deepcopy(trained_resnet)
    feat_model.fc = nn.Identity()  # outputs 512-d for resnet18
    return feat_model


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_feats: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    for imgs, labs in loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = model(imgs)
        all_feats.append(feats.detach().cpu().float())
        all_labels.append(labs.detach().cpu().long())
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


# -------------------------
# Class-specific affine PCA fit
# -------------------------
def fit_affine_pca_subspaces(
    feats: torch.Tensor,      # (N, D) CPU
    labels: torch.Tensor,     # (N,) CPU
    num_classes: int,
    energy_keep: float,
    k_max: int,
    eps: float,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    For each class c:
      mu:      (D,)
      Vk:      (D, k)
      lam_k:   (k,)
      Vperp:   (D, m)
      lam_perp:(m,)
      k:       scalar int (stored as tensor)
    """
    feats = feats.float()
    labels = labels.long()
    D = feats.shape[1]

    models: Dict[int, Dict[str, torch.Tensor]] = {}

    for c in range(num_classes):
        Xc = feats[labels == c]
        if Xc.shape[0] < 5:
            raise RuntimeError(f"class {c} has too few samples: {Xc.shape[0]}")

        mu = Xc.mean(dim=0)
        X = Xc - mu

        # SVD: X is (n, D)
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        V = Vh.transpose(0, 1)  # (D, r)
        n = X.shape[0]
        lam = (S ** 2) / max(n - 1, 1)  # (r,)
        r = V.shape[1]

        # choose k by energy, clamp by k_max and r-1 (keep at least 1 perp dim if possible)
        energy = torch.cumsum(lam, dim=0) / (lam.sum() + eps)
        k = int(torch.searchsorted(energy, torch.tensor(energy_keep)).item()) + 1
        k = min(k, k_max, max(r - 1, 1))

        Vk = V[:, :k].contiguous()            # (D, k)
        lam_k = lam[:k].contiguous()          # (k,)
        Vperp = V[:, k:].contiguous()         # (D, m)
        lam_perp = lam[k:].contiguous()       # (m,)

        # if complement empty, force one dim
        if Vperp.numel() == 0:
            Vperp = V[:, -1:].contiguous()
            lam_perp = lam[-1:].contiguous()

        models[c] = {
            "mu": mu.contiguous(),
            "Vk": Vk,
            "lam_k": lam_k,
            "Vperp": Vperp,
            "lam_perp": lam_perp,
            "k": torch.tensor(k, dtype=torch.int64),
        }

    return models


# -------------------------
# Scoring ablations
# -------------------------

@torch.no_grad()
def score_affine_pca_ratio_top2(
    feats: torch.Tensor,  # (N, D) CPU
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    mode: str,
    eps: float,
    invert_for_kept: bool = True,
) -> np.ndarray:
    """
    Computes per-class scores M[n,c] according to `mode`, then returns the ratio
    of the best (smallest) score to the second-best (smallest) score:

        ratio(x) = s1(x) / (s2(x) + eps),

    where s1 <= s2 are the two smallest values across classes.

    Interpretation:
      - For residual-type scores (perp_*), smaller is better (more ID-like).
      - For kept-energy scores (kept_*), larger is better (more ID-like).
        To use the same "smaller is better" convention before taking the ratio,
        we convert kept energy to a cost by negating it (if invert_for_kept=True).

    Returns:
      ratio: (N,) numpy float64
    """
    feats = feats.float()
    classes = sorted(subspaces.keys())
    per_class_vals: List[torch.Tensor] = []

    for c in classes:
        mu = subspaces[c]["mu"]                 # (D,)
        z = feats - mu.unsqueeze(0)             # (N, D)

        if mode == "perp_raw":
            Vp = subspaces[c]["Vperp"]          # (D, m)
            coeff = z @ Vp                      # (N, m)
            s = (coeff ** 2).sum(dim=1)         # (N,)
            per_class_vals.append(s)

        elif mode == "perp_white":
            Vp = subspaces[c]["Vperp"]          # (D, m)
            lam = subspaces[c]["lam_perp"]      # (m,)
            coeff = z @ Vp                      # (N, m)
            s = (coeff ** 2) / (lam.unsqueeze(0) + eps)
            s = s.sum(dim=1)                    # (N,)
            per_class_vals.append(s)

        elif mode == "kept_raw":
            Vk = subspaces[c]["Vk"]             # (D, k)
            coeff = z @ Vk                      # (N, k)
            e = (coeff ** 2).sum(dim=1)         # (N,)
            # convert "larger is better" to "smaller is better" if requested
            per_class_vals.append(-e if invert_for_kept else e)

        elif mode == "kept_white":
            Vk = subspaces[c]["Vk"]             # (D, k)
            lam = subspaces[c]["lam_k"]         # (k,)
            coeff = z @ Vk                      # (N, k)
            e = (coeff ** 2) / (lam.unsqueeze(0) + eps)
            e = e.sum(dim=1)                    # (N,)
            per_class_vals.append(-e if invert_for_kept else e)

        else:
            raise ValueError(f"unknown mode: {mode}")

    M = torch.stack(per_class_vals, dim=1)      # (N, C)

    # Take the two smallest "costs"
    vals, _ = torch.topk(M, k=2, dim=1, largest=False)  # (N, 2)
    s1 = vals[:, 0]
    s2 = vals[:, 1]

    ratio = s1 / (s2 + eps)
    return ratio.cpu().numpy().astype(np.float64)



def maybe_flip_scores(id_scores: np.ndarray, ood_scores: np.ndarray, flip: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    flip:
      - "none": do nothing
      - "flip": multiply by -1
      - "auto": if mean(ood) < mean(id), flip
    """
    if flip == "none":
        return id_scores, ood_scores, "none"
    if flip == "flip":
        return -id_scores, -ood_scores, "flip"
    if flip == "auto":
        if ood_scores.mean() < id_scores.mean():
            return -id_scores, -ood_scores, "auto->flip"
        return id_scores, ood_scores, "auto->none"
    raise ValueError(f"unknown flip option: {flip}")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18"])
    parser.add_argument("--ckpt", type=str, default="/home/sgchr/Documents/Subspaces/checkpoints/resnet18_imagenet_finetuned_nodropout_cifar10.pth", help="Finetuned CIFAR10 classifier checkpoint")
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--energy_keep", type=float, default=0.95)
    parser.add_argument("--k_max", type=int, default=50)
    parser.add_argument("--eps", type=float, default=1e-6)

    parser.add_argument("--mode", type=str, required=True,
                        choices=["perp_raw", "perp_white", "kept_raw", "kept_white"],
                        help="Which ablation scoring mode to run")

    parser.add_argument("--flip", type=str, default="auto", choices=["auto", "none", "flip"],
                        help="Ensure higher score => more OOD. auto flips if OOD mean < ID mean.")

    parser.add_argument("--fit_transform", type=str, default="aug", choices=["det", "aug"],
                        help="Transform used to extract ID train features for fitting PCA. det is stable, aug matches training.")

    parser.add_argument("--ood", type=str, default="svhn", choices=["svhn", "cifar100", "folder"])
    parser.add_argument("--ood_dir", type=str, default="", help="Path for ImageFolder OOD when --ood folder")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"Mode={args.mode} | flip={args.flip} | energy_keep={args.energy_keep} | k_max={args.k_max} | eps={args.eps}")

    # Transforms: ImageNet normalization for ResNet18
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    test_tfm = weights.transforms()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_aug_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    fit_tfm = test_tfm if args.fit_transform == "det" else train_aug_tfm

    # Datasets
    cifar_train = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=fit_tfm)
    cifar_test  = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_tfm)

    if args.ood == "svhn":
        ood_set = datasets.SVHN(root=args.data_dir, split="test", download=True, transform=test_tfm)
    elif args.ood == "cifar100":
        ood_set = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=test_tfm)
    else:
        if not args.ood_dir:
            raise ValueError("For --ood folder, you must set --ood_dir to an ImageFolder path.")
        ood_set = datasets.ImageFolder(args.ood_dir, transform=test_tfm)

    fit_loader  = DataLoader(cifar_train, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    id_loader   = DataLoader(cifar_test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    ood_loader  = DataLoader(ood_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Load classifier
    clf = build_resnet18_cifar10_classifier(imagenet_weights=True, num_classes=args.num_classes)
    sd = load_state_dict_flexible(args.ckpt, map_location="cpu")
    clf.load_state_dict(sd, strict=True)
    clf = clf.to(device).eval()

    # Feature extractor
    feat_model = make_penultimate_extractor(clf).to(device).eval()

    # Extract features
    print("\n==> Extracting features for PCA fit (ID train)")
    id_train_feats, id_train_labels = extract_features(feat_model, fit_loader, device)
    print("ID train feats:", tuple(id_train_feats.shape))

    print("==> Extracting features (ID test)")
    id_test_feats, _ = extract_features(feat_model, id_loader, device)
    print("ID test feats:", tuple(id_test_feats.shape))

    print("==> Extracting features (OOD test)")
    ood_feats, _ = extract_features(feat_model, ood_loader, device)
    print("OOD feats:", tuple(ood_feats.shape))

    # Fit class-specific affine PCA
    print("\n==> Fitting class-specific affine PCA subspaces")
    subspaces = fit_affine_pca_subspaces(
        feats=id_train_feats,
        labels=id_train_labels,
        num_classes=args.num_classes,
        energy_keep=args.energy_keep,
        k_max=args.k_max,
        eps=args.eps,
    )

    # Score
    print("\n==> Scoring")
    id_scores = score_affine_pca_ratio_top2(id_test_feats, subspaces, mode=args.mode, eps=args.eps)
    ood_scores = score_affine_pca_ratio_top2(ood_feats, subspaces, mode=args.mode, eps=args.eps)

    # Ensure higher => more OOD if requested
    id_scores, ood_scores, flip_tag = maybe_flip_scores(id_scores, ood_scores, args.flip)
    print(f"Flip applied: {flip_tag}")

    # Report
    title = f"class-affine PCA | mode={args.mode} | fit_tfm={args.fit_transform} | ood={args.ood}"
    report_metrics(title, id_scores, ood_scores)

    print("\nDone.")


if __name__ == "__main__":
    main()


# ==== class-affine PCA | mode=perp_white | fit_tfm=aug | ood=svhn ====
# AUROC (OOD positive):      0.8904
# FPR@95%TPR (OOD positive): 0.2927
# (score stats) ID  mean=0.6087 std=0.2068
# (score stats) OOD mean=0.9007 std=0.0830


# ==== class-affine PCA | mode=perp_raw | fit_tfm=aug | ood=svhn ====
# AUROC (OOD positive):      0.9103
# FPR@95%TPR (OOD positive): 0.2363
# (score stats) ID  mean=0.6208 std=0.1996
# (score stats) OOD mean=0.9196 std=0.0703


# ==== class-affine PCA | mode=kept_white | fit_tfm=aug | ood=svhn ====
# AUROC (OOD positive):      0.6037
# FPR@95%TPR (OOD positive): 0.9202
# (score stats) ID  mean=1.0872 std=0.0751
# (score stats) OOD mean=1.1113 std=0.0757


# ==== class-affine PCA | mode=kept_raw | fit_tfm=aug | ood=svhn ====
# AUROC (OOD positive):      0.5831
# FPR@95%TPR (OOD positive): 0.8920
# (score stats) ID  mean=1.0796 std=0.0638
# (score stats) OOD mean=1.0919 std=0.0553