#!/usr/bin/env python3
"""
Class-subspace (affine PCA) OOD detection:
ID = CIFAR-10, OOD = SVHN

Steps:
1) Train ImageNet-pretrained ResNet18 with replaced FC on CIFAR-10.
2) Extract penultimate features for:
   - CIFAR-10 train (fit subspaces)
   - CIFAR-10 test (ID eval)
   - SVHN test (OOD eval)
3) Fit per-class affine PCA subspaces using centered features (x - mu_c).
4) Score using whitened residual energy in orthogonal complement:
     s_c(x) = sum_{i>k} (v_i^T (x-mu_c))^2 / (lambda_i + eps)
   OOD score S(x) = min_c s_c(x)
5) Report AUROC and FPR@95%TPR (OOD positive).

Run:
  python subspace_ood_cifar10_svhn.py
"""

import os
import math
import random
import argparse
from typing import Dict, Tuple, List
import copy

import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights,  resnet34, ResNet34_Weights, vit_b_16, ViT_B_16_Weights
from resnet import ResNet18, BasicBlock

from sklearn.metrics import roc_auc_score, roc_curve


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      feats: (N, D) float32 on CPU
      labels: (N,) int64 on CPU
    """
    model.eval()
    all_feats = []
    all_labels = []

    for imgs, labs in loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = model(imgs)  # model should output penultimate features
        all_feats.append(feats.detach().cpu().float())
        all_labels.append(labs.detach().cpu().long())

    feats = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return feats, labels


def make_penultimate_extractor(trained_resnet: nn.Module) -> nn.Module:
    feat_model = copy.deepcopy(trained_resnet)  # already has fc=10 weights loaded
    feat_model.fc = nn.Identity()               # now outputs 512-d penultimate features
    return feat_model


# -------------------------
# Subspace fitting/scoring
# -------------------------
def fit_affine_pca_subspaces(
    feats: torch.Tensor,   # (N, D) CPU
    labels: torch.Tensor,  # (N,) CPU
    num_classes: int,
    energy_keep: float = 0.95,
    k_max: int = 150,
    eps: float = 1e-6,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    For each class c:
      - mu_c: (D,)
      - V_perp: (D, D-k)  orthogonal complement basis
      - lam_perp: (D-k,)  eigenvalues for complement directions

    Uses SVD on centered data matrix X (n x D):
      X = U S V^T
      eigenvalues ~ S^2 / (n-1)
      principal directions are columns of V
    """
    feats = feats.float()
    labels = labels.long()

    D = feats.shape[1]
    models: Dict[int, Dict[str, torch.Tensor]] = {}

    for c in range(num_classes):
        Xc = feats[labels == c]
        if Xc.shape[0] < 5:
            raise RuntimeError(f"Class {c} has too few samples ({Xc.shape[0]}).")

        mu = Xc.mean(dim=0)
        X = Xc - mu  # affine centering

        # SVD on data matrix (n x D). full_matrices=False gives Vh shape (r x D) where r=min(n,D)
        # V = Vh.T shape (D x r)
        # Eigenvalues correspond to variance along V columns.
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        V = Vh.transpose(0, 1)  # (D, r)
        n = X.shape[0]
        # Avoid divide-by-zero if n==1 (won't happen due to check above)
        lam = (S**2) / max(n - 1, 1)  # (r,)

        # Choose k based on cumulative energy
        energy = torch.cumsum(lam, dim=0) / (lam.sum() + eps)
        k = int(torch.searchsorted(energy, torch.tensor(energy_keep)).item()) + 1  # at least 1

        # Cap k and ensure we keep at least 1 complement dim if possible
        r = V.shape[1]
        k = min(k, k_max, max(r - 1, 1))  # ensure k <= r-1 when r>1

        V_perp = V[:, k:]  # (D, r-k)
        lam_perp = lam[k:]  # (r-k,)

        # If rank is very small, complement might be empty; force a tiny complement by backing off k
        if V_perp.numel() == 0:
            k = max(k - 1, 0)
            V_perp = V[:, k:]
            lam_perp = lam[k:]
            if V_perp.numel() == 0:
                # In worst case, fall back to using last direction
                V_perp = V[:, -1:].contiguous()
                lam_perp = lam[-1:].contiguous()

        models[c] = {
            "mu": mu.contiguous(),                 # (D,)
            "V_perp": V_perp.contiguous(),         # (D, m)
            "lam_perp": lam_perp.contiguous(),     # (m,)
        }

    return models


@torch.no_grad()
def score_whitened_residual_min(
    feats: torch.Tensor,  # (N, D) CPU
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      S_min: (N,) min_c s_c(x)  as numpy float64
      ratio: (N,) s1/(s2+eps)  winner vs runner-up residual ratio (optional diagnostic)
    """
    feats = feats.float()
    classes = sorted(subspaces.keys())
    N, D = feats.shape

    # Compute per-class scores (N, C)
    scores_per_class = []
    for c in classes:
        mu = subspaces[c]["mu"]            # (D,)
        Vp = subspaces[c]["V_perp"]        # (D, m)
        lam = subspaces[c]["lam_perp"]     # (m,)

        z = feats - mu.unsqueeze(0)        # (N, D)
        coeff = z @ Vp                     # (N, m)
        # whitened energy in complement
        s = (coeff**2) / (lam.unsqueeze(0) + eps)
        s = s.sum(dim=1)                   # (N,)
        scores_per_class.append(s)

    scores = torch.stack(scores_per_class, dim=1)  # (N, C)
    vals, _ = torch.topk(scores, k=2, dim=1, largest=False)  # smallest two
    s1 = vals[:, 0]
    s2 = vals[:, 1]
    ratio = s1 / (s2 + eps)

    return s1.cpu().numpy().astype(np.float64), ratio.cpu().numpy().astype(np.float64)


def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, tpr_level: float = 0.95) -> float:
    """
    y_true: 1 for OOD, 0 for ID
    scores: higher means "more OOD"
    Returns min FPR among thresholds achieving TPR >= tpr_level.
    """
    fpr, tpr, thr = roc_curve(y_true, scores, pos_label=1)
    mask = tpr >= tpr_level
    if not np.any(mask):
        return float("nan")
    return float(np.min(fpr[mask]))


# -------------------------
# Training
# -------------------------
def train_cifar10(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for ep in range(1, epochs + 1):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for imgs, labs in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labs = labs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, labs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * imgs.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == labs).sum().item())
            total += int(labs.size(0))

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # quick eval
        model.eval()
        t_correct, t_total = 0, 0
        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs = imgs.to(device, non_blocking=True)
                labs = labs.to(device, non_blocking=True)
                logits = model(imgs)
                pred = logits.argmax(dim=1)
                t_correct += int((pred == labs).sum().item())
                t_total += int(labs.size(0))
        test_acc = t_correct / max(t_total, 1)

        print(f"Epoch {ep:02d}/{epochs} | train loss {train_loss:.4f} | train acc {train_acc:.4f} | test acc {test_acc:.4f}")

CKPT_PATH = "/home/sgchr/Documents/Dropout_experiments/resnet18_imagenet_finetuned_nodropout_cifar10.pth"
# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--energy_keep", type=float, default=0.95)
    parser.add_argument("--k_max", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Use official preprocessing for ImageNet pretrained weights
    weights = ResNet18_Weights.DEFAULT
    base_tf = weights.transforms()
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # For train, you can optionally add small augmentation; keep minimal here.
    train_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_tfm = base_tf
    test_tfm_mnist = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        weights.transforms(),
        ])
    # Datasets
    cifar_train = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_tfm)
    cifar_test  = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_tfm)

    # ood_set   = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=test_tfm)
    # path = '/home/sgchr/Documents/Multiple_spaces/Imagenet/ood_data/iNaturalist'
    path = '/home/sgchr/Documents/Multiple_spaces/Imagenet/ood_data/Textures'
    # path = '/home/sgchr/Documents/Multiple_spaces/CIFAR10/ood_data/iSUN'
    # path = '/home/sgchr/Documents/Multiple_spaces/CIFAR10/ood_data/LSUN'
    # path = '/home/sgchr/Documents/Multiple_spaces/Imagenet/ood_data/Places'
    # path = '/home/sgchr/Documents/Dropout_experiments/data/tiny-imagenet-200/val'
    # ood_set   = torchvision.datasets.MNIST(root="./data", download=True, transform=test_tfm_mnist)

    ood_set   = torchvision.datasets.SVHN(root="./data", split="test", download=True, transform=test_tfm)
    # ood_set   = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=test_tfm)
    # ood_set = torchvision.datasets.ImageFolder(path, transform=test_tfm)

    # Loaders
    train_loader = DataLoader(cifar_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(cifar_test,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ood_loader  = DataLoader(ood_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model: pretrained backbone + new FC
    # ckpt_path = "/home/sgchr/Documents/Dropout_experiments/resnet18_imagenet_finetuned_nodropout_cifar10.pth"
    num_classes = 10
    # ckpt_path = '/home/sgchr/Documents/Subspaces/checkpoints/resnet34_cifar10_best.pt'
    ckpt_path= '/home/sgchr/Documents/Subspaces/checkpoints/vit_b16_cifar10_best_finetuning.pt'
    # model = ResNet18(num_class=num_classes)
    # model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    # 2) Build model and REPLACE fc BEFORE loading
    # model = resnet18(weights=None)  # IMPORTANT: weights=None avoids any confusion
    # in_features = model.fc.in_features
    # model.fc = nn.Linear(in_features, num_classes)
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)

    # Replace classifier head for CIFAR 10
    in_features = model.heads.head.in_features  # typically 768
    model.heads = nn.Sequential(
        nn.Linear(in_features, num_classes),
    )

    # # 3) Now load
    # state_dict = torch.load(ckpt_path, map_location="cpu")["model"]    #resnet18
    state_dict = torch.load(ckpt_path, map_location="cpu")  #vit
    # state_dict = torch.load(ckpt_path, map_location="cpu") #`resnet34`
    model.load_state_dict(state_dict)
    model = model.to(device)
    

    # 2) Feature extractor (penultimate features)
    feat_model = make_penultimate_extractor(model).to(device)

    # 3) Extract features
    print("\n==> Extracting features (CIFAR-10 train for fitting subspaces)")
    fit_loader = DataLoader(cifar_train, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    id_train_feats, id_train_labels = extract_features(feat_model, fit_loader, device)
    save_path = "cifar_vit_id_train_feats_labels.pt"
    torch.save(
        {"feats": id_train_feats.cpu(), "labels": id_train_labels.cpu()},
        save_path
    )

    # ckpt = torch.load("cifar_vit_id_train_feats_labels.pt", map_location="cpu")
    # id_train_feats = ckpt["feats"]
    # id_train_labels = ckpt["labels"]


    print("==> Extracting features (CIFAR-10 test for ID eval)")
    id_test_feats, _ = extract_features(feat_model, test_loader, device)
    torch.save(id_test_feats.cpu(), "id_test_vit_feat.pt")
    # torch.load( id_test_feats, 'id_test_vit_feat.pt')
    print("==> Extracting features (SVHN test for OOD eval)")
    ood_feats, _ = extract_features(feat_model, ood_loader, device)
    print(ood_feats.shape)
    # 4) Fit per-class affine PCA subspaces
    print("\n==> Fitting per-class affine PCA subspaces")
    subspaces = fit_affine_pca_subspaces(
        feats=id_train_feats,
        labels=id_train_labels,
        num_classes=num_classes,
        energy_keep=args.energy_keep,
        k_max=args.k_max,
    )

    # 5) Score ID + OOD
    print("\n==> Scoring ID test and OOD test")
    id_scores, id_ratio = score_whitened_residual_min(id_test_feats, subspaces)
    ood_scores, ood_ratio = score_whitened_residual_min(ood_feats, subspaces)

    # Higher score => more OOD
    scores = np.concatenate([id_ratio, ood_ratio], axis=0)
    y_true = np.concatenate([np.zeros_like(id_ratio, dtype=np.int64),
                             np.ones_like(ood_ratio, dtype=np.int64)], axis=0)

    # 6) Metrics
    auroc = roc_auc_score(y_true, scores)
    fpr95 = fpr_at_tpr(y_true, scores, tpr_level=0.95)

    print("\n========== RESULTS ==========")
    print(f"AUROC (ID=CIFAR10 test vs OOD=SVHN test): {auroc:.4f}")
    print(f"FPR@95%TPR (OOD positive):               {fpr95:.4f}")

    # Optional: show ratio behavior (diagnostic)
    print("\n(Extra diagnostic) Residual ratio s1/(s2+eps):")
    print(f"  ID  mean={id_ratio.mean():.4f}, std={id_ratio.std():.4f}")
    print(f"  OOD mean={ood_ratio.mean():.4f}, std={ood_ratio.std():.4f}")
    print("============================\n")


if __name__ == "__main__":
    main()
