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
import seaborn as sns
import matplotlib.pyplot as plt

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
    run_deca,
    scores_energy_from_logits,
    scores_msp_from_logits,
    fit_global_pca_basis,   
    fit_affine_pca_subspaces,
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
    run_deca,)

from new_methods import (
    # get_all_labels,
    # compute_class_means,
    # get_classifier_weight_matrix,
    # infer_pseudo_labels_by_cosine,
    # nc_centered_alignment_scores,
    compute_nc_subtracted_scores,
    neco_mahalanobis,

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


    if len(methods) == 0:
        raise RuntimeError("No methods selected. Use --methods ... or flags.")

    allowed = {"subspaces", "mahalanobis", "energy", "msp", "logit_gate", "kpca_rff", "gradsubspace", "neco", "vim", "nci", "deca"}
    out = []
    for m in methods:
        if m not in allowed:
            raise ValueError(f"Unknown/disabled method: {m}. Allowed: {sorted(list(allowed))}")
        if m not in out:
            out.append(m)
    return out


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--id_dataset", type=str, default="cifar10",
                        help="cifar10, cifar100, svhn, mnist, fashionmnist")
    parser.add_argument("--ood_dataset", type=str, default="tiny-imagenet-200",
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
    # parser.add_argument("--methods", nargs="+", default=None,
    #                     help="List: subspaces mahalanobis energy msp logit_gate kpca_rff gradsubspace neco")
    parser.add_argument(
    "--methods", nargs="+", default=None,
    choices=["subspaces","mahalanobis","energy","msp","logit_gate","kpca_rff","gradsubspace","neco","vim","nci","deca"],
)
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
    parser.add_argument("--deca", action="store_true")

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



    # head params for logits-based methods
    W, b = get_classifier_linear(clf_model, args.arch)
    def compute_metrics(id_scores: np.ndarray, ood_scores: np.ndarray):
        scores = np.concatenate([id_scores, ood_scores], axis=0)
        y_true = np.concatenate(
            [np.zeros_like(id_scores, dtype=np.int64),
            np.ones_like(ood_scores, dtype=np.int64)],
            axis=0,
        )
        auroc = float(roc_auc_score(y_true, scores))
        fpr95 = float(fpr_at_tpr(y_true, scores, tpr_level=0.95))
        return auroc, fpr95

    # methods = [m.lower() for m in args.methods]
    for m in methods:
    # run methods
        if m == "subspaces":
            logits_id  = logits_from_feats(id_feats,  W, b)
            logits_ood = logits_from_feats(ood_feats, W, b)

            id_scores, ood_scores = run_method_subspaces_simple(
                train_feats=train_feats,
                train_labels=train_labels,
                id_feats=id_feats,
                id_logits=logits_id,
                ood_feats=ood_feats,
                ood_logits=logits_ood,
                num_classes=num_classes,
                energy_keep=args.energy_keep,
                k_max=0,
                W=W, 
                beta=1,
                tune_beta=True,
                gamma=1,
                use_centered_for_cos=True,
            )

            # id_scores, ood_scores = run_method_subspaces(
            #     train_feats=train_feats,
            #     train_labels=train_labels,
            #     id_feats=id_feats,
            #     id_logits=logits_id,
            #     ood_feats=ood_feats,
            #     ood_logits=logits_ood,
            #     W=W, b=b,
            #     num_classes=num_classes,
            #     energy_keep=args.energy_keep,
            #     k_max=0,
            #     # beta=1,
            #     tune_beta=True,
            #     tune_alpha=True,
            #     # alpha=1,
            #     delta=1,
            #     gamma=1,
            #     use_centered_for_cos=True,
            # )
            report_metrics("subspaces (s1)", id_scores, ood_scores)
            # plt.figure(figsize=(8, 6))
            # sns.kdeplot(id_scores, label="CIFAR10", fill=True, alpha=0.5, color="green")
            # sns.kdeplot(ood_scores, label="CIFAR100", fill=True, alpha=0.5, color="red")

            # title_fs  = 18
            # label_fs  = 18   # x label / y label
            # tick_fs   = 18   # x ticks / y ticks
            # legend_fs = 18

            # # plt.title("KDE Plot of ID vs OOD", fontsize=title_fs)
            # plt.xlabel("Score", fontsize=label_fs)
            # plt.ylabel("Density", fontsize=label_fs)

            # plt.xticks(fontsize=tick_fs)
            # plt.yticks(fontsize=tick_fs)

            # plt.legend(fontsize=legend_fs)  # or: plt.legend(prop={"size": legend_fs})
            # plt.grid(False)  # your plt.grid('') is equivalent to "on"; this turns it off
            # plt.savefig('error')

        elif m == "mahalanobis":
            # id_scores, ood_scores = run_method_mahalanobis(
            #     train_feats=train_feats,
            #     train_labels=train_labels,
            #     id_feats=id_feats,
            #     ood_feats=ood_feats,
            #     num_classes=num_classes,
            # )
            id_scores, ood_scores = neco_mahalanobis(
                train_feats=train_feats,
                train_labels=train_labels,
                id_feats=id_feats,
                ood_feats=ood_feats,
                num_classes=num_classes,
            )
            report_metrics("mahalanobis (min dist)", id_scores, ood_scores)

        elif m == "energy":
            logits_id  = logits_from_feats(id_feats,  W, b)
            logits_ood = logits_from_feats(ood_feats, W, b)
            id_scores  = scores_energy_from_logits(logits_id,  T=args.energy_T)
            ood_scores = scores_energy_from_logits(logits_ood, T=args.energy_T)
            report_metrics(f"energy (T={args.energy_T})", id_scores, ood_scores)

        elif m == "msp":
            logits_id  = logits_from_feats(id_feats,  W, b)
            logits_ood = logits_from_feats(ood_feats, W, b)
            id_scores  = scores_msp_from_logits(logits_id)
            ood_scores = scores_msp_from_logits(logits_ood)
            report_metrics("msp (-max softmax)", id_scores, ood_scores)

        elif m == "logit_gate":
            feat_mean, Vk = fit_global_pca_basis(
                train_feats,
                energy_keep=args.gate_energy_keep,
                k_max=args.gate_k_max,
                eps=args.gate_eps,
            )
            id_scores = scores_logit_gate(
                id_feats, W, b, feat_mean, Vk,
                threshold=args.gate_threshold,
                eps=args.gate_eps,
            )
            ood_scores = scores_logit_gate(
                ood_feats, W, b, feat_mean, Vk,
                threshold=args.gate_threshold,
                eps=args.gate_eps,
            )
            report_metrics(
                f"logit_gate (thr={args.gate_threshold}, ek={args.gate_energy_keep}, kmax={args.gate_k_max})",
                id_scores, ood_scores
            )

        elif m == "kpca_rff":
            mu_map, Uq, q = fit_kpca_rff(
                train_feats=train_feats,
                gamma=args.kpca_gamma,
                M=args.kpca_M,
                exp_var_ratio=args.kpca_exp_var_ratio,
                seed=args.seed,
            )
            id_scores = scores_kpca_rff(
                query_feats=id_feats,
                mu=mu_map,
                Uq=Uq,
                gamma=args.kpca_gamma,
                M=args.kpca_M,
                seed=args.seed,
            )
            ood_scores = scores_kpca_rff(
                query_feats=ood_feats,
                mu=mu_map,
                Uq=Uq,
                gamma=args.kpca_gamma,
                M=args.kpca_M,
                seed=args.seed,
            )
            report_metrics(
                f"kpca_rff (gamma={args.kpca_gamma}, M={args.kpca_M}, q={q}, evr={args.kpca_exp_var_ratio})",
                id_scores, ood_scores
            )

        elif m == "gradsubspace":
            S, mu, k = fit_id_feature_subspace(
                train_feats=train_feats,
                n_batch=args.grad_n,
                exp_var_ratio=args.grad_exp_var_ratio,
                center=args.grad_center,
                seed=args.seed,
                eps=args.grad_eps,
            )
            id_scores = scores_gradsubspace_pseudo_resid(
                feats=id_feats, W=W, b=b, S=S, mu=mu,
                num_classes=num_classes, eps=args.grad_eps,
            )
            ood_scores = scores_gradsubspace_pseudo_resid(
                feats=ood_feats, W=W, b=b, S=S, mu=mu,
                num_classes=num_classes, eps=args.grad_eps,
            )
            report_metrics(
                f"gradsubspace (n={min(args.grad_n, train_feats.shape[0])}, k={k}, evr={args.grad_exp_var_ratio}, center={args.grad_center})",
                id_scores, ood_scores
            )
            

        elif m == "neco":
            id_scores, ood_scores = run_method_neco(
                train_feats=train_feats,
                id_feats=id_feats,
                ood_feats=ood_feats,
                W=W, b=b, arch=args.arch,
                neco_dim=args.neco_dim,
            )
            report_metrics("neco", -id_scores, -ood_scores)
            

        elif m == "vim":
            s_id, s_ood = run_method_vim(
                train_feats=train_feats,
                id_feats=id_feats,
                ood_feats=ood_feats,
                W=W, b=b,
                num_classes=num_classes,
                vim_dim=getattr(args, "vim_dim", 0),
                vim_fit_max=getattr(args, "vim_fit_max", 200_000),
                seed=args.seed,
                fit_device=torch.device("cpu"),
                score_device=device,
            )
            report_metrics("vim", s_id, s_ood)
         

        elif m == "deca":
            s_id, s_ood = run_deca(train_feats, id_feats, ood_feats)
            report_metrics("deca", s_id, s_ood)
            # return s_id, s_ood

        elif m == "nci":
            # ------------------------------------------------------------
            # NCI (paper-faithful): tune varpi on ID-val vs Gaussian noise,
            # then evaluate on ID-test vs real OOD.
            # Assumptions:
            # - feat_model outputs penultimate features h (do NOT L2-normalize for NCI)
            # - W, b are the classifier head weights/bias aligned with those features
            # ------------------------------------------------------------



            # -----------------------------
            # 1) Build ID validation loader (CIFAR-10 train subsample) for tuning varpi
            # -----------------------------
            ds_full = datasets.CIFAR10(
                root=args.data_dir,
                train=True,
                download=True,
                transform=test_tfm,  # deterministic preprocessing
            )
            n_full = len(ds_full)
            val_size = int(min(5000, n_full))

            rng = np.random.RandomState(args.seed)
            idx = rng.choice(n_full, size=val_size, replace=False).tolist()
            id_val_set = Subset(ds_full, idx)

            id_val_loader = DataLoader(
                id_val_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            # -----------------------------
            # 2) Extract validation features (real ID val)
            # -----------------------------
            feat_model.eval()
            outs = []
            with torch.no_grad():
                for x, _ in id_val_loader:
                    x = x.to(device, non_blocking=True)
                    outs.append(feat_model(x).detach().cpu())
            id_val_feats = torch.cat(outs, dim=0).float()

            # -----------------------------
            # 3) Extract Gaussian validation features (Gaussian noise passed through model)
            #     Uses your helper: extract_gaussian_feats_like_id(feat_model, loader, device)
            # -----------------------------
            gauss_val_feats = extract_gaussian_feats_like_id(feat_model, id_val_loader, device)

            # -----------------------------
            # 4) Compute global mean mu_G from ID train features
            #     Uses your helper: compute_mu_global(train_feats)
            # -----------------------------
            mu_g = compute_mu_global(train_feats)  # (D,)

            # -----------------------------
            # 5) Tune varpi on (ID val vs Gaussian val)
            #     Uses your helper: nci_scores_batched(feats, W, b, mu_g, alpha=varpi, p_norm=1)
            # -----------------------------
            varpi_grid = [1e-4, 1e-3, 1e-2, 1e-1]
            best_varpi, best_auc = None, -1.0

            for v in varpi_grid:
                _, s_id_val = nci_scores_batched(id_val_feats, W, b, mu_g, alpha=v, p_norm=1)
                _, s_gau_val = nci_scores_batched(gauss_val_feats, W, b, mu_g, alpha=v, p_norm=1)

                # Prefer your report_metrics return value if it returns AUROC for "NCI"
                auc = report_metrics("NCI", s_id_val, s_gau_val)

                # Safety fallback if report_metrics ever stops returning AUROC
                if auc is None:
                    scores = np.concatenate([s_id_val, s_gau_val], axis=0)
                    y_true = np.concatenate(
                        [np.zeros_like(s_id_val, dtype=np.int64),
                        np.ones_like(s_gau_val, dtype=np.int64)],
                        axis=0
                    )
                    auc = float(roc_auc_score(y_true, scores))

                if auc > best_auc:
                    best_auc, best_varpi = float(auc), float(v)

            print("Chosen varpi:", best_varpi, "val AUROC(ID vs Gaussian):", best_auc)

            # -----------------------------
            # 6) Final evaluation on (ID test vs real OOD test) using chosen varpi
            # -----------------------------
            _, s_id_test = nci_scores_batched(id_feats, W, b, mu_g, alpha=best_varpi, p_norm=1)
            _, s_ood_test = nci_scores_batched(ood_feats, W, b, mu_g, alpha=best_varpi, p_norm=1)

            report_metrics("NCI", s_id_test, s_ood_test)
            return s_id_test, s_ood_test





if __name__ == "__main__":
    main()

