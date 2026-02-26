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
from sklearn.decomposition import PCA


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
    run_method_mahalanobis_plus_align)

from new_methods import (
    # get_all_labels,
    # compute_class_means,
    # get_classifier_weight_matrix,
    # infer_pseudo_labels_by_cosine,
    # nc_centered_alignment_scores,
    compute_nc_subtracted_scores
)
plt.rcParams.update({
    "font.family": "DejaVu Serif",
})

label_fontsize = 18
tick_fontsize = 14
legend_fontsize = 18

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

    allowed = {"subspaces", "mahalanobis", "energy", "msp", "logit_gate", "kpca_rff", "gradsubspace", "neco", "vim", "nci", "maha_align"}
    out = []
    for m in methods:
        if m not in allowed:
            raise ValueError(f"Unknown/disabled method: {m}. Allowed: {sorted(list(allowed))}")
        if m not in out:
            out.append(m)
    return out



parser = argparse.ArgumentParser()

parser.add_argument("--id_dataset", type=str, default="cifar10",
                    help="cifar10, cifar100, svhn, mnist, fashionmnist")
parser.add_argument("--ood_dataset", type=str, default="cifar100",
                    help="svhn, cifar100, mnist, fashionmnist, textures, places, tiny-imagenet-200")
parser.add_argument("--ood_dir", type=str, default="/home/sgchr/Documents/Multiple_spaces/Imagenet/ood_data/",
                    help="For textures/places: pass folder path to ImageFolder root")
parser.add_argument("--data_dir", type=str, default="./data")

parser.add_argument("--arch", type=str, default="resnet18",
                    help="resnet18, resnet34, vit_b_16")
parser.add_argument("--ckpt", type=str, default="/home/sgchr/Documents/Subspaces/checkpoints/resnet18_imagenet_finetuned_nodropout_cifar10.pth",
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
# weights = ViT_B_16_Weights.DEFAULT
weights = ResNet18_Weights.DEFAULT
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

methods = [m.lower() for m in args.methods]
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
            tune_beta=False,
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
            # report_metrics("subspaces (s1)", id_scores, ood_scores)
            # plt.figure(figsize=(8, 5))

            # sns.kdeplot(id_scores, label="CIFAR10", fill=True, alpha=0.5, color="green")
            # sns.kdeplot(ood_scores, label="CIFAR100", fill=True, alpha=0.5, color="red")

            # # axis label sizes
            # plt.xlabel("Score", fontsize=16)
            # plt.ylabel("Density", fontsize=16)

            # # tick label sizes (numbers on x and y)
            # plt.xticks(fontsize=14)
            # plt.yticks(fontsize=14)

            # # legend size
            # plt.legend(fontsize=14, title_fontsize=14)

            # plt.tight_layout()
            # plt.savefig("error.png", dpi=300)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------
# Global style (requested)
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Serif",
})
label_fontsize = 18
tick_fontsize = 18
legend_fontsize = 20


def variance_decomposition_diagnostic(
    train_feats,        # (N_train, D)
    train_labels,       # (N_train,)
    id_feats,           # (N_id, D)
    ood_feats,          # (N_ood, D)
    k: int = None,      # if None, uses stable rank heuristic
    stable_rank_ratio: float = 0.5,
    save_path: str = "variance_decomp_diagnostic.png",
    show: bool = True,
    plot: bool = True,
    include_auroc_overlay: bool = True,
    bar_colors=("#4c86ff", "#7fd0ff"),
):
    """
    Variance decomposition diagnostic + corrected SNR plot styling.

    Produces a 1-panel figure:
      - Bar chart: SNR in principal vs complement subspace
      - Optional overlay: AUROC (secondary y-axis)
    """

    train_feats = np.asarray(train_feats)
    train_labels = np.asarray(train_labels)
    id_feats = np.asarray(id_feats)
    ood_feats = np.asarray(ood_feats)

    classes = np.unique(train_labels)
    D = train_feats.shape[1]

    # 1) per-class means + pooled within-class residuals
    class_means = {}
    residuals_list = []
    for c in classes:
        feats_c = train_feats[train_labels == c]
        mu_c = feats_c.mean(axis=0)
        class_means[c] = mu_c
        residuals_list.append(feats_c - mu_c)
    residuals = np.concatenate(residuals_list, axis=0)

    # 2) stable-rank heuristic for k
    cov = (residuals.T @ residuals) / max(1, len(residuals))
    eigvals = np.linalg.eigvalsh(cov)  # ascending
    eigvals_pos = eigvals[eigvals > 0]

    if eigvals_pos.size == 0:
        stable_rank = 1
    else:
        stable_rank = int((eigvals_pos.sum() ** 2) / (np.square(eigvals_pos).sum() + 1e-12))
        stable_rank = max(1, stable_rank)

    if k is None:
        k = max(1, int(stable_rank_ratio * stable_rank))

    # PCA constraint
    D_pca = min(D, residuals.shape[0])
    if k >= D_pca:
        k = max(1, D_pca - 1)

    print(f"[Diagnostic] D={D}, D_pca={D_pca}, stable_rank={stable_rank}, k={k}, complement={D_pca-k}")

    # 3) PCA on pooled residuals
    pca = PCA(n_components=D_pca, svd_solver="full")
    pca.fit(residuals)

    U = pca.components_  # (D_pca, D)
    U_k = U[:k]
    U_perp = U[k:]

    # 4) nearest-centroid subtraction on test features
    centroid_matrix = np.stack([class_means[c] for c in classes], axis=0)  # (C, D)

    def subtract_nearest_centroid(feats: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(feats[:, None, :] - centroid_matrix[None, :, :], axis=-1)
        nearest = dists.argmin(axis=1)
        return feats - centroid_matrix[nearest]

    id_res = subtract_nearest_centroid(id_feats)
    ood_res = subtract_nearest_centroid(ood_feats)

    # 5) energies in principal and complement
    id_par = np.sum((id_res @ U_k.T) ** 2, axis=1)
    id_perp = np.sum((id_res @ U_perp.T) ** 2, axis=1)

    ood_par = np.sum((ood_res @ U_k.T) ** 2, axis=1)
    ood_perp = np.sum((ood_res @ U_perp.T) ** 2, axis=1)

    # 6) SNR + AUROC
    def snr(id_scores: np.ndarray, ood_scores: np.ndarray, eps: float = 1e-8) -> float:
        return float((ood_scores.mean() - id_scores.mean()) / (id_scores.std() + eps))

    snr_par = snr(id_par, ood_par)
    snr_perp = snr(id_perp, ood_perp)

    y_true = np.concatenate([np.zeros(len(id_par)), np.ones(len(ood_par))])
    auroc_par = float(roc_auc_score(y_true, np.concatenate([id_par, ood_par])))
    auroc_perp = float(roc_auc_score(y_true, np.concatenate([id_perp, ood_perp])))

    print(f"  SNR_parallel : {snr_par:.4f}")
    print(f"  SNR_perp     : {snr_perp:.4f}")
    print(f"  AUROC_parallel : {auroc_par:.4f}")
    print(f"  AUROC_perp     : {auroc_perp:.4f}")

    # 7) Corrected plot
    if plot:
        labels = ["Principal\n(high var)", "Complement\n(low var)"]
        x = np.arange(len(labels))
        snr_vals = np.array([snr_par, snr_perp], dtype=float)

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(
            x, snr_vals,
            width=0.5,
            color=list(bar_colors),
            edgecolor="black",
            linewidth=1.2,
        )

        # ax.set_title("Diagnostic: subspace SNR comparison", fontsize=label_fontsize)
        ax.set_ylabel(r"SNR = (E[OOD] − E[ID]) / Std[ID]", fontsize=label_fontsize)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.axhline(0, color="black", linewidth=0.9)
        ax.grid(axis="y", alpha=0.2)

        # Fix headroom so annotations never collide with the frame
        y_min = float(np.min(snr_vals))
        y_max = float(np.max(snr_vals))
        span = (y_max - y_min) if (y_max != y_min) else (abs(y_max) + 1.0)
        pad = 0.18 * span
        ax.set_ylim(min(0.0, y_min - pad), y_max + pad)

        # annotate bar values cleanly
        for bar, val in zip(bars, snr_vals):
            x_txt = bar.get_x() + bar.get_width() / 2
            y_txt = bar.get_height()
            offset = 0.03 * span * (1 if val >= 0 else -1)
            ax.text(
                x_txt,
                y_txt + offset,
                f"{val:.3f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=tick_fontsize,
            )

        ax2 = None
        if include_auroc_overlay:
            ax2 = ax.twinx()
            ax2.plot(
                x, [auroc_par, auroc_perp],
                "D--",
                linewidth=1.8,
                markersize=9,
                label="AUROC",
            )
            ax2.set_ylabel("AUROC", fontsize=label_fontsize)
            ax2.tick_params(axis="y", labelsize=tick_fontsize)
            ax2.set_ylim(0.4, 1.05)
            ax2.legend(loc="lower right", fontsize=legend_fontsize, frameon=True)

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    return {
        "k": int(k),
        "stable_rank": int(stable_rank),
        "snr_parallel": float(snr_par),
        "snr_perp": float(snr_perp),
        "auroc_parallel": float(auroc_par),
        "auroc_perp": float(auroc_perp),
        "id_energy_parallel": id_par,
        "id_energy_perp": id_perp,
        "ood_energy_parallel": ood_par,
        "ood_energy_perp": ood_perp,
    }


# Example:
results = variance_decomposition_diagnostic(
    train_feats, train_labels,
    id_feats, ood_feats,
    k=None,
    stable_rank_ratio=0.3,
    save_path=f"diag_snr_{args.arch}_{args.ood_dataset}.png",
)


# def plot_cumulative_delta(delta, k, save_path="diag1_cumulative_delta.png"):
#     """
#     Plots cumulative average of delta_i for:
#       - principal subspace (top-k directions, left to right)
#       - complement subspace (directions k+1..D, left to right)
    
#     delta: (D,) array of per-eigenvector separations, descending eigenvalue order
#     k:     boundary between principal and complement
#     """
#     D = len(delta)

#     delta_par  = delta[:k]       # top-k (high variance)
#     delta_perp = delta[k:]       # remaining (low variance)

#     cum_avg_par  = np.cumsum(delta_par)  / np.arange(1, k + 1)
#     cum_avg_perp = np.cumsum(delta_perp) / np.arange(1, D - k + 1)

#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#     fig.suptitle(r"Cumulative Average of $\delta_i$ — Principal vs Complement Subspace",
#                  fontsize=13, fontweight='bold')

#     # ── Panel A: side-by-side on shared y-axis ──────────────────────────
#     ax = axes[0]
#     ax.plot(np.arange(1, k + 1),       cum_avg_par,
#             color='steelblue', linewidth=2, label=f'Principal (top-{k})')
#     ax.plot(np.arange(k + 1, D + 1),   cum_avg_perp,
#             color='tomato',    linewidth=2, label=f'Complement (bot-{D-k})')
#     ax.axvline(k, color='gray', linestyle='--', linewidth=1.2, label=f'Boundary k={k}')

#     # annotate final values
#     ax.annotate(f"Final: {cum_avg_par[-1]:.3f}",
#                 xy=(k, cum_avg_par[-1]),
#                 xytext=(k - k*0.4, cum_avg_par[-1] + 0.02),
#                 arrowprops=dict(arrowstyle='->', color='steelblue'),
#                 color='steelblue', fontsize=9)
#     ax.annotate(f"Final: {cum_avg_perp[-1]:.3f}",
#                 xy=(D, cum_avg_perp[-1]),
#                 xytext=(D - (D-k)*0.35, cum_avg_perp[-1] + 0.02),
#                 arrowprops=dict(arrowstyle='->', color='tomato'),
#                 color='tomato', fontsize=9)

#     ax.set_xlabel("Eigenvector rank (1 = highest variance)")
#     ax.set_ylabel(r"Cumulative average $\delta_i$")
#     ax.set_title("Full spectrum view")
#     ax.legend(fontsize=9)
#     ax.grid(True, alpha=0.3)

#     # ── Panel B: each subspace on its own x-axis (0→1 normalised) ───────
#     ax = axes[1]
#     x_par  = np.linspace(0, 1, k)
#     x_perp = np.linspace(0, 1, D - k)

#     ax.plot(x_par,  cum_avg_par,  color='steelblue', linewidth=2,
#             label=f'Principal (top-{k})')
#     ax.plot(x_perp, cum_avg_perp, color='tomato',    linewidth=2,
#             label=f'Complement (bot-{D-k})')

#     # shade the gap between them to emphasise the difference
#     # interpolate onto a common grid for shading
#     x_common   = np.linspace(0, 1, 500)
#     par_interp  = np.interp(x_common, x_par,  cum_avg_par)
#     perp_interp = np.interp(x_common, x_perp, cum_avg_perp)
#     ax.fill_between(x_common, par_interp, perp_interp,
#                     where=(perp_interp > par_interp),
#                     alpha=0.15, color='tomato',    label='Complement advantage')
#     ax.fill_between(x_common, par_interp, perp_interp,
#                     where=(perp_interp <= par_interp),
#                     alpha=0.15, color='steelblue', label='Principal advantage')

#     ax.set_xlabel("Relative position within subspace (0 → 1)")
#     ax.set_ylabel(r"Cumulative average $\delta_i$")
#     ax.set_title("Normalised x-axis (subspace-relative)")
#     ax.legend(fontsize=9)
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     print(f"Saved → {save_path}")
#     plt.show()

# plot_cumulative_delta(
#     delta     = results["delta_per_eigenvector"],
#     k         = results["k"],
#     save_path = f"diag1_cumdelta_{args.arch}_{args.ood_dataset}.png",
# )