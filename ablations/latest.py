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
    # scores_energy_from_logits,
    # scores_msp_from_logits,
    # fit_global_pca_basis,   
    fit_affine_pca_subspaces,
    # fit_id_feature_subspace,
    # fit_kpca_rff,
    # scores_logit_gate,
    # scores_kpca_rff,
    # scores_gradsubspace_pseudo_resid,
    # run_method_subspaces,
    # run_method_subspaces_simple,
    # run_method_mahalanobis,
    # run_method_neco,
    # run_method_vim,
    # compute_mu_global,
    # extract_gaussian_feats_like_id,
    # nci_scores_batched,
    # run_method_mahalanobis_plus_align
)

from new_methods import (
    # get_all_labels,
    # compute_class_means,
    # get_classifier_weight_matrix,
    # infer_pseudo_labels_by_cosine,
    # nc_centered_alignment_scores,
    # compute_nc_subtracted_scores,
    score_subspaces_resid_plus_align_simple,
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


def report_metrics(id_scores: np.ndarray, ood_scores: np.ndarray):
    scores = np.concatenate([id_scores, ood_scores], axis=0)
    y_true = np.concatenate(
        [np.zeros_like(id_scores, dtype=np.int64),
         np.ones_like(ood_scores, dtype=np.int64)],
        axis=0,
    )

    auroc = roc_auc_score(y_true, scores)
    fpr95 = fpr_at_tpr(y_true, scores, tpr_level=0.95)

    print(f"AUROC (OOD positive):      {auroc:.4f}")
    print(f"FPR@95%TPR (OOD positive): {fpr95:.4f}")
    print(f"(score means) ID  mean={id_scores.mean():.6f} std={id_scores.std():.6f}")
    print(f"(score means) OOD mean={ood_scores.mean():.6f} std={ood_scores.std():.6f}")
    print("======================================")

    return {"auroc": float(auroc), "fpr95": float(fpr95)}




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


parser = argparse.ArgumentParser()

parser.add_argument("--id_dataset", type=str, default="cifar10",
                    help="cifar10, cifar100, svhn, mnist, fashionmnist")
parser.add_argument("--ood_dataset", type=str, default="tiny-imagenet-200",
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

def ablation_alpha(
    train_feats, train_labels, id_feats, id_logits, ood_feats, ood_logits,
    num_classes, W, eps=1e-6
):
    # fit SVD once, then sweep alpha without refitting
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

    results = {}
    for alpha in [0.1, 0.2 ,0.3, 0.4, 0.5, 0.6, 0.7]:
        k = max(1, min(int(alpha * stable_r), r - 64))
        V_perp = V[:, k:].contiguous()
        V_keep = V[:, :k].contiguous()
        lam_perp = lam[k:].contiguous()
        subspaces = {0: {"mu_c": mu_c.contiguous(), "V_keep": V_keep,
                         "V_perp": V_perp, "lam_perp": lam_perp,
                         "counts": torch.zeros(num_classes)}}
        s_id, _ = score_subspaces_resid_plus_align_simple(
            id_feats, id_logits, W, subspaces, beta=0.0, tune_beta=False, eps=eps)
        s_ood, _ = score_subspaces_resid_plus_align_simple(
            ood_feats, ood_logits, W, subspaces, beta=0.0, tune_beta=False, eps=eps)
        results[alpha] = {"k": k, "perp_dim": r - k, "s_id": s_id, "s_ood": s_ood}
        print(f"alpha={alpha} | k={k} | perp_dim={r-k}")
    return results

def ablation_score_components(
    train_feats, train_labels, id_feats, id_logits, ood_feats, ood_logits,
    num_classes, W, eps=1e-6
):
    subspaces = fit_affine_pca_subspaces(
        train_feats, train_labels, num_classes, eps=eps)

    m = subspaces[0]
    mu_c = m["mu_c"]
    Vp   = m["V_perp"]
    Vk   = m["V_keep"]

    results = {}
    for split in ["id", "ood"]:
        feats  = id_feats  if split == "id" else ood_feats
        logits = id_logits if split == "id" else ood_logits
        feats  = feats.float()
        logits = logits.float()
        Wf     = W.float()

        pred     = torch.argmax(feats @ Wf.T, dim=1).long()
        z        = feats - mu_c[pred]

        # s_res
        coeff_out = z @ Vp
        s_res     = (coeff_out ** 2).sum(dim=1)

        # s_in and s_ratio
        coeff_in  = z @ Vk
        s_in      = (coeff_in ** 2).sum(dim=1)
        s_ratio   = s_res / (s_in + eps)

        # alignment a
        u_n  = F.normalize(feats, p=2, dim=1, eps=eps)
        W_n  = F.normalize(Wf,    p=2, dim=1, eps=eps)
        cos  = (u_n * W_n[pred]).sum(dim=1).clamp(-1.0, 1.0)
        a    = 1.0 - cos

        # calibrate beta on ID for combined variants
        med_res   = float(np.median(s_res.cpu().numpy()))
        med_a     = float(np.median(a.cpu().numpy()))
        med_ratio = float(np.median(s_ratio.cpu().numpy()))
        beta_a    = 0.0 if med_a     < 1e-12 else med_res / med_a
        beta_r    = 0.0 if med_ratio < 1e-12 else med_res / med_ratio

        to_np = lambda t: t.detach().cpu().numpy().astype(np.float64)
        results[split] = {
            "s_res":           to_np(s_res),
            "a_only":          to_np(a),
            "s_ratio":         to_np(s_ratio),
            "res_plus_align":  to_np(s_res + beta_a * a),
            "res_plus_ratio":  to_np(s_res + beta_r * s_ratio),
            "full":            to_np(s_res + beta_r * s_ratio + beta_a * a),
        }
    return results


def ablation_beta_calibration(
    train_feats, train_labels, id_feats, id_logits, ood_feats, ood_logits,
    num_classes, W, eps=1e-6
):
    subspaces = fit_affine_pca_subspaces(
        train_feats, train_labels, num_classes, eps=eps)

    # get raw s_res and a on ID with beta=0
    s_res_id, a_id = score_subspaces_resid_plus_align_simple(
        id_feats, id_logits, W, subspaces,
        beta=0.0, tune_beta=False, eps=eps)

    # compute beta variants from ID only
    med_res  = float(np.median(s_res_id))
    med_a    = float(np.median(a_id))
    std_res  = float(np.std(s_res_id))  + 1e-12
    std_a    = float(np.std(a_id))      + 1e-12

    beta_median_ratio = 0.0 if med_a  < 1e-12 else med_res / med_a
    beta_std_ratio    = std_res / std_a

    results = {}
    for name, beta_val in [
        ("beta_0_no_align",    0.0),
        ("beta_1_fixed",       1.0),
        ("beta_median_ratio",  beta_median_ratio),
        ("beta_std_ratio",     beta_std_ratio),
    ]:
        s_id, _  = score_subspaces_resid_plus_align_simple(
            id_feats,  id_logits,  W, subspaces,
            beta=beta_val, tune_beta=False, eps=eps)
        s_ood, _ = score_subspaces_resid_plus_align_simple(
            ood_feats, ood_logits, W, subspaces,
            beta=beta_val, tune_beta=False, eps=eps)
        results[name] = {"beta": beta_val, "s_id": s_id, "s_ood": s_ood}
        print(f"{name}: beta={beta_val:.4f}")
    return results


def ablation_subspace_direction(
    train_feats, train_labels, id_feats, id_logits, ood_feats, ood_logits,
    num_classes, W, eps=1e-6
):
    subspaces = fit_affine_pca_subspaces(
        train_feats, train_labels, num_classes, eps=eps)

    m    = subspaces[0]
    mu_c = m["mu_c"]
    Vp   = m["V_perp"]   # discarded
    Vk   = m["V_keep"]   # kept
    Wf   = W.float()

    results = {}
    for split in ["id", "ood"]:
        feats  = (id_feats  if split == "id" else ood_feats).float()
        logits = (id_logits if split == "id" else ood_logits).float()

        pred = torch.argmax(feats @ Wf.T, dim=1).long()
        z    = feats - mu_c[pred]

        s_discarded = ((z @ Vp) ** 2).sum(dim=1)   # your current score
        s_kept      = ((z @ Vk) ** 2).sum(dim=1)   # in-subspace energy

        to_np = lambda t: t.detach().cpu().numpy().astype(np.float64)
        results[split] = {
            "discarded_subspace": to_np(s_discarded),
            "kept_subspace":      to_np(s_kept),
        }
    return results


def run_all_ablations(
    train_feats, train_labels, id_feats, id_logits, ood_feats, ood_logits,
    num_classes, W, report_metrics, eps=1e-6
):
    """
    report_metrics(s_id, s_ood) -> {"auroc": float, "fpr95": float}
    """

    def report(name, s_id, s_ood):
        metrics = report_metrics(s_id, s_ood)
        print(f"  [{name}] AUROC={metrics['auroc']:.4f} | FPR95={metrics['fpr95']:.4f}")
        return metrics

    shared = dict(
        train_feats=train_feats, train_labels=train_labels,
        id_feats=id_feats, id_logits=id_logits,
        ood_feats=ood_feats, ood_logits=ood_logits,
        num_classes=num_classes, W=W, eps=eps,
    )

    all_results = {}

    # ------------------------------------------------------------------ #
    # A) Stable rank alpha sweep
    # ------------------------------------------------------------------ #
    print("\n===== A) Stable Rank Alpha Sweep =====")
    res_A = ablation_alpha(**shared)
    all_results["alpha_sweep"] = {}
    for alpha, val in res_A.items():
        all_results["alpha_sweep"][alpha] = report(
            f"alpha={alpha} k={val['k']} perp={val['perp_dim']}",
            val["s_id"], val["s_ood"]
        )

    # ------------------------------------------------------------------ #
    # B) Score component contribution
    # ------------------------------------------------------------------ #
    print("\n===== B) Score Component Contribution =====")
    res_B = ablation_score_components(**shared)
    all_results["score_components"] = {}
    for variant in ["s_res", "a_only", "s_ratio",
                    "res_plus_align", "res_plus_ratio", "full"]:
        all_results["score_components"][variant] = report(
            variant,
            res_B["id"][variant],
            res_B["ood"][variant],
        )

    # ------------------------------------------------------------------ #
    # C) Beta calibration strategy
    # ------------------------------------------------------------------ #
    print("\n===== C) Beta Calibration Strategy =====")
    res_C = ablation_beta_calibration(**shared)
    all_results["beta_calibration"] = {}
    for name, val in res_C.items():
        all_results["beta_calibration"][name] = report(
            f"{name} (beta={val['beta']:.4f})",
            val["s_id"], val["s_ood"]
        )

    # ------------------------------------------------------------------ #
    # D) Centering strategy
    # ------------------------------------------------------------------ #
    # print("\n===== D) Centering Strategy =====")
    # res_D = ablation_centering(**shared)
    # all_results["centering"] = {}
    # for variant in ["pred_class_centering", "nearest_class_centering"]:
    #     all_results["centering"][variant] = report(
    #         variant,
    #         res_D["id"][variant],
    #         res_D["ood"][variant],
    #     )

    # ------------------------------------------------------------------ #
    # E) Kept vs discarded subspace
    # ------------------------------------------------------------------ #
    print("\n===== E) Kept vs Discarded Subspace =====")
    res_E = ablation_subspace_direction(**shared)
    all_results["subspace_direction"] = {}
    for variant in ["discarded_subspace", "kept_subspace"]:
        all_results["subspace_direction"][variant] = report(
            variant,
            res_E["id"][variant],
            res_E["ood"][variant],
        )

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    print("\n===== SUMMARY =====")
    for ablation, variants in all_results.items():
        print(f"\n{ablation}")
        print(f"  {'Variant':<40} {'AUROC':>8} {'FPR95':>8}")
        print(f"  {'-'*56}")
        for name, metrics in variants.items():
            print(f"  {str(name):<40} {metrics['auroc']:>8.4f} {metrics['fpr95']:>8.4f}")

    return all_results


logits_id  = logits_from_feats(id_feats,  W, b)
logits_ood = logits_from_feats(ood_feats, W, b)

# id_scores, ood_scores = run_method_subspaces_simple(
#     train_feats=train_feats,
#     train_labels=train_labels,
#     id_feats=id_feats,
#     id_logits=logits_id,
#     ood_feats=ood_feats,
#     ood_logits=logits_ood,
#     num_classes=num_classes,
#     energy_keep=args.energy_keep,
#     k_max=0,
#     W=W, 
#     beta=0,
#     tune_beta=True,
#     gamma=1,
#     use_centered_for_cos=True,
# )

all_results = run_all_ablations(
    train_feats=train_feats,
    train_labels=train_labels,
    id_feats=id_feats,
    id_logits=logits_id,
    ood_feats=ood_feats,
    ood_logits=logits_ood,
    num_classes=num_classes,
    W=W,
    report_metrics=report_metrics,  # your existing function
)




   

