#!/usr/bin/env python3
"""
run.py — Unified OOD detection benchmark runner.

Works for all supported ID datasets (CIFAR-10/100, SVHN, MNIST, FashionMNIST,
ImageNet-1K) and any OOD dataset reachable via an ImageFolder path.

Quick examples
--------------
# CIFAR-10 (ViT-B/16, custom checkpoint) vs Tiny-ImageNet-200
python run.py \\
    --id_dataset cifar10 \\
    --arch vit_b_16 --ckpt /path/to/vit_cifar10.pt \\
    --ood_dir /data/ood --ood_dataset tiny-imagenet-200 \\
    --methods subspaces energy msp

# ImageNet-1K (pretrained ResNet-50) vs Places365
python run.py \\
    --id_dataset imagenet1k \\
    --imagenet_root /data/ILSVRC/Data/CLS-LOC \\
    --arch resnet50 \\
    --ood_dir /data/places365 --ood_dataset places \\
    --methods subspaces mahalanobis vim \\
    --batch_size 512 --num_workers 8
"""

import argparse
import sys

import numpy as np
import torch

from utils    import set_seed
from models   import build_model, get_classifier_Wb
from datasets import (
    build_id_dataset, build_ood_dataset,
    make_loader, num_classes_from_dataset,
)
from features import make_cache_paths, extract_all_features
from evaluate import parse_methods, run_method, ALLOWED_METHODS


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Unified OOD detection benchmark.",
    )

    # ── Data ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Data")
    g.add_argument("--id_dataset", type=str, default="cifar10",
                   help="ID dataset: cifar10 | cifar100 | svhn | mnist | "
                        "fashionmnist | imagenet1k")
    g.add_argument("--data_dir", type=str, default="./data",
                   help="Root for torchvision auto-download datasets (CIFAR, SVHN, …).")
    g.add_argument("--imagenet_root", type=str, default=None,
                   help="Path to ImageNet root or train folder. Required when "
                        "--id_dataset imagenet1k.")
    g.add_argument("--ood_dataset", type=str, default="cifar100",
                   help="Name of the OOD dataset (informational / used in cache keys).")
    g.add_argument("--ood_dir", type=str, default='/home/sgchr/Documents/Multiple_spaces/Imagenet/ood_data/',
                   help="Path to OOD ImageFolder root (required for non-torchvision OOD).")
    g.add_argument("--num_classes", type=int, default=10,
                   help="Override inferred number of ID classes.")

    # ── Model ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Model")
    g.add_argument("--arch", type=str, default="resnet18",
                   help="resnet18 | resnet34 | resnet50 | swint | vit_b_16")
    g.add_argument("--ckpt", type=str, default='/home/sgchr/Documents/CREWA/checkpoints/resnet18_cifar10_best.pth',
                   help="Checkpoint path (required for custom-trained models).")
    g.add_argument("--no_l2", action="store_true",
                   help="Disable L2 normalisation of cached features.")

    # ── DataLoader ───────────────────────────────────────────────────────────
    g = p.add_argument_group("DataLoader")
    g.add_argument("--batch_size",   type=int, default=256)
    g.add_argument("--num_workers",  type=int, default=4,
                   help="Aliased as --workers for compatibility.")
    g.add_argument("--no_amp",       action="store_true",
                   help="Disable automatic mixed precision (AMP).")
    g.add_argument("--train_max_images", type=int, default=0,
                   help="Cap on training images for caching (0 = all).")

    # ── Caching ──────────────────────────────────────────────────────────────
    g = p.add_argument_group("Caching")
    g.add_argument("--cache_dir", type=str, default="./cache_feats",
                   help="Directory for cached feature arrays.")

    # ── Methods ──────────────────────────────────────────────────────────────
    g = p.add_argument_group("Methods")
    g.add_argument("--methods", nargs="+", required=True,
                   help=f"One or more of: {sorted(ALLOWED_METHODS)}")
    g.add_argument("--seed", type=int, default=0)

    # ── subspaces / logit_gate shared ────────────────────────────────────────
    g = p.add_argument_group("Subspaces / LogitGate params")
    g.add_argument("--energy_keep",     type=float, default=0.3)
    g.add_argument("--gate_energy_keep",type=float, default=0.95)
    g.add_argument("--gate_k_max",      type=int,   default=150)
    g.add_argument("--gate_eps",        type=float, default=1e-6)
    g.add_argument("--gate_threshold",  type=float, default=0.0)

    # ── energy / msp ─────────────────────────────────────────────────────────
    g = p.add_argument_group("Energy / MSP params")
    g.add_argument("--energy_T", type=float, default=1.0)

    # ── kpca_rff ─────────────────────────────────────────────────────────────
    g = p.add_argument_group("KPCA-RFF params")
    g.add_argument("--kpca_gamma",         type=float, default=1.0)
    g.add_argument("--kpca_M",             type=int,   default=2048)
    g.add_argument("--kpca_exp_var_ratio", type=float, default=0.95)

    # ── gradsubspace ─────────────────────────────────────────────────────────
    g = p.add_argument_group("GradSubspace params")
    g.add_argument("--grad_n",             type=int,   default=512)
    g.add_argument("--grad_exp_var_ratio", type=float, default=0.95)
    g.add_argument("--grad_center",        action="store_true")
    g.add_argument("--grad_eps",           type=float, default=1e-8)

    # ── neco ─────────────────────────────────────────────────────────────────
    g = p.add_argument_group("NECO params")
    g.add_argument("--neco_dim", type=int, default=128)

    # ── vim ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("ViM params")
    g.add_argument("--vim_dim",        type=int, default=0)
    g.add_argument("--vim_fit_max",    type=int, default=200_000)
    g.add_argument("--vim_fit_device", type=str, default="cpu")

    return p


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = build_parser()
    # Alias --workers → --num_workers for backwards compat
    sys.argv = [
        a.replace("--workers=", "--num_workers=") if a.startswith("--workers=")
        else ("--num_workers" if a == "--workers" else a)
        for a in sys.argv
    ]
    args = parser.parse_args()

    # Derived flags
    args.use_amp    = (not args.no_amp) and torch.cuda.is_available()
    args.l2_normalize = not args.no_l2

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    methods = parse_methods(args.methods)
    print(f"Methods: {methods}")

    # ── Model ────────────────────────────────────────────────────────────────
    # We build a placeholder to get the transform; num_classes determined later
    # For imagenet pretrained archs, num_classes doesn't matter for ckpt loading
    _placeholder_nc = 10
    model, tfm = build_model(
        arch=args.arch,
        num_classes=_placeholder_nc,
        device=device,
        ckpt_path=args.ckpt,
    )

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("\n[datasets] Building dataset objects …")
    train_ds = build_id_dataset(
        args.id_dataset, "train", tfm,
        data_dir=args.data_dir,
        imagenet_root=args.imagenet_root,
    )
    id_ds = build_id_dataset(
        args.id_dataset, "val" if args.id_dataset == "imagenet1k" else "test",
        tfm,
        data_dir=args.data_dir,
        imagenet_root=args.imagenet_root,
    )
    ood_ds = build_ood_dataset(
        args.ood_dataset, tfm,
        data_dir=args.data_dir,
        ood_dir=args.ood_dir,
    )

    num_classes = args.num_classes or num_classes_from_dataset(train_ds)
    print(f"  ID dataset : {args.id_dataset}  (num_classes={num_classes})")
    print(f"  OOD dataset: {args.ood_dataset}")

    # If the model head doesn't match (e.g., pretrained ResNet-50 has 1000
    # classes but id_dataset=cifar10 has 10), we do NOT rebuild the model —
    # the pretrained head is used for logit-based methods and that is correct
    # because the features are extracted from the penultimate layer only.
    # For custom checkpoints (resnet18/34/vit) the build_model() already set
    # the right num_classes.

    W, b = get_classifier_Wb(model)

    # ── DataLoaders (only needed for caching) ────────────────────────────────
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers)
    id_loader    = make_loader(id_ds,    args.batch_size, args.num_workers)
    ood_loader   = make_loader(ood_ds,   args.batch_size, args.num_workers)

    # ── Feature caching ──────────────────────────────────────────────────────
    paths = make_cache_paths(
        cache_dir=args.cache_dir,
        arch=args.arch,
        id_name=args.id_dataset,
        ood_name=args.ood_dataset,
        l2_normalize=args.l2_normalize,
        train_max_images=args.train_max_images,
        seed=args.seed,
    )
    print("\n[cache] paths:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    train_feats, train_labels, id_feats, ood_feats = extract_all_features(
        model=model,
        train_dataset=train_ds,
        id_dataset=id_ds,
        ood_dataset=ood_ds,
        paths=paths,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
        l2_normalize=args.l2_normalize,
        train_max_images=args.train_max_images,
        seed=args.seed,
    )

    # ── Run methods ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  ID={args.id_dataset}  OOD={args.ood_dataset}  arch={args.arch}")
    print(f"{'=' * 55}")

    for m in methods:
        run_method(
            m,
            train_feats=train_feats,
            train_labels=train_labels,
            id_feats=id_feats,
            ood_feats=ood_feats,
            W=W, b=b,
            num_classes=num_classes,
            device=device,
            args=args,
            # NCI needs the live model and a loader for val extraction
            model=model,
            id_loader=id_loader,
            train_dataset=train_ds,
        )


if __name__ == "__main__":
    main()
