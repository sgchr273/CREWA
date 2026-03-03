# Step by step guide to run the Complement Residual Energy with Weight Alignment(CREWA: an out of distribution detection method)

A unified, well-organized codebase for out-of-distribution detection that
works across **CIFAR-10/100, SVHN, MNIST, FashionMNIST, and ImageNet-1K** as
ID datasets, and any ImageFolder-compatible directory as an OOD dataset.

---

## Project layout

```
ood_bench/
├── run.py          ← single CLI entry-point (replaces run.py + imgnet_1k_run.py)
├── utils.py        ← seed, metrics, score orientation, safe_name
├── models.py       ← backbone building, penultimate extractors, head extraction
├── datasets.py     ← ID/OOD dataset builders for all supported datasets
├── features.py     ← feature extraction + memmap caching
├── evaluate.py     ← method dispatch (calls methods.py)
└── methods.py  ← OOD scoring methods (unchanged)
```

---

## Quick start

### CIFAR-10 with a custom ViT-B/16 checkpoint

```bash
python run.py \
    --id_dataset cifar10 \
    --arch vit_b_16 \
    --ckpt /path/to/vit_b16_cifar10.pt \
    --ood_dir /data/ood \
    --ood_dataset tiny-imagenet-200 \
    --methods crewa energy msp neco vim \
    --cache_dir ./cache_feats
```

### ImageNet-1K with pretrained ResNet-50 vs Places365

```bash
python run.py \
    --id_dataset imagenet1k \
    --imagenet_root /data/ILSVRC/Data/CLS-LOC \
    --arch resnet50 \
    --ood_dir /data/places365 \
    --ood_dataset places \
    --methods crewa mahalanobis vim deca \
    --batch_size 512 --num_workers 8
```

### ImageNet-1K with pretrained SwinT vs iNaturalist

```bash
python run.py \
    --id_dataset imagenet1k \
    --imagenet_root /data/ILSVRC/Data/CLS-LOC \
    --arch swint \
    --ood_dir /data/inaturalist \
    --ood_dataset inaturalist \
    --methods crewa energy neco
```

---

## Supported ID datasets

| `--id_dataset` | Source | Notes |
|---|---|---|
| `cifar10` | torchvision (auto-download) | |
| `cifar100` | torchvision (auto-download) | |
| `svhn` | torchvision (auto-download) | |
| `mnist` | torchvision (auto-download) | RGB-converted |
| `fashionmnist` | torchvision (auto-download) | RGB-converted |
| `imagenet1k` | ImageFolder | needs `--imagenet_root` |

## Supported OOD datasets

Pass `--ood_dir /path/to/folder` and `--ood_dataset <name>` for any of:
`textures`, `places`, `sun`, `dtd`, `inaturalist`, `imagenet_o`,
`tiny-imagenet-200`, or **any custom ImageFolder layout**.

Torchvision OOD datasets (`svhn`, `cifar100`, `mnist`, `fashionmnist`,
`cifar10`) use `--data_dir` and are auto-downloaded.

## Supported architectures

| `--arch` | Weights | Notes |
|---|---|---|
| `resnet18` | custom `--ckpt` | num_classes from ID dataset |
| `resnet34` | custom `--ckpt` | num_classes from ID dataset |
| `vit_b_16` | custom `--ckpt` | num_classes from ID dataset |
| `resnet50` | ImageNet pretrained (V2) | no `--ckpt` needed |
| `swint` | ImageNet pretrained (V1) | no `--ckpt` needed |

## Available methods

`crewa`, `mahalanobis`, `energy`, `msp`, `logit_gate`, `kpca_rff`,
`gradsubspace`, `neco`, `vim`, `deca`, `nci`

---

## Feature caching

Features are extracted once and stored as memory-mapped `.npy` files under
`--cache_dir` (default `./cache_feats`).  The cache key encodes the arch,
dataset name, L2-normalisation flag, and train-image count, so re-runs with
the same configuration are instant.

Use `--no_l2` to disable L2 normalisation of cached features (enabled by
default for most methods).

Use `--train_max_images N` to cap the number of training images used for
fitting (useful for quick ablations on ImageNet-scale datasets).
