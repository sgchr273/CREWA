#!/usr/bin/env python3
"""
models.py — Model loading, penultimate-layer extractors, classifier head extraction.

Supported architectures
-----------------------
CIFAR-style (loaded from a custom checkpoint):
    resnet18, resnet34, vit_b_16

ImageNet pretrained (weights bundled with torchvision):
    resnet50, swint  (swin_t)

All models return a (feats, logits) pair when wrapped via build_feat_model(),
or just feats if l2_normalize is requested.
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet18,  ResNet18_Weights,
    resnet34,  ResNet34_Weights,
    resnet50,  ResNet50_Weights,
    swin_t,    Swin_T_Weights,
    vit_b_16,  ViT_B_16_Weights,
)


# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────

def load_state_dict_flex(ckpt_path: str) -> dict:
    """
    Load a checkpoint and return a plain state_dict regardless of how the
    checkpoint was saved (bare dict, or wrapped under 'model_state'/'model'/'state_dict').
    """
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("model_state", "model", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        # Treat the whole dict as a state_dict if all values are tensors
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise RuntimeError(
        f"Cannot interpret checkpoint at '{ckpt_path}' as a state_dict."
    )


# ─────────────────────────────────────────────
# Backbone wrappers (return feats, logits)
# ─────────────────────────────────────────────

class ResNetWithFeats(nn.Module):
    """Wraps a torchvision ResNet to return (penultimate_feats, logits)."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bb = self.backbone
        x = bb.conv1(x);  x = bb.bn1(x);  x = bb.relu(x);  x = bb.maxpool(x)
        x = bb.layer1(x); x = bb.layer2(x)
        x = bb.layer3(x); x = bb.layer4(x)
        x = bb.avgpool(x)
        feats  = torch.flatten(x, 1)
        logits = bb.fc(feats)
        return feats, logits


class SwinWithFeats(nn.Module):
    """
    Wraps a torchvision (or timm) Swin Transformer to return
    (penultimate_feats, logits), handling both tensor layout conventions.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bb = self.backbone

        # timm-style Swin
        if hasattr(bb, "forward_features"):
            feats = bb.forward_features(x)
            if feats.ndim == 3:        # (B, L, C)
                feats = feats.mean(dim=1)
            elif feats.ndim == 4:      # (B, C, H, W)
                feats = feats.mean(dim=(2, 3))
            logits = bb.head(feats) if (hasattr(bb, "head") and bb.head is not None) else None
            return feats, logits

        # torchvision-style Swin
        if hasattr(bb, "features") and hasattr(bb, "norm"):
            z = bb.features(x)
            if z.ndim == 4:
                C = None
                if hasattr(bb.norm, "normalized_shape"):
                    ns = bb.norm.normalized_shape
                    C = int(ns[0]) if isinstance(ns, (tuple, list)) else int(ns)
                if C is not None and z.shape[-1] != C:
                    for d in (1, 2, 3):
                        if z.shape[d] == C:
                            if d == 1:
                                z = z.permute(0, 2, 3, 1)
                            elif d == 2:
                                z = z.permute(0, 1, 3, 2)
                            break
            z      = bb.norm(z)
            feats  = z.mean(dim=(1, 2))
            logits = bb.head(feats) if (hasattr(bb, "head") and bb.head is not None) else None
            return feats, logits

        raise ValueError(
            "Unsupported Swin backbone: expected timm forward_features or "
            "torchvision features+norm."
        )


# ─────────────────────────────────────────────
# Build model
# ─────────────────────────────────────────────

def build_model(
    arch: str,
    num_classes: int,
    device: torch.device,
    ckpt_path: Optional[str] = None,
) -> Tuple[nn.Module, object]:
    """
    Build a classifier and return ``(model, transforms)``.

    For ImageNet pretrained architectures (resnet50, swint) the checkpoint is
    ignored and torchvision pretrained weights are used directly.
    For CIFAR-style architectures (resnet18, resnet34, vit_b_16) ``ckpt_path``
    is required and the model head is replaced to match ``num_classes``.

    Returns
    -------
    model : nn.Module wrapped with ResNetWithFeats / SwinWithFeats so that
            forward() returns (feats, logits).
    tfm   : torchvision transform object suitable for preprocessing.
    """
    a = arch.lower().replace("-", "_")

    # ── ImageNet pretrained ──────────────────────────────────────────────────
    if a == "resnet50":
        w       = ResNet50_Weights.IMAGENET1K_V2
        backbone = resnet50(weights=w)
        model    = ResNetWithFeats(backbone)
        return model.eval().to(device), w.transforms()

    if a in ("swint", "swin_t"):
        w       = Swin_T_Weights.IMAGENET1K_V1
        backbone = swin_t(weights=w)
        model    = SwinWithFeats(backbone)
        return model.eval().to(device), w.transforms()

    # ── Custom checkpoint ────────────────────────────────────────────────────
    if ckpt_path is None:
        raise ValueError(f"arch='{arch}' requires --ckpt (no bundled weights).")

    sd = load_state_dict_flex(ckpt_path)

    if a == "resnet18":
        backbone    = resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        backbone.load_state_dict(sd, strict=True)
        model = ResNetWithFeats(backbone)
        tfm   = ResNet18_Weights.IMAGENET1K_V1.transforms()
        return model.eval().to(device), tfm

    if a == "resnet34":
        backbone    = resnet34(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        backbone.load_state_dict(sd, strict=True)
        model = ResNetWithFeats(backbone)
        tfm   = ResNet34_Weights.IMAGENET1K_V1.transforms()
        return model.eval().to(device), tfm

    if a in ("vit_b_16", "vitb16", "vit_b16"):
        w       = ViT_B_16_Weights.IMAGENET1K_V1
        backbone = vit_b_16(weights=w)
        in_f    = backbone.heads.head.in_features
        backbone.heads = nn.Sequential(nn.Linear(in_f, num_classes))
        backbone.load_state_dict(sd, strict=True)
        # Use a plain identity wrapper so forward() returns (feats, logits)
        model = _ViTWithFeats(backbone)
        return model.eval().to(device), w.transforms()

    raise ValueError(f"Unknown --arch '{arch}'. "
                     "Supported: resnet18, resnet34, resnet50, swint, vit_b_16")


class _ViTWithFeats(nn.Module):
    """Wraps a torchvision ViT to return (penultimate_feats, logits)."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bb   = self.backbone
        # Standard torchvision ViT: _process_input → encoder → heads
        x    = bb._process_input(x)
        n    = x.shape[0]
        cls  = bb.class_token.expand(n, -1, -1)
        x    = torch.cat([cls, x], dim=1)
        x    = bb.encoder(x)
        feats  = x[:, 0]           # CLS token
        logits = bb.heads(feats)
        return feats, logits


# ─────────────────────────────────────────────
# Penultimate-only extractor (optional L2 norm)
# ─────────────────────────────────────────────

class PenultimateExtractor(nn.Module):
    """
    Thin wrapper that calls ``model.forward()`` and returns only the feature
    vector (first element), optionally L2-normalised.
    """

    def __init__(self, model: nn.Module, l2_normalize: bool = False):
        super().__init__()
        self.model        = model
        self.l2_normalize = l2_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out   = self.model(x)
        feats = out[0] if isinstance(out, (tuple, list)) else out
        if self.l2_normalize:
            feats = F.normalize(feats, p=2, dim=1)
        return feats


# ─────────────────────────────────────────────
# Classifier head extraction
# ─────────────────────────────────────────────

def get_classifier_Wb(
    model: nn.Module,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract weight W and bias b from the classifier head.

    Works with ResNetWithFeats (bb.fc), SwinWithFeats (bb.head),
    and _ViTWithFeats (bb.heads[0]).

    Returns
    -------
    W : Tensor [num_classes, feat_dim]
    b : Tensor [num_classes] or None
    """
    # Unwrap if the model is a PenultimateExtractor
    inner = getattr(model, "model", model)
    bb    = getattr(inner, "backbone", inner)

    # ResNet-style
    if hasattr(bb, "fc") and isinstance(bb.fc, nn.Linear):
        head = bb.fc

    # Swin / ViM-style
    elif hasattr(bb, "head") and isinstance(bb.head, nn.Linear):
        head = bb.head

    # ViT-style (heads is nn.Sequential with one Linear)
    elif hasattr(bb, "heads"):
        h = bb.heads
        head = h[0] if isinstance(h, nn.Sequential) else h
        if not isinstance(head, nn.Linear):
            raise RuntimeError("bb.heads does not contain an nn.Linear.")

    else:
        raise ValueError(
            "Cannot locate a Linear classifier head. "
            "Expected bb.fc, bb.head, or bb.heads."
        )

    W = head.weight.detach().cpu().float()
    b = head.bias.detach().cpu().float() if head.bias is not None else None
    return W, b
