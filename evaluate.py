#!/usr/bin/env python3
"""
evaluate.py — OOD method dispatch.

Each method takes the pre-extracted feature tensors plus method-specific
hyper-parameters and returns (id_scores, ood_scores) as numpy arrays with the
convention that *higher score ⟹ more OOD* (enforced automatically via
utils.ensure_ood_higher before reporting).

Adding a new method
-------------------
1.  Implement the logic in methods.py (or here for small helpers).
2.  Add a branch in ``run_method()``.
3.  Add the method name to ALLOWED_METHODS.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils    import ensure_ood_higher, report_metrics
from features import logits_from_feats_batched, extract_gaussian_feats_like_id

import methods as M


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

ALLOWED_METHODS = {
    "crewa",
    "mahalanobis",
    "energy",
    "msp",
    "logit_gate",
    "kpca_rff",
    "gradsubspace",
    "neco",
    "vim",
    "deca",
    "nci",
}


def parse_methods(raw: list[str]) -> list[str]:
    out = []
    for m in raw:
        m = m.lower()
        if m not in ALLOWED_METHODS:
            raise ValueError(
                f"Unknown method '{m}'. Allowed: {sorted(ALLOWED_METHODS)}"
            )
        if m not in out:
            out.append(m)
    return out


# ─────────────────────────────────────────────
# Main dispatch
# ─────────────────────────────────────────────

def run_method(
    method: str,
    *,
    train_feats:  torch.Tensor,
    train_labels: torch.Tensor,
    id_feats:     torch.Tensor,
    ood_feats:    torch.Tensor,
    W:            torch.Tensor,
    b:            Optional[torch.Tensor],
    num_classes:  int,
    device:       torch.device,
    args,                           # the full parsed argparse Namespace
    # NCI only
    model:        Optional[nn.Module]    = None,
    id_loader:    Optional[DataLoader]   = None,
    train_dataset = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a single OOD detection method and print its metrics.

    Returns (id_scores, ood_scores) oriented so that higher ⟹ more OOD.
    """
    m = method.lower()

    # ── crewa ────────────────────────────────────────────────────────────
    if m == "crewa":
        logits_id  = logits_from_feats_batched(id_feats,  W, b, device=device)
        logits_ood = logits_from_feats_batched(ood_feats, W, b, device=device)

        id_s, ood_s = M.run_crewa(
            train_feats=train_feats,
            train_labels=train_labels,
            id_feats=id_feats,
            id_logits=logits_id,
            ood_feats=ood_feats,
            ood_logits=logits_ood,
            num_classes=num_classes,
            energy_keep=getattr(args, "energy_keep", 0.3),
            k_max=0,
            W=W,
            beta=1,
            tune_beta=True,
            gamma=1,
            use_centered_for_cos=True,
        )
        id_s, ood_s = ensure_ood_higher(np.asarray(id_s), np.asarray(ood_s))
        report_metrics("crewa", id_s, ood_s)
        return id_s, ood_s

    # ── mahalanobis ──────────────────────────────────────────────────────────
    if m == "mahalanobis":
        id_s, ood_s = M.neco_mahalanobis(
            train_feats=train_feats,
            train_labels=train_labels,
            id_feats=id_feats,
            ood_feats=ood_feats,
            num_classes=num_classes,
        )
        id_s, ood_s = ensure_ood_higher(np.asarray(id_s), np.asarray(ood_s))
        report_metrics("mahalanobis", id_s, ood_s)
        return id_s, ood_s

    # ── energy ───────────────────────────────────────────────────────────────
    if m == "energy":
        logits_id  = logits_from_feats_batched(id_feats,  W, b, device=device)
        logits_ood = logits_from_feats_batched(ood_feats, W, b, device=device)
        T          = getattr(args, "energy_T", 1.0)
        id_s  = M.scores_energy_from_logits(logits_id,  T=T)
        ood_s = M.scores_energy_from_logits(logits_ood, T=T)
        id_s, ood_s = ensure_ood_higher(np.asarray(id_s), np.asarray(ood_s))
        report_metrics(f"energy (T={T})", id_s, ood_s)
        return id_s, ood_s

    # ── msp ──────────────────────────────────────────────────────────────────
    if m == "msp":
        logits_id  = logits_from_feats_batched(id_feats,  W, b, device=device)
        logits_ood = logits_from_feats_batched(ood_feats, W, b, device=device)
        id_s  = M.scores_msp_from_logits(logits_id)
        ood_s = M.scores_msp_from_logits(logits_ood)
        id_s, ood_s = ensure_ood_higher(np.asarray(id_s), np.asarray(ood_s))
        report_metrics("msp", id_s, ood_s)
        return id_s, ood_s

    # ── logit_gate ───────────────────────────────────────────────────────────
    if m == "logit_gate":
        feat_mean, Vk = M.fit_global_pca_basis(
            train_feats,
            energy_keep=getattr(args, "gate_energy_keep", 0.95),
            k_max=getattr(args, "gate_k_max", 150),
            eps=getattr(args, "gate_eps", 1e-6),
        )
        eps = getattr(args, "gate_eps", 1e-6)
        thr = getattr(args, "gate_threshold", 0.0)
        id_s  = M.scores_logit_gate(id_feats,  W, b, feat_mean, Vk, threshold=thr, eps=eps)
        ood_s = M.scores_logit_gate(ood_feats, W, b, feat_mean, Vk, threshold=thr, eps=eps)
        id_s, ood_s = ensure_ood_higher(np.asarray(id_s), np.asarray(ood_s))
        report_metrics("logit_gate", id_s, ood_s)
        return id_s, ood_s

    # ── kpca_rff ─────────────────────────────────────────────────────────────
    if m == "kpca_rff":
        gamma = getattr(args, "kpca_gamma", 1.0)
        MM    = getattr(args, "kpca_M",     2048)
        evr   = getattr(args, "kpca_exp_var_ratio", 0.95)
        mu_map, Uq, q = M.fit_kpca_rff(
            train_feats=train_feats, gamma=gamma, M=MM,
            exp_var_ratio=evr, seed=args.seed,
        )
        kw = dict(mu=mu_map, Uq=Uq, gamma=gamma, M=MM, seed=args.seed)
        id_s  = M.scores_kpca_rff(query_feats=id_feats,  **kw)
        ood_s = M.scores_kpca_rff(query_feats=ood_feats, **kw)
        id_s, ood_s = ensure_ood_higher(np.asarray(id_s), np.asarray(ood_s))
        report_metrics(f"kpca_rff (gamma={gamma}, M={MM}, q={q})", id_s, ood_s)
        return id_s, ood_s

    # ── gradsubspace ─────────────────────────────────────────────────────────
    if m == "gradsubspace":
        eps = getattr(args, "grad_eps", 1e-8)
        S, mu, k = M.fit_id_feature_subspace(
            train_feats=train_feats,
            n_batch=getattr(args, "grad_n", 512),
            exp_var_ratio=getattr(args, "grad_exp_var_ratio", 0.95),
            center=getattr(args, "grad_center", False),
            seed=args.seed,
            eps=eps,
        )
        print(f"[gradsubspace] k={k} components selected.")
        kw    = dict(W=W, b=b, S=S, mu=mu, num_classes=num_classes, eps=eps)
        id_s  = M.scores_gradsubspace_pseudo_resid(feats=id_feats,  **kw)
        ood_s = M.scores_gradsubspace_pseudo_resid(feats=ood_feats, **kw)
        id_s, ood_s = ensure_ood_higher(np.asarray(id_s), np.asarray(ood_s))
        report_metrics("gradsubspace", id_s, ood_s)
        return id_s, ood_s

    # ── neco ─────────────────────────────────────────────────────────────────
    if m == "neco":
        id_s, ood_s = M.run_method_neco(
            train_feats=train_feats,
            id_feats=id_feats,
            ood_feats=ood_feats,
            W=W, b=b,
            arch=args.arch,
            neco_dim=getattr(args, "neco_dim", 128),
        )
        id_s, ood_s = ensure_ood_higher(np.asarray(id_s), np.asarray(ood_s))
        report_metrics("neco", id_s, ood_s)
        return id_s, ood_s

    # ── vim ───────────────────────────────────────────────────────────────────
    if m == "vim":
        s_id, s_ood = M.run_method_vim(
            train_feats=train_feats,
            id_feats=id_feats,
            ood_feats=ood_feats,
            W=W, b=b,
            num_classes=num_classes,
            vim_dim=getattr(args, "vim_dim", 0),
            vim_fit_max=getattr(args, "vim_fit_max", 200_000),
            seed=args.seed,
            fit_device=torch.device(getattr(args, "vim_fit_device", "cpu")),
            score_device=device,
        )
        s_id, s_ood = ensure_ood_higher(np.asarray(s_id), np.asarray(s_ood))
        report_metrics("vim", s_id, s_ood)
        return s_id, s_ood

    # ── deca ─────────────────────────────────────────────────────────────────
    if m == "deca":
        s_id, s_ood = M.run_deca(
            train_feats=train_feats,
            id_feats=id_feats,
            ood_feats=ood_feats,
        )
        s_id, s_ood = ensure_ood_higher(np.asarray(s_id), np.asarray(s_ood))
        report_metrics("deca", s_id, s_ood)
        return s_id, s_ood

    # ── nci ──────────────────────────────────────────────────────────────────
    if m == "nci":
        return _run_nci(
            train_feats=train_feats,
            id_feats=id_feats,
            ood_feats=ood_feats,
            W=W, b=b,
            model=model,
            id_loader=id_loader,
            train_dataset=train_dataset,
            device=device,
            args=args,
        )

    raise ValueError(f"Method '{method}' not implemented in run_method().")


# ─────────────────────────────────────────────
# NCI helper (requires live model for val extraction)
# ─────────────────────────────────────────────

def _run_nci(
    *,
    train_feats:   torch.Tensor,
    id_feats:      torch.Tensor,
    ood_feats:     torch.Tensor,
    W:             torch.Tensor,
    b:             Optional[torch.Tensor],
    model:         nn.Module,
    id_loader:     DataLoader,
    train_dataset,
    device:        torch.device,
    args,
) -> Tuple[np.ndarray, np.ndarray]:
    from torch.utils.data import DataLoader, Subset

    # 1) Build ID-val loader (subsample from ID train dataset)
    n_full   = len(train_dataset)
    val_size = int(min(5000, n_full))
    rng      = np.random.RandomState(args.seed)
    idx      = rng.choice(n_full, size=val_size, replace=False).tolist()
    id_val_loader = DataLoader(
        Subset(train_dataset, idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(args, "num_workers", getattr(args, "workers", 4)),
        pin_memory=True,
    )

    # 2) Extract val feats + Gaussian feats
    use_amp      = getattr(args, "use_amp", False)
    id_val_feats = _extract_plain_feats(model, id_val_loader, device, use_amp)
    gau_val_feats = extract_gaussian_feats_like_id(model, id_val_loader, device, use_amp=use_amp)

    # 3) Global mean from train feats
    mu_g = M.compute_mu_global(train_feats)

    # 4) Tune varpi on ID-val vs Gaussian
    varpi_grid = [1e-4, 1e-3, 1e-2, 1e-1]
    best_varpi, best_auc = None, -1.0
    for v in varpi_grid:
        _, s_id_v   = M.nci_scores_batched(id_val_feats,  W, b, mu_g, alpha=v, p_norm=1)
        _, s_gau_v  = M.nci_scores_batched(gau_val_feats, W, b, mu_g, alpha=v, p_norm=1)
        auroc, _    = report_metrics(f"NCI-val (varpi={v:g})", s_id_v, s_gau_v)
        if auroc > best_auc:
            best_auc, best_varpi = auroc, float(v)

    print(f"[NCI] chosen varpi={best_varpi}  val AUROC={best_auc:.4f}")

    # 5) Final eval on real OOD
    _, id_s   = M.nci_scores_batched(id_feats,  W, b, mu_g, alpha=best_varpi, p_norm=1)
    _, ood_s  = M.nci_scores_batched(ood_feats, W, b, mu_g, alpha=best_varpi, p_norm=1)
    id_s, ood_s = ensure_ood_higher(np.asarray(id_s), np.asarray(ood_s))
    report_metrics("nci", id_s, ood_s)
    return id_s, ood_s


@torch.no_grad()
def _extract_plain_feats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> torch.Tensor:
    """Extract features from a model that returns (feats, logits) or just feats."""
    from features import _unwrap_feats
    model.eval()
    outs    = []
    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with amp_ctx:
            out = model(x)
        outs.append(_unwrap_feats(out).detach().float().cpu())
    return torch.cat(outs, dim=0)
