import os
import copy
import random
import argparse
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from typing import Tuple, Optional, Union

NormOrd = Union[int, float, str]  # e.g. 1, 2, "inf"


from typing import Dict, Tuple
import numpy as np
import torch


# =========================
# 1) Fit: pooled within class PCA (shared basis, class means)
# =========================
def fit_affine_pca_subspaces(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    energy_keep: float = 0.95,
    k_max: int = 150,   # kept for signature compatibility, NOT used
    eps: float = 1e-6,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Pooled within class PCA (tied covariance):
      mu_c: class means
      X = feats - mu_c[labels]
      SVD on X to get a shared PCA basis
      keep smallest k reaching energy_keep
      store V_perp and lam_perp for residual scoring
    """
    feats = feats.float()
    labels = labels.long()
    _ = k_max

    N, D = feats.shape
    if N < 5:
        raise RuntimeError(f"Too few samples ({N}).")
    if labels.numel() != N:
        raise RuntimeError(f"labels length mismatch: {labels.numel()} vs {N}")

    # class means
    mu_c = feats.new_zeros((num_classes, D))
    counts = torch.zeros((num_classes,), device=feats.device, dtype=torch.long)
    for c in range(num_classes):
        idx = (labels == c)
        nc = int(idx.sum().item())
        if nc == 0:
            raise RuntimeError(f"class {c} has zero samples")
        mu_c[c] = feats[idx].mean(dim=0)
        counts[c] = nc

    # pooled within class residuals
    X = feats - mu_c[labels]  # [N, D]

    # SVD
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh.transpose(0, 1)  # [D, r]
    r = V.shape[1]
    lam = (S ** 2) / max(N - 1, 1)
    print('here')
    # energy keep
    energy = torch.cumsum(lam, dim=0) / (lam.sum() + eps)
    k_energy = int(torch.searchsorted(
        energy, torch.tensor(energy_keep, device=energy.device)
    ).item()) + 1
    k = min(k_energy, max(r - 1, 1))
    stable_r = float(((S**2).sum()**2) / ((S**4).sum() + eps))
    k_stable  = max(1, min(int(0.3 * stable_r), r - 64))
    print(
        f"[dim summary] r={r} | stable_rank={stable_r:.1f} | "
        f"k_used={k} (energy_keep={energy_keep}) | "
        f"k_stable_suggestion={k_stable} | "
        f"perp_dim={r - k} | retained_energy={float(energy[k-1]):.4f}"
    )
    k = k_stable
    # ensure at least 1 perp direction
    
    # print(f"[pooled within class pca] retained components k={k} (energy_keep={energy_keep}, rank={r}, perp_dim={r-k})")
    # instead of energy_keep driving k, derive it from stable rank
    V_perp = V[:, k:]
    lam_perp = lam[k:]
    V_keep = V[:, :k]
    if V_perp.numel() == 0:
        V_perp = V[:, -1:].contiguous()
        lam_perp = lam[-1:].contiguous()

    return {
        0: {
            "mu_c": mu_c.contiguous(),          # [C, D]
            "V_keep":  V_keep.contiguous(),
            "V_perp": V_perp.contiguous(),      # [D, p]
            "lam_perp": lam_perp.contiguous(),  # [p]
            "counts": counts.contiguous(),      # [C]
        }
    }


# =========================


@torch.no_grad()
def score_subspaces_resid_plus_align_simple(
    feats: torch.Tensor,
    logits: torch.Tensor,
    W: torch.Tensor,                      # [C, D] classifier weights
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    *,
    beta: float = 1.0,
    use_centered_for_cos: bool = False,   # if True: use (x - mu_c[pred]) for cosine
    eps: float = 1e-6,
    # ---- beta tuning (ID-only) ----
    tune_beta: bool = False,
    gamma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score = residual energy in discarded space + beta * alignment penalty

    If tune_beta=True, compute an ID-only calibrated beta:
      beta_used = gamma * median(s_res) / median(a)
    using the *current* (feats, logits) passed to this function as the calibration set.

    Returns:
      s_total: numpy scores (float64)
      a:       alignment penalty (float64) (kept for signature stability)
    """
    feats = feats.float()
    logits = logits.float()
    W = W.float()

    m = subspaces[0] if 0 in subspaces else subspaces[next(iter(subspaces.keys()))]
    if "mu_c" not in m:
        raise KeyError("expected pooled within class fit: subspaces[0] must contain 'mu_c'")

    mu_c = m["mu_c"]    # [C, D]
    Vp   = m["V_perp"]  # [D, p]

    if feats.ndim != 2:
        raise RuntimeError(f"feats must be [N, D], got {tuple(feats.shape)}")
    if logits.ndim != 2:
        raise RuntimeError(f"logits must be [N, C], got {tuple(logits.shape)}")
    if feats.shape[0] != logits.shape[0]:
        raise RuntimeError(f"N mismatch feats vs logits: {feats.shape[0]} vs {logits.shape[0]}")
    if feats.shape[1] != mu_c.shape[1]:
        raise RuntimeError(f"D mismatch feats vs mu_c: {feats.shape[1]} vs {mu_c.shape[1]}")
    if logits.shape[1] != mu_c.shape[0]:
        raise RuntimeError(f"C mismatch logits vs mu_c: {logits.shape[1]} vs {mu_c.shape[0]}")
    if W.shape != (mu_c.shape[0], mu_c.shape[1]):
        raise RuntimeError(f"W must be [C, D] = {tuple(mu_c.shape)}, got {tuple(W.shape)}")

    # predicted class

    # dists = (
    #     (feats ** 2).sum(dim=1, keepdim=True)          # [N, 1]
    #     + (mu_c ** 2).sum(dim=1, keepdim=True).T        # [1, C]
    #     - 2.0 * feats @ mu_c.T                          # [N, C]
    # )  # [N, C]

    # # nearest class
    # c_star = torch.argmin(dists, dim=1)  # [N]

    # # centered residual
    # z = feats - mu_c[c_star]  

    # ----- residual energy in discarded space -----
    # pred = torch.argmax(logits, dim=1).long()     # [N]
    pred = torch.argmax(feats @ W.T , dim=1).long()
    diff = (logits - (feats @ W.T)).abs().max()
    print(diff)
    z = feats - mu_c[pred]                         # [N, D]
    coeff = z @ Vp                                 # [N, p]
    s_res = (coeff ** 2).sum(dim=1)                # [N]
    

    # ----- alignment penalty -----
    u = z if use_centered_for_cos else feats       # [N, D]
    u_n = F.normalize(u, p=2, dim=1, eps=eps)      # [N, D]
    W_n = F.normalize(W, p=2, dim=1, eps=eps)      # [C, D]
    # w_pred = W_n[c_star]                             # [N, D]
    w_pred = W_n[pred]                             # [N, D]

    cos = (u_n * w_pred).sum(dim=1).clamp(-1.0, 1.0)  # [N]
    a = 1.0 - cos                                      # [N]

    # ----- beta tuning (ID-only) -----
    beta_used = float(beta)
    if tune_beta:
        # ID-only calibration on the provided batch: beta = gamma * median(s_res) / median(a)
        s_res_np = s_res.detach().cpu().numpy().astype(np.float64)
        a_np_tmp = a.detach().cpu().numpy().astype(np.float64)

        med_res = float(np.median(s_res_np))
        med_a   = float(np.median(a_np_tmp))
        beta_used = 0.0 if med_a < 1e-12 else float(gamma) * (med_res / med_a)

    # ----- combined score -----
    s_total = s_res + float(beta_used) * a
    s_np = s_total.detach().cpu().numpy().astype(np.float64)
    a_np = a.detach().cpu().numpy().astype(np.float64)
    return s_np, a_np


def run_method_subspaces_simple(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    id_feats: torch.Tensor,
    id_logits: torch.Tensor,
    ood_feats: torch.Tensor,
    ood_logits: torch.Tensor,
    num_classes: int,
    energy_keep: float,
    k_max: int,
    *,
    W: torch.Tensor,                 # [C, D]
    beta: float = 1.0,
    use_centered_for_cos: bool = False,
    tune_beta: bool = False,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    subspaces = fit_affine_pca_subspaces(
        feats=train_feats,
        labels=train_labels,
        num_classes=num_classes,
        energy_keep=energy_keep,
        k_max=k_max,
        eps=eps,
    )

    # Tune beta on ID only, then reuse that beta for both ID and OOD scoring
    beta_used = float(beta)
    if tune_beta:
        # get s_res and a on ID by calling with tune_beta=False and beta=0, then compute beta_used
        s_res_id, a_id = score_subspaces_resid_plus_align_simple(
            id_feats, id_logits, W, subspaces,
            beta=0.0,
            use_centered_for_cos=use_centered_for_cos,
            eps=eps,
            tune_beta=False,
        )
        med_res = float(np.median(s_res_id))
        med_a   = float(np.median(a_id))
        beta_used = 0.0 if med_a < 1e-12 else float(gamma) * (med_res / med_a)

    print(f"[subspaces] beta_used={beta_used:.6g} (tune_beta={tune_beta}, gamma={gamma})")

    s_id, _  = score_subspaces_resid_plus_align_simple(
        id_feats, id_logits, W, subspaces,
        beta=beta_used,
        use_centered_for_cos=use_centered_for_cos,
        eps=eps,
        tune_beta=False,
    )
    s_ood, _ = score_subspaces_resid_plus_align_simple(
        ood_feats, ood_logits, W, subspaces,
        beta=beta_used,
        use_centered_for_cos=use_centered_for_cos,
        eps=eps,
        tune_beta=False,
    )
    return s_id, s_ood

#####Following version does not k_max cap

# @torch.no_grad()

# def fit_affine_pca_subspaces(
#     feats: torch.Tensor,
#     labels: torch.Tensor,
#     num_classes: int,
#     energy_keep: float = 0.95,
#     k_max: int = 150,   # kept for signature compatibility, NOT used
#     eps: float = 1e-6,
# ) -> Dict[int, Dict[str, torch.Tensor]]:
#     # GLOBAL PCA, NO CAP: keep exactly the smallest k that reaches energy_keep
#     feats = feats.float()
#     _ = labels
#     _ = num_classes
#     _ = k_max

#     Xc = feats
#     if Xc.shape[0] < 5:
#         raise RuntimeError(f"Too few samples ({Xc.shape[0]}).")

#     mu = Xc.mean(dim=0)
#     X = Xc - mu

#     _, S, Vh = torch.linalg.svd(X, full_matrices=False)
#     V = Vh.transpose(0, 1)
#     n = X.shape[0]
#     lam = (S**2) / max(n - 1, 1)

#     energy = torch.cumsum(lam, dim=0) / (lam.sum() + eps)
#     k_energy = int(torch.searchsorted(energy, torch.tensor(energy_keep, device=energy.device)).item()) + 1

#     r = V.shape[1]
#     # no cap; only ensure we leave at least 1 perp direction
#     k = min(k_energy, max(r - 1, 1))
#     print(f"[global pca] retained components k={k} (energy_keep={energy_keep}, rank={r}, perp_dim={r-k})")

#     V_perp = V[:, k:]
#     lam_perp = lam[k:]

#     if V_perp.numel() == 0:
#         # safety (should be rare due to k<=r-1)
#         V_perp = V[:, -1:].contiguous()
#         lam_perp = lam[-1:].contiguous()

#     return {
#         0: {
#             "mu": mu.contiguous(),
#             "V_perp": V_perp.contiguous(),
#             "lam_perp": lam_perp.contiguous(),
#         }
#     }



# @torch.no_grad()
# def score_subspaces_ratio_min(
#     feats: torch.Tensor,
#     subspaces: Dict[int, Dict[str, torch.Tensor]],
#     eps: float = 1e-6,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     # GLOBAL PCA scoring (use the single global subspace)
#     feats = feats.float()

#     # accept any dict; take the first (or key 0 if present)
#     if 0 in subspaces:
#         m = subspaces[0]
#     else:
#         m = subspaces[next(iter(subspaces.keys()))]

#     mu = m["mu"]
#     Vp = m["V_perp"]
#     lam = m["lam_perp"]  # not used (kept in model for optional whitening)

#     z = feats - mu.unsqueeze(0)
#     coeff = z @ Vp
#     s = ((coeff**2))  #/ (lam.unsqueeze(0) + eps)).sum(dim=1)
#     s = s.sum(dim=1) #/ Vp.shape[1]
#     s1 = s.cpu().numpy().astype(np.float64)
#     ratio = np.ones_like(s1, dtype=np.float64)  # placeholder to keep return type stable
#     return s1, ratio


# def run_method_subspaces(
#     train_feats: torch.Tensor,
#     train_labels: torch.Tensor,
#     id_feats: torch.Tensor,
#     ood_feats: torch.Tensor,
#     num_classes: int,
#     energy_keep: float,
#     k_max: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     subspaces = fit_affine_pca_subspaces(
#         feats=train_feats,
#         labels=train_labels,
#         num_classes=num_classes,
#         energy_keep=energy_keep,
#         k_max=k_max,
#     )
#     s1_id, _ = score_subspaces_ratio_min(id_feats, subspaces)
#     s1_ood, _ = score_subspaces_ratio_min(ood_feats, subspaces)


#     return s1_id, s1_ood


from typing import Dict, Tuple
import numpy as np
import torch


# =========================
# 1) Fit: pooled within class PCA (shared basis, class means)
# =========================
# def fit_affine_pca_subspaces(
#     feats: torch.Tensor,
#     labels: torch.Tensor,
#     num_classes: int,
#     energy_keep: float = 0.95,
#     k_max: int = 150,   # kept for signature compatibility, NOT used
#     eps: float = 1e-6,
# ) -> Dict[int, Dict[str, torch.Tensor]]:
#     """
#     Pooled within class PCA (tied covariance):
#       mu_c: class means
#       X = feats - mu_c[labels]
#       SVD on X to get a shared PCA basis
#       keep smallest k reaching energy_keep
#       store V_perp and lam_perp for residual scoring
#     """
#     feats = feats.float()
#     labels = labels.long()
#     _ = k_max

#     N, D = feats.shape
#     if N < 5:
#         raise RuntimeError(f"Too few samples ({N}).")
#     if labels.numel() != N:
#         raise RuntimeError(f"labels length mismatch: {labels.numel()} vs {N}")

#     # class means
#     mu_c = feats.new_zeros((num_classes, D))
#     counts = torch.zeros((num_classes,), device=feats.device, dtype=torch.long)
#     for c in range(num_classes):
#         idx = (labels == c)
#         nc = int(idx.sum().item())
#         if nc == 0:
#             raise RuntimeError(f"class {c} has zero samples")
#         mu_c[c] = feats[idx].mean(dim=0)
#         counts[c] = nc

#     # pooled within class residuals
#     X = feats - mu_c[labels]  # [N, D]

#     # SVD
#     _, S, Vh = torch.linalg.svd(X, full_matrices=False)
#     V = Vh.transpose(0, 1)  # [D, r]
#     r = V.shape[1]
#     lam = (S ** 2) / max(N - 1, 1)

#     # energy keep
#     energy = torch.cumsum(lam, dim=0) / (lam.sum() + eps)
#     k_energy = int(torch.searchsorted(
#         energy, torch.tensor(energy_keep, device=energy.device)
#     ).item()) + 1

#     # ensure at least 1 perp direction
#     k = min(k_energy, max(r - 1, 1))
#     print(f"[pooled within class pca] retained components k={k} (energy_keep={energy_keep}, rank={r}, perp_dim={r-k})")

#     V_perp = V[:, k:]
#     lam_perp = lam[k:]

#     if V_perp.numel() == 0:
#         V_perp = V[:, -1:].contiguous()
#         lam_perp = lam[-1:].contiguous()

#     return {
#         0: {
#             "mu_c": mu_c.contiguous(),          # [C, D]
#             "V_perp": V_perp.contiguous(),      # [D, p]
#             "lam_perp": lam_perp.contiguous(),  # [p]
#             "counts": counts.contiguous(),      # [C]
#         }
#     }


# =========================
# 2) Score: predicted class scoring (highlighted method)
# =========================
@torch.no_grad()
def score_subspaces_ratio_min(
    feats: torch.Tensor,
    logits: torch.Tensor,
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicted class scoring:
      c_hat = argmax(logits)
      z = x - mu_{c_hat}
      s = || V_perp^T z ||^2  (same residual energy you use now)

    Returns:
      s1: numpy scores (higher means more OOD if you treat residual as OOD score)
      ratio: placeholder ones to keep return type stable
    """
    feats = feats.float()
    logits = logits.float()

    m = subspaces[0] if 0 in subspaces else subspaces[next(iter(subspaces.keys()))]

    if "mu_c" not in m:
        raise KeyError("expected pooled within class fit: subspaces[0] must contain 'mu_c'")

    mu_c = m["mu_c"]         # [C, D]
    Vp = m["V_perp"]         # [D, p]
    # lam = m["lam_perp"]    # available if you want whitening later

    if feats.ndim != 2:
        raise RuntimeError(f"feats must be [N, D], got {tuple(feats.shape)}")
    if logits.ndim != 2:
        raise RuntimeError(f"logits must be [N, C], got {tuple(logits.shape)}")
    if feats.shape[0] != logits.shape[0]:
        raise RuntimeError(f"N mismatch feats vs logits: {feats.shape[0]} vs {logits.shape[0]}")
    if feats.shape[1] != mu_c.shape[1]:
        raise RuntimeError(f"D mismatch feats vs mu_c: {feats.shape[1]} vs {mu_c.shape[1]}")
    if logits.shape[1] != mu_c.shape[0]:
        raise RuntimeError(f"C mismatch logits vs mu_c: {logits.shape[1]} vs {mu_c.shape[0]}")

    pred = torch.argmax(logits, dim=1).long()     # [N]
    z = feats - mu_c[pred]                        # [N, D]
    coeff = z @ Vp                                # [N, p]
    s = (coeff ** 2).sum(dim=1)                   # [N]

    s1 = s.detach().cpu().numpy().astype(np.float64)
    ratio = np.ones_like(s1, dtype=np.float64)
    return s1, ratio


# =========================
# 3) End to end runner for this method
# =========================
def run_method_subspaces(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    id_feats: torch.Tensor,
    id_logits: torch.Tensor,
    ood_feats: torch.Tensor,
    ood_logits: torch.Tensor,
    num_classes: int,
    energy_keep: float,
    k_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    End to end:
      fit pooled within class PCA on train_feats
      score ID and OOD using predicted class from logits
    """
    subspaces = fit_affine_pca_subspaces(
        feats=train_feats,
        labels=train_labels,
        num_classes=num_classes,
        energy_keep=energy_keep,
        k_max=k_max,
    )

    s1_id, _  = score_subspaces_ratio_min(id_feats,  id_logits,  subspaces)
    s1_ood, _ = score_subspaces_ratio_min(ood_feats, ood_logits, subspaces)

    return s1_id, s1_ood




# def run_method_subspaces(
#     train_feats: torch.Tensor,
#     train_labels: torch.Tensor,
#     id_feats: torch.Tensor,
#     ood_feats: torch.Tensor,
#     num_classes: int,
#     energy_keep: float,
#     k_max: int,
#     W: torch.Tensor,
#     b: torch.Tensor,
#     threshold: float = 10.0,
#     alpha: float = 1.0,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     subspaces = fit_affine_pca_subspaces(
#         feats=train_feats,
#         labels=train_labels,
#         num_classes=num_classes,
#         energy_keep=energy_keep,
#         k_max=k_max,
#     )

#     s1_id, _ = score_subspaces_ratio_min(id_feats, subspaces)
#     s1_ood, _ = score_subspaces_ratio_min(ood_feats, subspaces)

#     with torch.no_grad():
#         Xid = torch.clamp(id_feats.float(), max=threshold)
#         Xod = torch.clamp(ood_feats.float(), max=threshold)
#         logits_id = Xid @ W.t() + b.unsqueeze(0)
#         logits_ood = Xod @ W.t() + b.unsqueeze(0)
#         e_id = (-torch.logsumexp(logits_id, dim=1)).cpu().numpy().astype(np.float64)
#         e_ood = (-torch.logsumexp(logits_ood, dim=1)).cpu().numpy().astype(np.float64)

#     fused_id = e_id + alpha * s1_id
#     fused_ood = e_ood + alpha * s1_ood
#     return fused_id, fused_ood



@torch.no_grad()
def score_subspaces_top2pred_s1(
    feats: torch.Tensor,
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    W: torch.Tensor,  # (C,d)
    b: torch.Tensor,  # (C,)
    eps: float = 1e-6,
) -> np.ndarray:
    feats = feats.float()
    classes = sorted(subspaces.keys())
    C = len(classes)

    # logits for class selection
    logits = feats @ W.t() + b.unsqueeze(0)  # (N,C)
    top2 = torch.topk(logits, k=2, dim=1, largest=True).indices  # (N,2)

    # pre-stack per-class params for fast gather
    mu = torch.stack([subspaces[c]["mu"] for c in classes], dim=0)          # (C,d)
    Vp = [subspaces[c]["V_perp"] for c in classes]                           # list of (d,pc)
    lam = [subspaces[c]["lam_perp"] for c in classes]                        # list of (pc,)

    N = feats.shape[0]
    s_best = torch.empty((N,), dtype=torch.float32)

    # compute min score over the 2 selected classes (loop over 2 only)
    for i in range(N):
        c1, c2 = int(top2[i, 0]), int(top2[i, 1])
        z1 = feats[i] - mu[c1]
        z2 = feats[i] - mu[c2]

        coeff1 = z1 @ Vp[c1]
        coeff2 = z2 @ Vp[c2]

        s1 = ((coeff1 * coeff1) / (lam[c1] + eps)).sum() / max(Vp[c1].shape[1], 1)
        s2 = ((coeff2 * coeff2) / (lam[c2] + eps)).sum() / max(Vp[c2].shape[1], 1)

        s_best[i] = torch.minimum(s1, s2)

    return s_best.cpu().numpy().astype(np.float64)


def run_method_subspaces_top2pred(
    train_feats, train_labels, id_feats, ood_feats, num_classes, energy_keep, k_max, W, b,
):
    subspaces = fit_affine_pca_subspaces(train_feats, train_labels, num_classes, energy_keep, k_max)
    id_scores = score_subspaces_top2pred_s1(id_feats, subspaces, W=W, b=b, eps=1e-6)
    ood_scores = score_subspaces_top2pred_s1(ood_feats, subspaces, W=W, b=b, eps=1e-6)
    return id_scores, ood_scores


# -------------------------
# Method: Mahalanobis
# -------------------------
def fit_shared_gaussian(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = feats.float()
    labels = labels.long()

    means = []
    centered_all = []
    for c in range(num_classes):
        Xc = feats[labels == c]
        mu = Xc.mean(dim=0)
        means.append(mu)
        centered_all.append(Xc - mu)

    means = torch.stack(means, dim=0)
    X = torch.cat(centered_all, dim=0)

    D = feats.shape[1]
    cov = (X.T @ X) / max(X.shape[0] - 1, 1)
    cov = cov + eps * torch.eye(D)
    precision = torch.linalg.inv(cov)
    return means, precision


@torch.no_grad()
def scores_mahalanobis_min(
    feats: torch.Tensor,
    means: torch.Tensor,
    precision: torch.Tensor,
) -> np.ndarray:
    feats = feats.float()
    means = means.float()
    precision = precision.float()

    dists = []
    for c in range(means.shape[0]):
        z = feats - means[c].unsqueeze(0)
        q = (z @ precision) * z
        d = q.sum(dim=1)
        dists.append(d)

    dists = torch.stack(dists, dim=1)
    dmin = dists.min(dim=1).values
    return dmin.cpu().numpy().astype(np.float64)


def run_method_mahalanobis(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    id_feats: torch.Tensor,
    ood_feats: torch.Tensor,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    means, precision = fit_shared_gaussian(train_feats, train_labels, num_classes=num_classes)
    id_scores = scores_mahalanobis_min(id_feats, means, precision)
    ood_scores = scores_mahalanobis_min(ood_feats, means, precision)
    return id_scores, ood_scores


# -------------------------
# Method: Energy / MSP from logits computed via (feat @ W.T + b)
# -------------------------
def logits_from_feats(feats: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    X = feats.float()
    return X @ W.t() + b.unsqueeze(0)


def scores_energy_from_logits(logits: torch.Tensor, T: float = 1.0) -> np.ndarray:
    logits = logits.float()
    e = -T * torch.logsumexp(logits / T, dim=1)
    return e.cpu().numpy().astype(np.float64)


def scores_msp_from_logits(logits: torch.Tensor) -> np.ndarray:
    p = F.softmax(logits.float(), dim=1)
    msp = p.max(dim=1).values
    return (-msp).cpu().numpy().astype(np.float64)


# -------------------------
# Method: logit_gate
# -------------------------
def fit_global_pca_basis(
    feats: torch.Tensor,
    energy_keep: float = 0.95,
    k_max: int = 64,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = feats.float()
    mean = X.mean(dim=0)
    Z = X - mean

    _, S, Vh = torch.linalg.svd(Z, full_matrices=False)
    V = Vh.transpose(0, 1)
    n = Z.shape[0]
    lam = (S**2) / max(n - 1, 1)

    energy = torch.cumsum(lam, dim=0) / (lam.sum() + eps)
    k = int(torch.searchsorted(energy, torch.tensor(energy_keep)).item()) + 1
    r = V.shape[1]
    k = min(k, k_max, r)

    Vk = V[:, :k].contiguous()
    return mean.contiguous(), Vk


@torch.no_grad()
def scores_logit_gate(
    feats: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    feat_mean: torch.Tensor,
    Vk: torch.Tensor,
    threshold: float = 10.0,
    eps: float = 1e-6,
) -> np.ndarray:
    X = feats.float()
    Xc = torch.clamp(X, max=threshold)
    logits = Xc @ W.t() + b.unsqueeze(0)

    Z = X - feat_mean.unsqueeze(0)
    proj = (Z @ Vk) @ Vk.t()
    residual = Z - proj
    rec_norm = torch.linalg.norm(residual, dim=1)

    denom = torch.linalg.norm(Xc, dim=1).clamp_min(eps)
    r = rec_norm / denom

    lse = torch.logsumexp(logits, dim=1)
    score_raw = lse * (1.0 - r)
    return (-score_raw).cpu().numpy().astype(np.float64)


# -------------------------
# Method: kpca_rff (RFF + linear PCA + reconstruction error)
# -------------------------
def _l2_normalize_np(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    denom = np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + eps
    return (x / denom).astype(np.float32, copy=False)


def _rff_map_np(x: np.ndarray, W: np.ndarray, u: np.ndarray) -> np.ndarray:
    M = W.shape[0]
    return (np.sqrt(2.0 / M) * np.cos(x.dot(W.T) + u[np.newaxis, :])).astype(np.float32, copy=False)


def fit_kpca_rff(
    train_feats: torch.Tensor,
    gamma: float,
    M: int,
    exp_var_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    X = train_feats.cpu().numpy().astype(np.float32, copy=False)
    X = np.ascontiguousarray(_l2_normalize_np(X))

    m = X.shape[1]
    rng = np.random.RandomState(seed)
    W = np.sqrt(2.0 * gamma) * rng.normal(size=(M, m)).astype(np.float32)
    u = (2.0 * np.pi * rng.rand(M)).astype(np.float32)

    Xmap = _rff_map_np(X, W, u)

    mu = Xmap.mean(axis=0)
    Xmap = Xmap - mu

    K = Xmap.T.dot(Xmap)
    U_full, s, _ = np.linalg.svd(K, full_matrices=False)

    s_accuml = np.zeros(U_full.shape[1], dtype=np.float64)
    denom = float(np.sum(s)) if float(np.sum(s)) > 0 else 1.0
    q = -1
    for i in range(U_full.shape[1]):
        s_accuml[i] = float(np.sum(s[:i])) / denom
        if i > 0 and q < 0:
            if s_accuml[i - 1] < exp_var_ratio and s_accuml[i] >= exp_var_ratio:
                q = i
                break
    if q < 0:
        q = min(1, U_full.shape[1])

    Uq = U_full[:, :q].astype(np.float32, copy=False)
    return mu.astype(np.float32, copy=False), Uq, q


def scores_kpca_rff(
    query_feats: torch.Tensor,
    mu: np.ndarray,
    Uq: np.ndarray,
    gamma: float,
    M: int,
    seed: int,
) -> np.ndarray:
    Xq = query_feats.cpu().numpy().astype(np.float32, copy=False)
    Xq = np.ascontiguousarray(_l2_normalize_np(Xq))

    m = Xq.shape[1]
    rng = np.random.RandomState(seed)
    W = np.sqrt(2.0 * gamma) * rng.normal(size=(M, m)).astype(np.float32)
    u = (2.0 * np.pi * rng.rand(M)).astype(np.float32)

    Xmap = _rff_map_np(Xq, W, u)
    Xmap = Xmap - mu

    proj = (Xmap.dot(Uq)).dot(Uq.T)
    err = np.linalg.norm(Xmap - proj, ord=2, axis=1)
    return err.astype(np.float64, copy=False)


# -------------------------
# Method: gradsubspace (KL-to-uniform mixed target)
def fit_id_feature_subspace(
    train_feats: torch.Tensor,
    n_batch: int,
    exp_var_ratio: float,
    center: bool,
    seed: int,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """
    Build ID subspace S from a random minibatch of ID train features.

    Returns:
      S  : (D, k) right singular vectors (orthonormal columns)
      mu : (1, D) mean used for centering (None if center=False)
      k  : chosen subspace dimension
    """
    X = train_feats.float()
    N, D = X.shape
    rng = np.random.RandomState(seed)
    n = int(min(n_batch, N))
    idx = rng.choice(N, size=n, replace=False)

    B = X[idx]  # (n, D)

    mu = None
    if center:
        mu = B.mean(dim=0, keepdim=True)  # (1, D)
        B = B - mu

    _, Svals, Vh = torch.linalg.svd(B, full_matrices=False)
    V = Vh.transpose(0, 1)  # (D, r)

    lam = (Svals ** 2)
    cum = torch.cumsum(lam, dim=0) / (lam.sum() + eps)
    k = int(torch.searchsorted(cum, torch.tensor(exp_var_ratio, device=cum.device)).item()) + 1
    k = max(1, min(k, V.shape[1]))

    S = V[:, :k].contiguous()
    return S, mu, k


@torch.no_grad()
def scores_gradsubspace_pseudo_resid(
    feats: torch.Tensor,          # (N, D) CPU
    W: torch.Tensor,              # (C, D) CPU
    b: torch.Tensor,              # (C,) CPU
    S: torch.Tensor,              # (D, k) CPU
    mu: Optional[torch.Tensor],   # (1, D) mean used when fitting S, or None
    num_classes: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Label-free version (same scoring for ID and OOD).

    a(x): uncertainty = 1 - max softmax prob
    r(x): residual norm to ID subspace (computed in same centered space used to fit S)

    score(x) = a(x) * r(x)

    Convention: higher score => more OOD-like.
    """
    X = feats.float()
    logits = X @ W.t() + b.unsqueeze(0)
    p = F.softmax(logits, dim=1)

    N, C = p.shape
    assert C == num_classes, f"p has {C} classes, expected {num_classes}"

    # uncertainty term (high when model is unsure)
    a = 1.0 - p.max(dim=1).values
    a = a.clamp_min(eps)

    # residual term computed in same space as subspace fit
    Xr = X - mu if mu is not None else X
    f_proj = (Xr @ S) @ S.t()
    f_resid = Xr - f_proj
    r = torch.linalg.norm(f_resid, dim=1).clamp_min(eps)

    score = a * r
    return score.cpu().numpy().astype(np.float64)














@torch.no_grad()
def residual_matrix_per_class(
    feats: torch.Tensor,
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    eps: float = 1e-6,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Returns residual scores matrix R of shape (N, C), lower is better (more ID-like).
    This matches your lam_eff + /Vp.shape[1] convention.
    """
    feats = feats.float()
    classes = sorted(subspaces.keys())
    per_class = []

    for c in classes:
        mu = subspaces[c]["mu"]
        Vp = subspaces[c]["V_perp"]
        lam = subspaces[c]["lam_perp"]

        z = feats - mu.unsqueeze(0)
        coeff = z @ Vp

        lam_eff = (1 - alpha) * lam + alpha * lam.mean()
        s = (coeff**2) / (lam_eff.unsqueeze(0) + eps)
        s = s.sum(dim=1) / float(max(Vp.shape[1], 1))
        per_class.append(s)

    return torch.stack(per_class, dim=1)  # (N,C)

@torch.no_grad()
def score_subspace_entropy(
    feats: torch.Tensor,
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    eps: float = 1e-6,
    temp: float = 1.0,
) -> np.ndarray:
    R = residual_matrix_per_class(feats, subspaces, eps=eps)  # (N,C)
    T = float(max(temp, 1e-8))
    P = torch.softmax(-R / T, dim=1)                          # (N,C)
    H = -(P * (P.clamp_min(1e-12).log())).sum(dim=1)          # entropy
    return H.cpu().numpy().astype(np.float64)

@torch.no_grad()
def score_subspace_negmaxprob(
    feats: torch.Tensor,
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    eps: float = 1e-6,
    temp: float = 1.0,
) -> np.ndarray:
    R = residual_matrix_per_class(feats, subspaces, eps=eps)
    T = float(max(temp, 1e-8))
    P = torch.softmax(-R / T, dim=1)
    pmax = P.max(dim=1).values
    return (-pmax).cpu().numpy().astype(np.float64)  # higher => more OOD

@torch.no_grad()
def fit_conformal_residuals(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    eps: float = 1e-6,
) -> Dict[int, np.ndarray]:
    """
    For each class c, store the residuals r_c(x) for train points with label c.
    """
    R = residual_matrix_per_class(train_feats, subspaces, eps=eps)  # (N,C)
    y = train_labels.long()
    conf = {}
    for c in sorted(subspaces.keys()):
        rc = R[y == c, c].detach().cpu().numpy().astype(np.float64)
        rc.sort()
        conf[c] = rc
    return conf

@torch.no_grad()
def score_conformal_pmax(
    feats: torch.Tensor,
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    conf: Dict[int, np.ndarray],
    eps: float = 1e-6,
) -> np.ndarray:
    """
    p_c(x) = (#{train rc >= r_c(x)} + 1) / (n_c + 1)
    pmax = max_c p_c(x)
    score = 1 - pmax  (higher => more OOD)
    """
    R = residual_matrix_per_class(feats, subspaces, eps=eps)  # (N,C)
    Rn = R.detach().cpu().numpy().astype(np.float64)          # (N,C)

    classes = sorted(subspaces.keys())
    N = Rn.shape[0]
    pmax = np.zeros(N, dtype=np.float64)

    for ci, c in enumerate(classes):
        rc_train = conf[c]          # sorted array
        n = rc_train.size
        # count how many train residuals are >= r (using searchsorted on sorted asc)
        # idx = first position where rc_train >= r  => count_ge = n - idx
        idx = np.searchsorted(rc_train, Rn[:, ci], side="left")
        count_ge = n - idx
        p = (count_ge + 1.0) / (n + 1.0)
        pmax = np.maximum(pmax, p)

    score = 1.0 - pmax
    return score
def run_method_subspaces_detector(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    id_feats: torch.Tensor,
    ood_feats: torch.Tensor,
    num_classes: int,
    energy_keep: float,
    k_max: int,
    mode: str = "conformal",     # "entropy" | "negmaxprob" | "conformal"
    eps: float = 1e-6,
    temp: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    subspaces = fit_affine_pca_subspaces(
        feats=train_feats,
        labels=train_labels,
        num_classes=num_classes,
        energy_keep=energy_keep,
        k_max=k_max,
        eps=eps,
    )

    mode = mode.lower()
    if mode == "entropy":
        id_scores = score_subspace_entropy(id_feats, subspaces, eps=eps, temp=temp)
        ood_scores = score_subspace_entropy(ood_feats, subspaces, eps=eps, temp=temp)
        return id_scores, ood_scores

    if mode == "negmaxprob":
        id_scores = score_subspace_negmaxprob(id_feats, subspaces, eps=eps, temp=temp)
        ood_scores = score_subspace_negmaxprob(ood_feats, subspaces, eps=eps, temp=temp)
        return id_scores, ood_scores

    if mode == "conformal":
        conf = fit_conformal_residuals(train_feats, train_labels, subspaces, eps=eps)
        id_scores = score_conformal_pmax(id_feats, subspaces, conf, eps=eps)
        ood_scores = score_conformal_pmax(ood_feats, subspaces, conf, eps=eps)
        return id_scores, ood_scores

    raise ValueError(f"Unknown mode: {mode}")
