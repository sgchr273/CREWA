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
from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance


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
# 2) Score: predicted class scoring (highlighted method)
# =========================
# @torch.no_grad()
# def score_subspaces_ratio_min(
#     feats: torch.Tensor,
#     logits: torch.Tensor,
#     subspaces: Dict[int, Dict[str, torch.Tensor]],
#     eps: float = 1e-6,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Predicted class scoring:
#       c_hat = argmax(logits)
#       z = x - mu_{c_hat}
#       s = || V_perp^T z ||^2  (same residual energy you use now)

#     Returns:
#       s1: numpy scores (higher means more OOD if you treat residual as OOD score)
#       ratio: placeholder ones to keep return type stable
#     """
#     feats = feats.float()
#     logits = logits.float()

#     m = subspaces[0] if 0 in subspaces else subspaces[next(iter(subspaces.keys()))]

#     if "mu_c" not in m:
#         raise KeyError("expected pooled within class fit: subspaces[0] must contain 'mu_c'")

#     mu_c = m["mu_c"]         # [C, D]
#     Vp = m["V_perp"]         # [D, p]
#     # lam = m["lam_perp"]    # available if you want whitening later

#     if feats.ndim != 2:
#         raise RuntimeError(f"feats must be [N, D], got {tuple(feats.shape)}")
#     if logits.ndim != 2:
#         raise RuntimeError(f"logits must be [N, C], got {tuple(logits.shape)}")
#     if feats.shape[0] != logits.shape[0]:
#         raise RuntimeError(f"N mismatch feats vs logits: {feats.shape[0]} vs {logits.shape[0]}")
#     if feats.shape[1] != mu_c.shape[1]:
#         raise RuntimeError(f"D mismatch feats vs mu_c: {feats.shape[1]} vs {mu_c.shape[1]}")
#     if logits.shape[1] != mu_c.shape[0]:
#         raise RuntimeError(f"C mismatch logits vs mu_c: {logits.shape[1]} vs {mu_c.shape[0]}")

#     pred = torch.argmax(logits, dim=1).long()     # [N]
#     z = feats - mu_c[pred]                        # [N, D]
#     coeff = z @ Vp                                # [N, p]
#     s = (coeff ** 2).sum(dim=1)                   # [N]

#     s1 = s.detach().cpu().numpy().astype(np.float64)
#     ratio = np.ones_like(s1, dtype=np.float64)
#     return s1, ratio


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


def run_crewa(
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


# -------------------------

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
# 2) Components: s_res, align, distance-feature
# =========================
@torch.no_grad()
def _subspace_components(
    feats: torch.Tensor,
    logits: torch.Tensor,
    W: torch.Tensor,                      # [C, D]
    b: torch.Tensor,                      # [C]
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    *,
    use_centered_for_cos: bool = False,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feats = feats.float()
    logits = logits.float()
    W = W.float()

    m = subspaces[0] if 0 in subspaces else subspaces[next(iter(subspaces.keys()))]
    if "mu_c" not in m:
        raise KeyError("expected pooled within class fit: subspaces[0] must contain 'mu_c'")

    mu_c = m["mu_c"]    # [C, D]
    Vp   = m["V_perp"]  # [D, p]
    lam_perp = m["lam_perp"]  # [p] available if you want whitening later

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

    # pred = torch.argmax(logits, dim=1).long()     # [N]
    pred = torch.argmax(feats @ W.T , dim=1).long()
    diff = (logits - (feats @ W.T)).abs().max()
    print(diff)
    # lam_min = lam_perp.min().item()
    # lam_max = lam_perp.max().item()
    # lam_med = lam_perp.median().item()

    # print("lam_perp min:", lam_min)
    # print("lam_perp median:", lam_med)
    # print("lam_perp max:", lam_max)



    z = feats - mu_c[pred]                         # [N, D]

    # residual energy in discarded space
    coeff = z @ Vp                                 # [N, p]
    lam = lam_perp.detach()
    # den = torch.where(lam < 1e-4, torch.full_like(lam, 1e-5), torch.full_like(lam, 1e-4))
    # # => explicit: either 1e-5 or 1e-4

    # s_res = (coeff.pow(2) / den).sum(dim=1)  
    # print("frac < 1e-4:", (lam_perp < 1e-4).float().mean().item())
    # print("frac >=1e-4:", (lam_perp >= 1e-4).float().mean().item())
 
    s_res = (coeff ** 2).sum(dim=1)                # [N]

    # distance to predicted class mean (log1p for stability)
    d2_pred = (z ** 2).sum(dim=1)                  # [N]
    d_feat = torch.log1p(d2_pred)                  # [N]

    # alignment penalty
    u = z if use_centered_for_cos else feats
    u_n = F.normalize(u, p=2, dim=1, eps=eps)
    W_n = F.normalize(W, p=2, dim=1, eps=eps)
    w_pred = W_n[pred]
    cos = (u_n * w_pred).sum(dim=1).clamp(-1.0, 1.0)
    a = 1.0 - cos

    return s_res, a, d_feat


# =========================
# 3) Score: residual + beta*align + alpha*distance
# =========================
@torch.no_grad()
def score_subspaces_resid_plus_align(
    feats: torch.Tensor,
    logits: torch.Tensor,
    W: torch.Tensor,                      # [C, D]
    b: torch.Tensor,                      # [C]
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    *,
    beta: float = 1.0,
    alpha: float = 0.0,
    use_centered_for_cos: bool = False,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score = s_res + beta * align + alpha * log(1 + ||x - mu_pred||^2)

    Returns:
      s_total: numpy scores (float64)
      a:       alignment penalty (float64) (kept for signature stability)
    """
    s_res, a, d_feat = _subspace_components(
        feats, logits, W,b, subspaces,
        use_centered_for_cos=use_centered_for_cos,
        eps=eps,
    )

    s_total = s_res + float(beta) * a + float(alpha) * d_feat
    s_np = s_total.detach().cpu().numpy().astype(np.float64)
    a_np = a.detach().cpu().numpy().astype(np.float64)
    return s_np, a_np


# =========================
# 4) Run method + tune beta/alpha on ID + correlation print
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
    *,
    W: torch.Tensor,                 # [C, D]
    b: torch.Tensor,                 # [C]
    beta: float = 1.0,
    alpha: float = 0.0,
    use_centered_for_cos: bool = False,
    # tuning flags
    tune_beta: bool = False,
    tune_alpha: bool = False,
    # tuning multipliers (ID-only)
    gamma: float = 1.0,   # beta multiplier
    delta: float = 0.25,  # alpha multiplier (recommended start: 0.25)
    # diagnostics
    print_corr: bool = True,
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

    # ---- diagnostics: redundancy correlation on ID ----
    # if print_corr:
    #     with torch.no_grad():
    #         s_res_t, _, d_feat_t = _subspace_components(
    #             id_feats, id_logits, W, subspaces,
    #             use_centered_for_cos=use_centered_for_cos,
    #             eps=eps,
    #         )
    #         a_cpu = s_res_t.detach().cpu().double()
    #         b_cpu = d_feat_t.detach().cpu().double()
    #         corr = torch.corrcoef(torch.stack([a_cpu, b_cpu], dim=0))[0, 1].item()
    #         print(f"[redundancy] corr(ID) between s_res and log1p(||x-mu_pred||^2) = {corr:.4f}")

    # ---- tune beta / alpha on ID only (median matching) ----
    beta_used = float(beta)
    alpha_used = float(alpha)

    if tune_beta or tune_alpha:
        with torch.no_grad():
            s_res_t, a_t, d_feat_t = _subspace_components(
                id_feats, id_logits, W, b,subspaces,
                use_centered_for_cos=use_centered_for_cos,
                eps=eps,
            )
            s_res_np = s_res_t.detach().cpu().numpy().astype(np.float64)
            a_np     = a_t.detach().cpu().numpy().astype(np.float64)
            d_np     = d_feat_t.detach().cpu().numpy().astype(np.float64)

            med_res = float(np.median(s_res_np))

            if tune_beta:
                med_a = float(np.median(a_np))
                beta_used = 0.0 if med_a < 1e-12 else float(gamma) * (med_res / med_a)

            if tune_alpha:
                med_d = float(np.median(d_np))
                alpha_used = 0.0 if med_d < 1e-12 else float(delta) * (med_res / med_d)

    print(f"[subspaces] beta_used={beta_used:.6g} (tune_beta={tune_beta}, gamma={gamma})")
    print(f"[subspaces] alpha_used={alpha_used:.6g} (tune_alpha={tune_alpha}, delta={delta})")

    # ---- score ID and OOD with the same calibrated weights ----
    s_id, _ = score_subspaces_resid_plus_align(
        id_feats, id_logits, W, b, subspaces,
        beta=beta_used,
        alpha=alpha_used,
        use_centered_for_cos=use_centered_for_cos,
        eps=eps,
    )
    s_ood, _ = score_subspaces_resid_plus_align(
        ood_feats, ood_logits, W,b, subspaces,
        beta=beta_used,
        alpha=alpha_used,
        use_centered_for_cos=use_centered_for_cos,
        eps=eps,
    )
    return s_id, s_ood


@torch.no_grad()
def get_all_labels_torch(loader, device=None) -> torch.Tensor:
    ys = []
    for _, y in loader:
        ys.append(y)
    y = torch.cat(ys, dim=0).long()
    return y.to(device) if device is not None else y

def get_classifier_weight_matrix_torch(model: nn.Module) -> torch.Tensor:
    """
    Return the final classifier weight matrix W with shape [C, D] for:
      - ResNet18 / ResNet50 (torchvision): model.fc.weight
      - ViT-B/16 (torchvision): model.heads.head.weight (or model.head.weight in some wrappers)
      - Swin-T (torchvision): model.head.weight

    Notes:
      - Returns the Parameter tensor directly (on whatever device/dtype it lives on).
      - Works with common "WithFeats" wrappers as long as the underlying attribute exists.
    """
    # --- ResNet family (resnet18/resnet50) ---
    if hasattr(model, "fc") and isinstance(getattr(model, "fc"), nn.Linear):
        return model.fc.weight  # [C, D]

    # --- Swin-T (torchvision.swin_t) ---
    # torchvision Swin has a top-level Linear 'head'
    if hasattr(model, "head") and isinstance(getattr(model, "head"), nn.Linear):
        return model.head.weight  # [C, D]

    # --- ViT-B/16 (torchvision.vit_b_16) ---
    # torchvision ViT typically has model.heads.head as the classifier Linear
    if hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
        return model.heads.head.weight  # [C, D]

    # Some wrappers / variants expose classifier as 'classifier'
    if hasattr(model, "classifier") and isinstance(getattr(model, "classifier"), nn.Linear):
        return model.classifier.weight  # [C, D]

    # Some third-party or wrapper variants expose 'head' but not as nn.Linear directly
    # (e.g., head is a Sequential containing a Linear). Try to find the last Linear.
    for attr in ("head", "heads", "classifier", "fc"):
        if hasattr(model, attr):
            m = getattr(model, attr)
            if isinstance(m, nn.Sequential):
                linears = [layer for layer in m.modules() if isinstance(layer, nn.Linear)]
                if len(linears) > 0:
                    return linears[-1].weight  # [C, D]

    raise ValueError(
        "Could not locate final linear classifier weights. "
        "Checked: fc, head, heads.head, classifier (and Sequential fallbacks)."
    )
@torch.no_grad()
def l2_normalize_torch(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    n = torch.linalg.norm(x, dim=dim, keepdim=True).clamp_min(eps)
    return x / n

@torch.no_grad()
def compute_class_means_torch(
    train_feats: torch.Tensor,   # [N, D]
    train_labels: torch.Tensor,  # [N]
    num_classes: int,
) -> torch.Tensor:
    """
    Returns means [C, D]. Uses scatter add for speed.
    If a class has zero count, its mean stays 0.
    """
    device = train_feats.device
    train_feats = train_feats.float()
    train_labels = train_labels.long()

    N, D = train_feats.shape
    means = torch.zeros((num_classes, D), device=device, dtype=torch.float32)

    counts = torch.bincount(train_labels, minlength=num_classes).float().to(device)  # [C]
    means.index_add_(0, train_labels, train_feats)  # sum per class

    denom = counts.clamp_min(1.0).unsqueeze(1)      # avoid divide by 0
    means = means / denom
    return means

@torch.no_grad()
def infer_pseudo_labels_by_cosine_torch(
    feats: torch.Tensor,  # [N, D]
    W: torch.Tensor,      # [C, D]
) -> torch.Tensor:
    feats_n = l2_normalize_torch(feats.float(), dim=1)
    W_n = l2_normalize_torch(W.float(), dim=1)
    sims = feats_n @ W_n.t()   # [N, C]
    return sims.argmax(dim=1).long()

@torch.no_grad()
def nc_centered_alignment_scores_torch(
    feats: torch.Tensor,   # [N, D]
    labels: torch.Tensor,  # [N]
    means: torch.Tensor,   # [C, D]
    W: torch.Tensor,       # [C, D]
    *,
    use_norm: bool = True,
) -> torch.Tensor:
    """
    score(x) = ||h||2 * (1 - cos(h, w_c))   if use_norm
              = (1 - cos(h, w_c))          if not use_norm
    where h = f(x) - mu_c
    Larger score means more OOD.
    """
    device = feats.device
    feats = feats.float()
    labels = labels.long().to(device)
    means = means.float().to(device)
    W = W.float().to(device)

    h = feats - means.index_select(0, labels)              # [N, D]
    eps = 1e-12
    h_norm = torch.linalg.norm(h, dim=1).clamp_min(eps)    # [N]
    h_n = h / h_norm.unsqueeze(1)                          # [N, D]

    W_n = l2_normalize_torch(W, dim=1)                     # [C, D]
    wc = W_n.index_select(0, labels)                       # [N, D]
    cos = (h_n * wc).sum(dim=1)                            # [N]

    penalty = 1.0 - cos
    score = h_norm * penalty if use_norm else penalty
    return score

# -------------------------
# Full method: compute S_id and S_ood in torch
# -------------------------
@torch.no_grad()
def compute_nc_subtracted_scores(
    *,
    train_feats: torch.Tensor,        # [Ntr, D] torch
    train_labels: torch.Tensor,       # [Ntr]   torch
    id_feats: torch.Tensor,           # [Nid, D] torch
    id_loader,                        # to read id test labels
    ood_feats: torch.Tensor,          # [Nood, D] torch
    clf_model: nn.Module,
    num_classes: int,
    device: torch.device,
    use_norm: bool = True,
):
    """
    Returns:
      S_id  torch.Tensor [Nid]
      S_ood torch.Tensor [Nood]
    """
    train_feats = train_feats.to(device).float()
    id_feats = id_feats.to(device).float()
    ood_feats = ood_feats.to(device).float()

    if not torch.is_tensor(train_labels):
        train_labels = torch.as_tensor(train_labels)
    train_labels = train_labels.to(device).long()

    # ID test labels from loader (ground truth)
    # id_labels = get_all_labels_torch(id_loader, device=device)  # [Nid]
    

    # class means from ID train
    means = compute_class_means_torch(train_feats, train_labels, num_classes=num_classes)  # [C, D]

    # classifier weights
    W = get_classifier_weight_matrix_torch(clf_model).detach().to(device).float()  # [C, D]
    id_pseudo = infer_pseudo_labels_by_cosine_torch(id_feats, W)
    # ID score uses ground truth ID labels
    S_id = nc_centered_alignment_scores_torch(id_feats, id_pseudo, means, W, use_norm=use_norm)

    # OOD score uses pseudo label (argmax cosine with W), then subtract that mean
    ood_pseudo = infer_pseudo_labels_by_cosine_torch(ood_feats, W)
    S_ood = nc_centered_alignment_scores_torch(ood_feats, ood_pseudo, means, W, use_norm=use_norm)

    return S_id, S_ood






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


def run_method_neco(
    train_feats,   # torch.Tensor [Ntrain, D]
    id_feats,      # torch.Tensor [Nid, D]
    ood_feats,     # torch.Tensor [Nood, D]
    W,             # torch.Tensor [C, D]
    b,             # torch.Tensor [C]
    arch: str,
    neco_dim: int,
    eps: float = 1e-12,
    use_scaler=None,   # None: auto, True or False: force
):
    """
    NECO scores only (no AUROC/FPR computed here).

    Returns:
      score_id:  np.ndarray [Nid]
      score_ood: np.ndarray [Nood]
    """

    arch_l = (arch or "").lower()
    is_resnet = ("resnet" in arch_l)

    if use_scaler is None:
        # typical behavior: scaler helps for resnet features; often unnecessary for ViT style
        use_scaler = is_resnet

    # torch -> numpy (sklearn works on numpy)
    Xtr = train_feats.detach().float().cpu().numpy()
    Xid = id_feats.detach().float().cpu().numpy()
    Xod = ood_feats.detach().float().cpu().numpy()

    if use_scaler:
        ss = StandardScaler()
        Xtr_fit = ss.fit_transform(Xtr)
        Xid_fit = ss.transform(Xid)
        Xod_fit = ss.transform(Xod)
    else:
        Xtr_fit, Xid_fit, Xod_fit = Xtr, Xid, Xod

    D = Xtr_fit.shape[1]
    k = int(neco_dim)
    if k <= 0 or k > D:
        raise ValueError(f"neco_dim must be in [1, {D}], got {k}")

    # PCA fit on ID train
    pca = PCA(n_components=D)
    pca.fit(Xtr_fit)

    Zid = pca.transform(Xid_fit)[:, :k]
    Zod = pca.transform(Xod_fit)[:, :k]

    # ratio of norms (vectorized)
    score_id = np.linalg.norm(Zid, axis=1) / (np.linalg.norm(Xid_fit, axis=1) + eps)
    score_ood = np.linalg.norm(Zod, axis=1) / (np.linalg.norm(Xod_fit, axis=1) + eps)

    # logits from head for the optional max-logit scaling (original NECO behavior)
    with torch.no_grad():
        logits_id = (id_feats.detach().float() @ W.detach().float().t()) + b.detach().float()
        logits_ood = (ood_feats.detach().float() @ W.detach().float().t()) + b.detach().float()
        maxlog_id = logits_id.max(dim=1).values.detach().cpu().numpy()
        maxlog_ood = logits_ood.max(dim=1).values.detach().cpu().numpy()

    # keep NECO behavior: multiply by max logit for non-resnet
    if not is_resnet:
        score_id = score_id * maxlog_id
        score_ood = score_ood * maxlog_ood

    return score_id, score_ood


@torch.no_grad()
def _ensure_WCb_shapes(W: torch.Tensor, b: torch.Tensor, feat_dim: int, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (W_cls, b_cls) where W_cls is [C, D] and b_cls is [C].
    Assumes logits = feats @ W_cls.T + b_cls.
    """
    W = W.detach().float()
    b = b.detach().float().view(-1)

    if W.ndim != 2:
        raise RuntimeError(f"W must be 2D, got {W.ndim}D")
    if b.ndim != 1:
        raise RuntimeError(f"b must be 1D, got {b.ndim}D")

    # common cases:
    # 1) W is [C, D]
    # 2) W is [D, C]
    if W.shape == (num_classes, feat_dim):
        W_cls = W
    elif W.shape == (feat_dim, num_classes):
        W_cls = W.t().contiguous()
    else:
        raise RuntimeError(
            f"Unexpected W shape {tuple(W.shape)} for feat_dim={feat_dim}, num_classes={num_classes}. "
            f"Expected {(num_classes, feat_dim)} or {(feat_dim, num_classes)}."
        )

    if b.shape[0] != num_classes:
        raise RuntimeError(f"Unexpected b shape {tuple(b.shape)}; expected [{num_classes}]")

    return W_cls, b


@torch.no_grad()
def vim_fit(
    train_feats: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    num_classes: int,
    D: int = 0,
    fit_max: int = 200_000,
    seed: int = 0,
    eps: float = 1e-6,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Fits ViM parameters:
      o (bias removing shift),
      U (principal subspace basis, [feat_dim, D]),
      alpha (matching constant).

    Notes:
      ViM assumes train_feats are the features that feed the final linear classifier.
      If you L2 normalize features in your extractor but the classifier uses unnormalized,
      ViM will be inconsistent. In that case run ViM with l2_normalize=False.
    """
    train_feats = train_feats.detach().float().to("cpu")  # keep fit on CPU by default
    N = int(train_feats.shape[0])
    feat_dim = int(train_feats.shape[1])

    W_cls, b_cls = _ensure_WCb_shapes(W, b, feat_dim=feat_dim, num_classes=num_classes)

    # Choose D if not provided
    if D is None or int(D) <= 0:
        # Paper suggestion: if feat_dim >= C, natural D = C; else choose in [N/3, 2N/3]
        if feat_dim >= num_classes:
            D = min(num_classes, feat_dim - 1)
        else:
            D = max(1, int(round(2.0 * feat_dim / 3.0)))
    D = int(D)
    if not (1 <= D < feat_dim):
        raise RuntimeError(f"D must satisfy 1 <= D < feat_dim, got D={D}, feat_dim={feat_dim}")

    # Subsample K examples uniformly for stability and speed
    K = min(int(fit_max), N)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    idx = torch.randperm(N, generator=g)[:K]
    X = train_feats[idx]  # [K, feat_dim]

    # o = -(W^T)^+ b ; with W_cls = W^T in paper notation (shape [C, feat_dim])
    # so o = - pinv(W_cls) @ b_cls   => [feat_dim]
    o = -torch.linalg.pinv(W_cls) @ b_cls

    # shift features to bias free coordinates
    Z = (X - o.view(1, -1)).to(device)  # [K, feat_dim]

    # PCA on covariance of Z:
    # cov = Z^T Z / (K-1), eigenvectors sorted descending
    cov = (Z.t() @ Z) / max(K - 1, 1)   # [feat_dim, feat_dim]
    evals, evecs = torch.linalg.eigh(cov)  # ascending
    order = torch.argsort(evals, descending=True)
    evecs = evecs[:, order]
    U = evecs[:, :D].contiguous()  # [feat_dim, D], orthonormal basis of principal subspace

    # Residual norm without constructing R:
    # ||x_perp|| = sqrt( ||z||^2 - ||U^T z||^2 )
    proj = Z @ U                         # [K, D]
    proj_norm2 = (proj * proj).sum(dim=1)  # [K]
    z_norm2 = (Z * Z).sum(dim=1)           # [K]
    resid = torch.sqrt(torch.clamp(z_norm2 - proj_norm2, min=0.0) + eps)  # [K]

    # logits for the same sampled K points
    # logits = X @ W_cls.T + b
    logits = (X.to(device) @ W_cls.t().to(device)) + b_cls.to(device).view(1, -1)  # [K, C]
    max_logit = logits.max(dim=1).values  # [K]

    # alpha = sum max_logit / sum resid  (Eq. 6)
    alpha = (max_logit.sum() / (resid.sum() + eps)).detach()

    return {
        "o": o.detach().to("cpu"),        # [feat_dim]
        "U": U.detach().to("cpu"),        # [feat_dim, D]
        "alpha": alpha.detach().to("cpu"),# scalar
        "D": torch.tensor([D], dtype=torch.long),
    }


@torch.no_grad()
def vim_scores_from_feats(
    feats: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    vim_state: Dict[str, torch.Tensor],
    num_classes: int,
    eps: float = 1e-6,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Returns ViM score in the monotonic form:
      s(x) = alpha * ||x_perp|| - logsumexp(logits)
    Higher means more OOD.
    """
    feats = feats.detach().float().to(device)
    feat_dim = int(feats.shape[1])

    W_cls, b_cls = _ensure_WCb_shapes(W, b, feat_dim=feat_dim, num_classes=num_classes)
    W_cls = W_cls.to(device)
    b_cls = b_cls.to(device)

    o = vim_state["o"].to(device).view(1, -1)     # [1, D]
    U = vim_state["U"].to(device)                 # [D, k]
    alpha = float(vim_state["alpha"].item())

    Z = feats - o                                 # [N, feat_dim]

    # residual norm
    proj = Z @ U
    proj_norm2 = (proj * proj).sum(dim=1)
    z_norm2 = (Z * Z).sum(dim=1)
    resid = torch.sqrt(torch.clamp(z_norm2 - proj_norm2, min=0.0) + eps)

    # energy term from logits
    logits = (feats @ W_cls.t()) + b_cls.view(1, -1)
    energy = torch.logsumexp(logits, dim=1)

    s = (alpha * resid) - energy
    return s.detach().cpu().numpy().astype(np.float64)


def run_method_vim(
    train_feats: torch.Tensor,
    id_feats: torch.Tensor,
    ood_feats: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    num_classes: int,
    vim_dim: int = 0,
    vim_fit_max: int = 200_000,
    seed: int = 0,
    eps: float = 1e-6,
    fit_device: torch.device = torch.device("cpu"),
    score_device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    End to end ViM:
      1) fit o, principal subspace U, alpha on ID train feats
      2) score ID test feats and OOD feats

    Returns:
      s_id, s_ood  (higher means more OOD)
    """
    vim_state = vim_fit(
        train_feats=train_feats,
        W=W,
        b=b,
        num_classes=num_classes,
        D=vim_dim,
        fit_max=vim_fit_max,
        seed=seed,
        eps=eps,
        device=fit_device,
    )

    s_id = vim_scores_from_feats(
        feats=id_feats,
        W=W,
        b=b,
        vim_state=vim_state,
        num_classes=num_classes,
        eps=eps,
        device=score_device,
    )
    s_ood = vim_scores_from_feats(
        feats=ood_feats,
        W=W,
        b=b,
        vim_state=vim_state,
        num_classes=num_classes,
        eps=eps,
        device=score_device,
    )
    return s_id, s_ood



@torch.no_grad()
def compute_mu_global(train_feats: torch.Tensor) -> torch.Tensor:
    """
    mu_G = global mean feature vector, computed from ID train features.
    train_feats: (N, D) on CPU or GPU
    returns: (D,)
    """
    return train_feats.float().mean(dim=0)


def _vector_norm(x: torch.Tensor, ord: NormOrd, dim: int, eps: float) -> torch.Tensor:
    if ord == "inf" or ord == float("inf"):
        n = torch.linalg.vector_norm(x, ord=float("inf"), dim=dim)
    else:
        n = torch.linalg.vector_norm(x, ord=float(ord), dim=dim)
    return n.clamp_min(eps)


@torch.no_grad()
def nci_scores_batch(
    feats: torch.Tensor,
    W: torch.Tensor,
    b: Optional[torch.Tensor],
    mu_g: torch.Tensor,
    *,
    alpha: float = 0.0,
    p_norm: NormOrd = 1,
    use_bias: bool = True,
    eps: float = 1e-12,
    abs_pscore: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implements:
      pScore = cos(w_c, h - mu_G) * ||w_c||_2
            = ((h - mu_G) · w_c) / ||h - mu_G||_2
      detScore = pScore + alpha * ||h||_p

    Returns:
      det_score: (B,)  higher means more ID (per paper text)
      ood_score: (B,)  higher means more OOD (use this for AUROC with OOD positive)
      pred:      (B,)  predicted class indices
    """
    feats = feats.float()
    W = W.float()
    if b is not None:
        b = b.float()
    mu_g = mu_g.float()

    logits = feats @ W.t()
    if use_bias and (b is not None):
        logits = logits + b

    pred = torch.argmax(logits, dim=1)  # (B,)
    w_pred = W[pred]                    # (B, D)

    z = feats - mu_g.unsqueeze(0)       # (B, D)
    z_norm = _vector_norm(z, 2, dim=1, eps=eps)  # ||h - mu_G||_2

    # pScore = ((h - mu_G) · w_c) / ||h - mu_G||_2
    pscore = (z * w_pred).sum(dim=1) / z_norm
    if abs_pscore:
        pscore = pscore.abs()

    h_norm_p = _vector_norm(feats, p_norm, dim=1, eps=eps)  # ||h||_p
    det_score = pscore + float(alpha) * h_norm_p

    # Paper: lower det_score => more OOD. Most eval: higher => more OOD.
    ood_score = -det_score
    return det_score, ood_score, pred

@torch.no_grad()
def choose_alpha_id_only(train_feats, W, b, mu_g, p_norm=1, gamma=1.0, eps=1e-12, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pScore only (alpha=0)
    det_train, _ = nci_scores_batched(
        train_feats, W, b, mu_g, alpha=0.0, p_norm=p_norm,
        use_bias=True, batch_size=16384, device=device
    )  # det_train == pScore here

    h = train_feats.float()
    hnorm = torch.linalg.vector_norm(h, ord=float(p_norm), dim=1).clamp_min(eps)

    med_ps = det_train.abs().median().item()
    med_hn = hnorm.median().item()
    return float(gamma * (med_ps / (med_hn + eps)))


@torch.no_grad()
def nci_scores_batched(
    feats: torch.Tensor,
    W: torch.Tensor,
    b: Optional[torch.Tensor],
    mu_g: torch.Tensor,
    *,
    alpha: float = 0.0,
    p_norm: NormOrd = 1,
    use_bias: bool = True,
    eps: float = 1e-12,
    abs_pscore: bool = False,
    batch_size: int = 65536,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked scoring to avoid GPU OOM for large ID/OOD sets.
    Returns (det_score_cpu, ood_score_cpu), both on CPU float32.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Wd = W.to(device, non_blocking=True)
    bd = b.to(device, non_blocking=True) if b is not None else None
    mud = mu_g.to(device, non_blocking=True)

    N = feats.shape[0]
    det_out = torch.empty((N,), dtype=torch.float32, device="cpu")
    ood_out = torch.empty((N,), dtype=torch.float32, device="cpu")

    for i in range(0, N, batch_size):
        x = feats[i:i + batch_size].to(device, non_blocking=True)
        det, ood, _ = nci_scores_batch(
            x, Wd, bd, mud,
            alpha=alpha, p_norm=p_norm, use_bias=use_bias, eps=eps, abs_pscore=abs_pscore
        )
        det_out[i:i + det.shape[0]] = det.detach().cpu()
        ood_out[i:i + ood.shape[0]] = ood.detach().cpu()

    return det_out, ood_out

@torch.no_grad()
def extract_gaussian_feats_like_id(
    feat_model: torch.nn.Module,
    id_loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Builds Gaussian-noise validation features as in the paper:
    - for each ID batch x (already normalized by your test_tfm),
      replace it with x_gauss ~ N(0,1) per pixel in the same tensor space
    - pass through feat_model to get penultimate features
    Returns:
      feats_gauss: (N, D) float32 on CPU
    """
    feat_model.eval()
    feats = []

    for x, _ in id_loader:
        x = x.to(device, non_blocking=True)

        # Gaussian noise per pixel in the *same space the model sees*
        x_gauss = torch.randn_like(x)

        h = feat_model(x_gauss)          # (B, D)
        feats.append(h.detach().cpu())

    return torch.cat(feats, dim=0).float()




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



import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


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
    cov = cov + eps * torch.eye(D, device=feats.device, dtype=feats.dtype)
    precision = torch.linalg.inv(cov)
    return means, precision


@torch.no_grad()
def scores_mahalanobis_min_plus_align(
    feats: torch.Tensor,
    means: torch.Tensor,              # (C, D)
    precision: torch.Tensor,          # (D, D)
    W: torch.Tensor,                  # (C, D)
    use_centered_for_cos: bool = True,
    eps: float = 1e-6,
    tune_alpha: bool = True,
    delta: float = 0.25,
    alpha: float = 1.0,               # used only if tune_alpha=False
    return_alpha: bool = False,
) -> Tuple[np.ndarray, float] | np.ndarray:
    feats = feats.float()
    means = means.float()
    precision = precision.float()
    W = W.float()

    # pred class (logits)
    pred = torch.argmax(feats @ W.T, dim=1).long()  # (N,)

    # Mahalanobis min distance
    dists = []
    for c in range(means.shape[0]):
        zc = feats - means[c].unsqueeze(0)
        q = (zc @ precision) * zc
        d = q.sum(dim=1)
        dists.append(d)
    dists = torch.stack(dists, dim=1)               # (N, C)
    dmin = dists.min(dim=1).values                  # (N,)

    # alignment penalty a = 1 - cos(u, w_pred)
    z = feats - means[pred]
    u = z if use_centered_for_cos else feats

    u_n = F.normalize(u, p=2, dim=1, eps=eps)
    W_n = F.normalize(W, p=2, dim=1, eps=eps)
    w_pred = W_n[pred]
    cos = (u_n * w_pred).sum(dim=1).clamp(-1.0, 1.0)
    a = 1.0 - cos                                   # (N,)

    # tune alpha using your rule
    if tune_alpha:
        med_res = float(torch.median(dmin).item())  # base term median
        d_np = a.detach().cpu().numpy().astype(np.float64)
        med_d = float(np.median(d_np))
        alpha_used = 0.0 if med_d < 1e-12 else float(delta) * (med_res / med_d)
    else:
        alpha_used = float(alpha)

    s = dmin + alpha_used * a
    s_np = s.detach().cpu().numpy().astype(np.float64)

    if return_alpha:
        return s_np, alpha_used
    return s_np


def run_method_mahalanobis_plus_align(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    id_feats: torch.Tensor,
    ood_feats: torch.Tensor,
    num_classes: int,
    W: torch.Tensor,
    use_centered_for_cos: bool = True,
    eps_cov: float = 1e-5,
    eps_cos: float = 1e-6,
    tune_alpha: bool = True,
    delta: float = 0.25,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    means, precision = fit_shared_gaussian(
        train_feats, train_labels, num_classes=num_classes, eps=eps_cov
    )

    id_scores, alpha_used = scores_mahalanobis_min_plus_align(
        id_feats, means, precision, W,
        use_centered_for_cos=use_centered_for_cos, eps=eps_cos,
        tune_alpha=tune_alpha, delta=delta, alpha=alpha,
        return_alpha=True,
    )
    ood_scores = scores_mahalanobis_min_plus_align(
        ood_feats, means, precision, W,
        use_centered_for_cos=use_centered_for_cos, eps=eps_cos,
        tune_alpha=False, alpha=alpha_used, return_alpha=False,
    )
    print(f"mahalanobis_plus_align: alpha_used={alpha_used:.6f}")
    return id_scores, ood_scores


def neco_mahalanobis(train_feats, train_labels, id_feats, ood_feats, num_classes):
    method = "Mahalanobis"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- ensure numpy for sklearn part ---
    if torch.is_tensor(train_feats):
        train_feats_np = train_feats.detach().cpu().numpy()
    else:
        train_feats_np = np.asarray(train_feats)

    if torch.is_tensor(train_labels):
        train_labels_np = train_labels.detach().cpu().numpy()
    else:
        train_labels_np = np.asarray(train_labels)

    print("computing classwise mean feature...")
    train_means = []
    train_feat_centered = []

    for i in tqdm(range(num_classes)):
        fs = train_feats_np[train_labels_np == i]
        _m = fs.mean(axis=0)
        train_means.append(_m)
        train_feat_centered.append(fs - _m)

    train_feat_centered = np.concatenate(train_feat_centered, axis=0)
    print(f" len of train_feat_centered {train_feat_centered.shape[0]}")

    print("computing precision matrix...")
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(train_feat_centered.astype(np.float64))

    print("go to gpu...")
    mean = torch.from_numpy(np.array(train_means)).to(device=device, dtype=torch.float32)
    prec = torch.from_numpy(ec.precision_).to(device=device, dtype=torch.float32)

    # --- ensure torch for scoring part ---
    def to_torch(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.float32)
        return torch.from_numpy(np.asarray(x)).to(device=device, dtype=torch.float32)

    id_feats_t = to_torch(id_feats)
    ood_feats_t = to_torch(ood_feats)

    # score = - min_c ( (f-μ_c)^T Σ^{-1} (f-μ_c) )
    score_id = torch.stack([
        ((((f.unsqueeze(0) - mean) @ prec) * (f.unsqueeze(0) - mean)).sum(dim=-1)).min()
        for f in tqdm(id_feats_t)
    ]).detach().cpu().numpy()

    score_ood = torch.stack([
        ((((f.unsqueeze(0) - mean) @ prec) * (f.unsqueeze(0) - mean)).sum(dim=-1)).min()
        for f in tqdm(ood_feats_t)
    ]).detach().cpu().numpy()

    return score_id, score_ood

####################################################################################
# DECA: Dual-Error Component Analysis  (internal helper)
# deca.py
# Full DECA pipeline with two critical fixes:
#   (1) remove spectrum "double counting" (no w = S/sum(S) inside projections)
#   (2) normalize WPC/NPC per-dimension so curves represent rates, not raw totals



# ──────────────────────────────────────────────────────────────────────────────
# 1) FIT
# ──────────────────────────────────────────────────────────────────────────────

def fit_deca(
    train_feats: torch.Tensor,     # [N, D]
    *,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Decompose mean-centred training features via SVD (eq 2).

    Returns model dict:
        mean         [D]
        V            [D, r]   right singular vectors, descending sigma order
        S            [r]      singular values
        norm_q_sq    [r]      cumsum(sigma_i^2), i.e. norm_q^2 for each q
        sigma_cumsum [r]      cumsum(sigma_i)   for r* feature (eq 26)
        r            int
    """
    feats = train_feats.float()
    N, D  = feats.shape

    mu = feats.mean(dim=0)                                      # [D]
    X  = feats - mu                                             # [N, D]

    # economy SVD  (full_matrices=False → r = min(N,D))
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)         # S [r], Vh [r, D]
    V = Vh.T.contiguous()                                       # [D, r]
    r = int(S.shape[0])

    norm_q_sq    = torch.cumsum(S ** 2, dim=0)                  # [r]  norm_q^2
    sigma_cumsum = torch.cumsum(S,      dim=0)                  # [r]  for r* ratio

    print(
        f"[DECA fit] N={N}  D={D}  r={r}  "
        f"top-5 singular values: {S[:5].cpu().numpy().round(2)}"
    )

    return {
        "mean":         mu.contiguous(),
        "V":            V,
        "S":            S.contiguous(),
        "norm_q_sq":    norm_q_sq.contiguous(),
        "sigma_cumsum": sigma_cumsum.contiguous(),
        "r":            torch.tensor(r, dtype=torch.long),
        "eps":          torch.tensor(eps),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CURVE COMPUTATION  (internal, vectorised)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _compute_curves(
    feats:   torch.Tensor,               # [N, D]
    model:   Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:  # npc_curve [N, r],  wpc_curve [N, r]
    """
    Efficiently compute E_NPC(q) and E_WPC(q) for q = 1 … r.

    Uses the decomposition:

        E_NPC(q) = total - A(q)
        E_WPC(q) = A(q) - (2/norm_q) * B(q) + (1/norm_q^2) * C(q)

    where
        p_i       = x . v_i                    (raw projection)
        A(q)      = sum_{i=1}^q  p_i^2
        B(q)      = sum_{i=1}^q  sigma_i * p_i^2
        C(q)      = sum_{i=1}^q  sigma_i^2 * p_i^2
        norm_q^2  = sum_{i=1}^q  sigma_i^2      (precomputed, shape [r])
    """
    mu          = model["mean"]             # [D]
    V           = model["V"]                # [D, r]
    S           = model["S"]               # [r]
    norm_q_sq   = model["norm_q_sq"]        # [r]
    eps         = float(model["eps"].item())

    z = feats.float() - mu                  # [N, D]
    p = z @ V                               # [N, r]  raw projections

    p2   = p ** 2                           # [N, r]
    sp2  = S        * p2                    # [N, r]   sigma_i * p_i^2
    s2p2 = (S ** 2) * p2                    # [N, r]   sigma_i^2 * p_i^2

    total = p2.sum(dim=1, keepdim=True)     # [N, 1]   total energy

    A = torch.cumsum(p2,   dim=1)           # [N, r]
    B = torch.cumsum(sp2,  dim=1)           # [N, r]
    C = torch.cumsum(s2p2, dim=1)           # [N, r]

    # norm_q_sq[q-1] = sum_{j=1}^q sigma_j^2  (broadcast over N)
    nq2 = norm_q_sq.unsqueeze(0).clamp(min=eps)   # [1, r]

    npc_curve = total - A                                       # [N, r]
    wpc_curve = A - (2.0 / nq2.sqrt()) * B + C / nq2           # [N, r]   eq 12

    # clamp numerical negatives to zero
    npc_curve = npc_curve.clamp(min=0.0)
    wpc_curve = wpc_curve.clamp(min=0.0)

    return npc_curve, wpc_curve     # both [N, r]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  7-FEATURE EXTRACTION  (internal)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _extract_raw_features(
    feats: torch.Tensor,
    model: Dict[str, torch.Tensor],
) -> torch.Tensor:   # [N, 7]
    """
    Extract the 7 raw (un-normalised) features from the dual error curves.

    Feature index mapping (0-based):
        0 : Delta_E_init = |E_NPC(1) - E_WPC(1)|              (eq 25)
        1 : mu_NPC       = mean over q of E_NPC(q)             (eq 13)
        2 : mu_WPC       = mean over q of E_WPC(q)             (eq 14)
        3 : E_NPC(q*)    at balance point                      (eq 19)
        4 : E_WPC(q*)    at balance point                      (eq 20)
        5 : r*           = cumsum_sigma[q*] / cumsum_sigma[-1] (eq 26)
        6 : Delta_nabla  = |nabla E_NPC(q*) - nabla E_WPC(q*)|(eq 27)
    """
    npc, wpc = _compute_curves(feats, model)    # [N, r] each
    N, r = npc.shape

    sigma_cumsum = model["sigma_cumsum"]         # [r]

    # ── feature 0: initial gap (q = 1, index 0) ──────────────────────────────
    f0 = (npc[:, 0] - wpc[:, 0]).abs()          # [N]   eq 25
    # Note: E_WPC(1) = 0 theoretically (w_1*(1) = sigma_1/sigma_1 = 1),
    # so f0 = E_NPC(1) in the ideal case.

    # ── features 1, 2: global averages ───────────────────────────────────────
    f1 = npc.mean(dim=1)                         # [N]   eq 13  mu_NPC
    f2 = wpc.mean(dim=1)                         # [N]   eq 14  mu_WPC

    # ── balance point q* (eq 18) ─────────────────────────────────────────────
    gap    = (npc - wpc).abs()                   # [N, r]
    q_star = gap.argmin(dim=1).long()            # [N]   0-based index

    idx    = q_star.unsqueeze(1)                 # [N, 1]
    f3 = npc.gather(1, idx).squeeze(1)           # [N]   E_NPC(q*)  eq 19
    f4 = wpc.gather(1, idx).squeeze(1)           # [N]   E_WPC(q*)  eq 20

    # ── feature 5: r* = cumsum_sigma[q*] / total_sigma (eq 26) ──────────────
    total_sigma = sigma_cumsum[-1].clamp(min=float(model["eps"].item()))
    f5 = sigma_cumsum[q_star] / total_sigma      # [N]   in (0, 1]

    # ── feature 6: gradient gap at q* (eq 27-29) ─────────────────────────────
    # nabla E(q) = E(q+1) - E(q)
    # Use finite differences; clamp indices to valid range
    q_next = (q_star + 1).clamp(max=r - 1)
    q_prev = q_star                              # nabla at q* uses (q*+1) - q*

    npc_next = npc.gather(1, q_next.unsqueeze(1)).squeeze(1)
    wpc_next = wpc.gather(1, q_next.unsqueeze(1)).squeeze(1)
    npc_curr = npc.gather(1, q_prev.unsqueeze(1)).squeeze(1)
    wpc_curr = wpc.gather(1, q_prev.unsqueeze(1)).squeeze(1)

    grad_npc = npc_next - npc_curr               # nabla E_NPC(q*)  eq 28
    grad_wpc = wpc_next - wpc_curr               # nabla E_WPC(q*)  eq 29
    f6 = (grad_npc - grad_wpc).abs()             # [N]              eq 27

    return torch.stack([f0, f1, f2, f3, f4, f5, f6], dim=1)  # [N, 7]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CALIBRATE  (min/max from ID features for [0,1] normalisation)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def calibrate_deca(
    id_feats: torch.Tensor,
    model:    Dict[str, torch.Tensor],
    *,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-feature [min, max] on ID test features.
    Stored in model dict and used for [0,1] normalisation in score_deca.
    Must be called once before scoring.
    """
    F = _extract_raw_features(id_feats, model)   # [N, 7]
    model["F_min"] = F.min(dim=0).values          # [7]
    model["F_max"] = F.max(dim=0).values          # [7]
    model["norm_eps"] = torch.tensor(eps)

    np.set_printoptions(precision=4, suppress=True)
    print("[DECA calibrate] ID feature mins :", model["F_min"].cpu().numpy())
    print("[DECA calibrate] ID feature maxs :", model["F_max"].cpu().numpy())
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SCORE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_deca(
    feats: torch.Tensor,
    model: Dict[str, torch.Tensor],
) -> np.ndarray:
    """
    Compute DECA OOD scores.  Higher = more OOD.

    Implements eq 30:
        s(x) = -1/7 * sum_i [ normalised_feature_i ]

    Paper's classifier (eq 31): s(x) > tau => ID,  s(x) <= tau => OOD.
    We return  -s(x)  so that the conventional "higher = more OOD" holds.

    Requires calibrate_deca() to have been called first.
    """
    if "F_min" not in model:
        raise RuntimeError("Call calibrate_deca(id_feats, model) before scoring.")

    F     = _extract_raw_features(feats, model)                   # [N, 7]
    F_min = model["F_min"]                                        # [7]
    F_max = model["F_max"]                                        # [7]
    norm_eps = float(model["norm_eps"].item())

    # normalise each feature to [0, 1] using ID-calibrated min/max (eq 30)
    # values outside [min,max] are clipped to [0,1]
    scale  = (F_max - F_min).clamp(min=norm_eps)
    F_norm = ((F - F_min) / scale).clamp(0.0, 1.0)               # [N, 7]

    # eq 30: s(x) = -1/7 * [f0 + f1 + f2 + f3 + f4 + f5 + f6]
    # higher s(x) => more ID  →  negate for "higher = OOD" convention
    s = F_norm.sum(dim=1) / 7.0                                   # [N]
    return s.cpu().numpy().astype(np.float64)                     # return +s: higher = OOD


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CONVENIENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def run_deca(
    train_feats: torch.Tensor,
    id_feats:    torch.Tensor,
    ood_feats:   torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full DECA pipeline.

    Returns
    -------
    id_scores  [N_id]   lower  => more in-distribution
    ood_scores [N_ood]  higher => more out-of-distribution

    Usage with sklearn:
        from sklearn.metrics import roc_auc_score
        y_true  = [0]*len(id_scores) + [1]*len(ood_scores)
        y_score = np.concatenate([id_scores, ood_scores])
        auroc   = roc_auc_score(y_true, y_score)
    """
    model = fit_deca(train_feats)
    calibrate_deca(id_feats, model)

    id_scores  = score_deca(id_feats,  model)
    ood_scores = score_deca(ood_feats, model)

    print(
        f"[DECA score] ID  : mean={id_scores.mean():.6f}  std={id_scores.std():.6f}\n"
        f"             OOD : mean={ood_scores.mean():.6f}  std={ood_scores.std():.6f}"
    )
    return id_scores, ood_scores



