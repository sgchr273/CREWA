#!/usr/bin/env python3
"""
methods.py — OOD scoring methods.

Public API (everything called from evaluate.py)
------------------------------------------------
CREWA
    fit_affine_pca_subspaces       fit pooled within-class PCA basis
    score_subspaces_resid_plus_align_simple   score one split
    run_crewa                      end-to-end pipeline

Mahalanobis
    neco_mahalanobis

Energy / MSP
    scores_energy_from_logits
    scores_msp_from_logits

Logit-gate
    fit_global_pca_basis
    scores_logit_gate

KPCA-RFF
    fit_kpca_rff
    scores_kpca_rff

GradSubspace
    fit_id_feature_subspace
    scores_gradsubspace_pseudo_resid

NECO
    run_method_neco

ViM
    run_method_vim

NCI
    compute_mu_global
    nci_scores_batched

DECA
    run_deca
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

NormOrd = Union[int, float, str]


# ══════════════════════════════════════════════════════════════════════════════
# CREWA  —  Complement-Residual + Alignment
# ══════════════════════════════════════════════════════════════════════════════

def fit_affine_pca_subspaces(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    energy_keep: float = 0.95,
    k_max: int = 150,   # kept for API compatibility, not used
    eps: float = 1e-6,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Pooled within-class PCA (tied covariance).

    Computes class means, pools within-class residuals, runs economy SVD,
    then selects the *complement* (low-variance) subspace using stable-rank
    heuristic rather than energy threshold.

    Returns a dict keyed by 0 with entries:
        mu_c      [C, D]  class means
        V_keep    [D, k]  top-k principal directions
        V_perp    [D, p]  discarded (complement) directions
        lam_perp  [p]     eigenvalues of discarded directions
        counts    [C]     per-class sample counts
    """
    feats  = feats.float()
    labels = labels.long()

    N, D = feats.shape
    if N < 5:
        raise RuntimeError(f"Too few samples ({N}).")
    if labels.numel() != N:
        raise RuntimeError(f"labels/feats length mismatch: {labels.numel()} vs {N}")

    # class means
    mu_c   = feats.new_zeros((num_classes, D))
    counts = torch.zeros(num_classes, device=feats.device, dtype=torch.long)
    for c in range(num_classes):
        mask = labels == c
        nc   = int(mask.sum().item())
        if nc == 0:
            raise RuntimeError(f"Class {c} has zero training samples.")
        mu_c[c]   = feats[mask].mean(dim=0)
        counts[c] = nc

    # pooled within-class SVD
    X = feats - mu_c[labels]                        # [N, D]
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V   = Vh.t()                                    # [D, r]
    r   = V.shape[1]
    lam = (S ** 2) / max(N - 1, 1)

    # stable-rank based split: keep bottom (r - k_stable) complement directions
    stable_r = float((S ** 2).sum() ** 2 / ((S ** 4).sum() + eps))
    k        = max(1, min(int(0.3 * stable_r), r - 64))

    print(
        f"[CREWA fit] r={r} | stable_rank={stable_r:.1f} | "
        f"k_keep={k} | perp_dim={r - k}"
    )

    V_keep   = V[:, :k].contiguous()
    V_perp   = V[:, k:].contiguous()
    lam_perp = lam[k:].contiguous()

    if V_perp.numel() == 0:             # safety: always keep ≥1 perp direction
        V_perp   = V[:, -1:].contiguous()
        lam_perp = lam[-1:].contiguous()

    return {
        0: {
            "mu_c":     mu_c.contiguous(),
            "V_keep":   V_keep,
            "V_perp":   V_perp,
            "lam_perp": lam_perp,
            "counts":   counts.contiguous(),
        }
    }


@torch.no_grad()
def score_subspaces_resid_plus_align_simple(
    feats: torch.Tensor,
    logits: torch.Tensor,
    W: torch.Tensor,                        # [C, D]
    subspaces: Dict[int, Dict[str, torch.Tensor]],
    *,
    beta: float = 1.0,
    use_centered_for_cos: bool = False,
    eps: float = 1e-6,
    tune_beta: bool = False,
    gamma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OOD score = residual energy in complement space + beta * alignment penalty.

    When tune_beta=True, beta is calibrated on the *current* batch (intended
    to be ID-only):
        beta = gamma * median(s_res) / median(a)

    Returns
    -------
    s_total : [N]  combined OOD score  (higher => more OOD)
    a       : [N]  alignment penalty   (returned for diagnostics)
    """
    feats  = feats.float()
    logits = logits.float()
    W      = W.float()

    m    = subspaces[0]
    mu_c = m["mu_c"]    # [C, D]
    Vp   = m["V_perp"]  # [D, p]

    # shape checks
    if feats.shape[1] != mu_c.shape[1]:
        raise RuntimeError(f"D mismatch: feats {feats.shape[1]} vs mu_c {mu_c.shape[1]}")
    if logits.shape[1] != mu_c.shape[0]:
        raise RuntimeError(f"C mismatch: logits {logits.shape[1]} vs mu_c {mu_c.shape[0]}")
    if W.shape != (mu_c.shape[0], mu_c.shape[1]):
        raise RuntimeError(
            f"W must be [C={mu_c.shape[0]}, D={mu_c.shape[1]}], got {tuple(W.shape)}"
        )

    # predicted class via classifier weights
    pred  = torch.argmax(feats @ W.t(), dim=1).long()   # [N]
    z     = feats - mu_c[pred]                          # [N, D]

    # residual energy in complement subspace
    coeff = z @ Vp                                      # [N, p]
    s_res = (coeff ** 2).sum(dim=1)                     # [N]

    # alignment penalty  a = 1 − cos(u, w_pred)
    u   = z if use_centered_for_cos else feats
    u_n = F.normalize(u, p=2, dim=1, eps=eps)
    W_n = F.normalize(W, p=2, dim=1, eps=eps)
    cos = (u_n * W_n[pred]).sum(dim=1).clamp(-1.0, 1.0)
    a   = 1.0 - cos                                     # [N]

    # optional ID-only beta calibration
    if tune_beta:
        s_np    = s_res.detach().cpu().numpy().astype(np.float64)
        a_np    = a.detach().cpu().numpy().astype(np.float64)
        med_res = float(np.median(s_np))
        med_a   = float(np.median(a_np))
        beta    = 0.0 if med_a < 1e-12 else float(gamma) * (med_res / med_a)

    s_total = (s_res + float(beta) * a).detach().cpu().numpy().astype(np.float64)
    a_out   = a.detach().cpu().numpy().astype(np.float64)
    return s_total, a_out


def run_crewa(
    train_feats:  torch.Tensor,
    train_labels: torch.Tensor,
    id_feats:     torch.Tensor,
    id_logits:    torch.Tensor,
    ood_feats:    torch.Tensor,
    ood_logits:   torch.Tensor,
    num_classes:  int,
    energy_keep:  float,
    k_max:        int,
    *,
    W:                    torch.Tensor,
    beta:                 float = 1.0,
    use_centered_for_cos: bool  = False,
    tune_beta:            bool  = False,
    gamma:                float = 1.0,
    eps:                  float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    End-to-end CREWA pipeline.

    1. Fit pooled within-class PCA on training features.
    2. Calibrate beta on ID features only (if tune_beta=True).
    3. Score ID and OOD features with the fixed beta.

    Returns (id_scores, ood_scores), higher => more OOD.
    """
    subspaces = fit_affine_pca_subspaces(
        feats=train_feats, labels=train_labels, num_classes=num_classes,
        energy_keep=energy_keep, k_max=k_max, eps=eps,
    )

    beta_used = float(beta)
    if tune_beta:
        s_res_id, a_id = score_subspaces_resid_plus_align_simple(
            id_feats, id_logits, W, subspaces,
            beta=0.0, use_centered_for_cos=use_centered_for_cos,
            eps=eps, tune_beta=False,
        )
        med_res   = float(np.median(s_res_id))
        med_a     = float(np.median(a_id))
        beta_used = 0.0 if med_a < 1e-12 else float(gamma) * (med_res / med_a)

    print(f"[CREWA] beta_used={beta_used:.6g}  (tune_beta={tune_beta}, gamma={gamma})")

    kw = dict(
        W=W, subspaces=subspaces,
        beta=beta_used, use_centered_for_cos=use_centered_for_cos,
        eps=eps, tune_beta=False,
    )
    s_id,  _ = score_subspaces_resid_plus_align_simple(id_feats,  id_logits,  **kw)
    s_ood, _ = score_subspaces_resid_plus_align_simple(ood_feats, ood_logits, **kw)
    return s_id, s_ood


# ══════════════════════════════════════════════════════════════════════════════
# Mahalanobis  (sklearn EmpiricalCovariance variant)
# ══════════════════════════════════════════════════════════════════════════════

def neco_mahalanobis(
    train_feats:  torch.Tensor,
    train_labels: torch.Tensor,
    id_feats:     torch.Tensor,
    ood_feats:    torch.Tensor,
    num_classes:  int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Class-conditional Mahalanobis distance with a shared precision matrix
    estimated via sklearn's EmpiricalCovariance.

    Returns (id_scores, ood_scores); lower score => more ID-like.
    Use ensure_ood_higher() in evaluate.py before reporting.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    tf_np = _np(train_feats)
    tl_np = _np(train_labels)

    print("[Mahalanobis] computing class-wise means …")
    means, centered = [], []
    for c in tqdm(range(num_classes)):
        fs  = tf_np[tl_np == c]
        mu  = fs.mean(axis=0)
        means.append(mu)
        centered.append(fs - mu)

    print("[Mahalanobis] fitting precision matrix …")
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.concatenate(centered, axis=0).astype(np.float64))

    mean = torch.from_numpy(np.stack(means)).to(device, dtype=torch.float32)  # [C, D]
    prec = torch.from_numpy(ec.precision_).to(device, dtype=torch.float32)   # [D, D]

    def _score(feats_t: torch.Tensor) -> np.ndarray:
        f = feats_t.to(device, dtype=torch.float32)
        return torch.stack([
            (((f[i] - mean) @ prec * (f[i] - mean)).sum(dim=-1)).min()
            for i in tqdm(range(f.shape[0]))
        ]).detach().cpu().numpy()

    return _score(id_feats), _score(ood_feats)


# ══════════════════════════════════════════════════════════════════════════════
# Energy / MSP  (logits-based baselines)
# ══════════════════════════════════════════════════════════════════════════════

def scores_energy_from_logits(logits: torch.Tensor, T: float = 1.0) -> np.ndarray:
    """Free energy score: −T · log Σ_c exp(f_c / T). Higher => more OOD."""
    e = -T * torch.logsumexp(logits.float() / T, dim=1)
    return e.detach().cpu().numpy().astype(np.float64)


def scores_msp_from_logits(logits: torch.Tensor) -> np.ndarray:
    """−max softmax probability. Higher => more OOD."""
    msp = F.softmax(logits.float(), dim=1).max(dim=1).values
    return (-msp).detach().cpu().numpy().astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Logit-gate
# ══════════════════════════════════════════════════════════════════════════════

def fit_global_pca_basis(
    feats: torch.Tensor,
    energy_keep: float = 0.95,
    k_max: int = 64,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a global PCA basis retaining `energy_keep` fraction of variance.
    Returns (feat_mean [D], Vk [D, k]).
    """
    X    = feats.float()
    mean = X.mean(dim=0)
    Z    = X - mean

    _, S, Vh = torch.linalg.svd(Z, full_matrices=False)
    V   = Vh.t()
    lam = (S ** 2) / max(Z.shape[0] - 1, 1)

    energy = torch.cumsum(lam, dim=0) / (lam.sum() + eps)
    k      = int(torch.searchsorted(energy, torch.tensor(energy_keep)).item()) + 1
    k      = min(k, k_max, V.shape[1])

    return mean.contiguous(), V[:, :k].contiguous()


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
    """
    Logit-gate score combining free energy with a PCA reconstruction penalty.
    Higher => more OOD.
    """
    X   = feats.float()
    Xc  = torch.clamp(X, max=threshold)
    lse = torch.logsumexp(Xc @ W.t() + b, dim=1)

    Z        = X - feat_mean
    proj     = (Z @ Vk) @ Vk.t()
    rec_norm = torch.linalg.norm(Z - proj, dim=1)
    denom    = torch.linalg.norm(Xc, dim=1).clamp_min(eps)
    r        = rec_norm / denom

    return (-(lse * (1.0 - r))).detach().cpu().numpy().astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# KPCA-RFF  (Random Fourier Feature approximation of kernel PCA)
# ══════════════════════════════════════════════════════════════════════════════

def _l2_normalize_np(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)).astype(np.float32)


def _rff_map_np(x: np.ndarray, W: np.ndarray, u: np.ndarray) -> np.ndarray:
    return (np.sqrt(2.0 / W.shape[0]) * np.cos(x @ W.T + u)).astype(np.float32)


def fit_kpca_rff(
    train_feats: torch.Tensor,
    gamma: float,
    M: int,
    exp_var_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Fit KPCA via Random Fourier Features.
    Returns (mu [M], Uq [M, q], q).
    """
    X   = _l2_normalize_np(train_feats.cpu().numpy().astype(np.float32))
    D   = X.shape[1]
    rng = np.random.RandomState(seed)
    Wr  = (np.sqrt(2.0 * gamma) * rng.normal(size=(M, D))).astype(np.float32)
    u   = (2.0 * np.pi * rng.rand(M)).astype(np.float32)

    Xm = _rff_map_np(X, Wr, u)
    mu = Xm.mean(axis=0)
    Xm = Xm - mu

    U, s, _ = np.linalg.svd(Xm.T @ Xm, full_matrices=False)
    denom   = float(s.sum()) or 1.0
    q, cum  = 1, 0.0
    for i, sv in enumerate(s):
        cum += float(sv) / denom
        if cum >= exp_var_ratio:
            q = i + 1
            break

    return mu.astype(np.float32), U[:, :q].astype(np.float32), q


def scores_kpca_rff(
    query_feats: torch.Tensor,
    mu: np.ndarray,
    Uq: np.ndarray,
    gamma: float,
    M: int,
    seed: int,
) -> np.ndarray:
    """RFF reconstruction error in the KPCA subspace. Higher => more OOD."""
    Xq  = _l2_normalize_np(query_feats.cpu().numpy().astype(np.float32))
    D   = Xq.shape[1]
    rng = np.random.RandomState(seed)
    Wr  = (np.sqrt(2.0 * gamma) * rng.normal(size=(M, D))).astype(np.float32)
    u   = (2.0 * np.pi * rng.rand(M)).astype(np.float32)

    Xm   = _rff_map_np(Xq, Wr, u) - mu
    proj = (Xm @ Uq) @ Uq.T
    return np.linalg.norm(Xm - proj, axis=1).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# GradSubspace  (uncertainty × residual norm to ID feature subspace)
# ══════════════════════════════════════════════════════════════════════════════

def fit_id_feature_subspace(
    train_feats:   torch.Tensor,
    n_batch:       int,
    exp_var_ratio: float,
    center:        bool,
    seed:          int,
    eps:           float = 1e-8,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """
    Build an ID principal subspace S from a random mini-batch.
    Returns (S [D, k], mu [1, D] or None, k).
    """
    X   = train_feats.float()
    N   = X.shape[0]
    idx = np.random.RandomState(seed).choice(N, size=min(n_batch, N), replace=False)
    B   = X[idx]

    mu = None
    if center:
        mu = B.mean(dim=0, keepdim=True)
        B  = B - mu

    _, Sv, Vh = torch.linalg.svd(B, full_matrices=False)
    V   = Vh.t()
    lam = Sv ** 2
    cum = torch.cumsum(lam, dim=0) / (lam.sum() + eps)
    k   = int(torch.searchsorted(cum, torch.tensor(exp_var_ratio, device=cum.device)).item()) + 1
    k   = max(1, min(k, V.shape[1]))

    return V[:, :k].contiguous(), mu, k


@torch.no_grad()
def scores_gradsubspace_pseudo_resid(
    feats:       torch.Tensor,
    W:           torch.Tensor,
    b:           torch.Tensor,
    S:           torch.Tensor,
    mu:          Optional[torch.Tensor],
    num_classes: int,
    eps:         float = 1e-8,
) -> np.ndarray:
    """
    score(x) = (1 − max_softmax) × ‖x_perp‖. Higher => more OOD.
    """
    X      = feats.float()
    logits = X @ W.t() + b
    a      = (1.0 - F.softmax(logits, dim=1).max(dim=1).values).clamp_min(eps)

    Xr     = X - mu if mu is not None else X
    f_proj = (Xr @ S) @ S.t()
    r      = torch.linalg.norm(Xr - f_proj, dim=1).clamp_min(eps)

    return (a * r).detach().cpu().numpy().astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# NECO  (Neural Collapse Energy ratio)
# ══════════════════════════════════════════════════════════════════════════════

def run_method_neco(
    train_feats: torch.Tensor,
    id_feats:    torch.Tensor,
    ood_feats:   torch.Tensor,
    W:           torch.Tensor,
    b:           torch.Tensor,
    arch:        str,
    neco_dim:    int,
    eps:         float = 1e-12,
    use_scaler   = None,    # None = auto-detect from arch; True/False = override
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ratio of energy in top-k PCA directions to total feature energy.

    For ResNet, applies StandardScaler before PCA.
    For ViT/Swin, multiplies score by max logit (original NECO paper behaviour).

    Returns (id_scores, ood_scores); higher => more ID-like (flip in evaluate.py).
    """
    is_resnet  = "resnet" in arch.lower()
    use_scaler = is_resnet if use_scaler is None else bool(use_scaler)

    def _np(t): return t.detach().float().cpu().numpy()
    Xtr, Xid, Xod = _np(train_feats), _np(id_feats), _np(ood_feats)

    if use_scaler:
        ss  = StandardScaler()
        Xtr = ss.fit_transform(Xtr)
        Xid = ss.transform(Xid)
        Xod = ss.transform(Xod)

    D = Xtr.shape[1]
    k = int(neco_dim)
    if not (1 <= k <= D):
        raise ValueError(f"neco_dim must be in [1, {D}], got {k}")

    pca = PCA(n_components=D).fit(Xtr)
    Zid = pca.transform(Xid)[:, :k]
    Zod = pca.transform(Xod)[:, :k]

    s_id  = np.linalg.norm(Zid, axis=1) / (np.linalg.norm(Xid, axis=1) + eps)
    s_ood = np.linalg.norm(Zod, axis=1) / (np.linalg.norm(Xod, axis=1) + eps)

    if not is_resnet:
        with torch.no_grad():
            Wf, bf = W.detach().float(), b.detach().float()
            ml_id  = (id_feats.float()  @ Wf.t() + bf).max(dim=1).values.cpu().numpy()
            ml_ood = (ood_feats.float() @ Wf.t() + bf).max(dim=1).values.cpu().numpy()
        s_id  *= ml_id
        s_ood *= ml_ood

    return s_id, s_ood


# ══════════════════════════════════════════════════════════════════════════════
# ViM  (Virtual-logit Matching)
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_Wb(
    W: torch.Tensor, b: torch.Tensor, feat_dim: int, num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (W [C, D], b [C]) regardless of transposition."""
    W = W.detach().float()
    b = b.detach().float().view(-1)
    if W.shape == (num_classes, feat_dim):
        return W, b
    if W.shape == (feat_dim, num_classes):
        return W.t().contiguous(), b
    raise RuntimeError(
        f"W shape {tuple(W.shape)} incompatible with "
        f"feat_dim={feat_dim}, num_classes={num_classes}."
    )


def _vim_fit(
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
    """Fit ViM: bias offset o, principal subspace U, and scaling alpha."""
    feats = train_feats.detach().float().cpu()
    N, feat_dim = feats.shape
    W_cls, b_cls = _ensure_Wb(W, b, feat_dim, num_classes)

    if D is None or int(D) <= 0:
        D = min(num_classes, feat_dim - 1) if feat_dim >= num_classes \
            else max(1, int(round(2.0 * feat_dim / 3.0)))
    D = int(D)
    if not (1 <= D < feat_dim):
        raise RuntimeError(f"D={D} out of valid range [1, feat_dim={feat_dim}).")

    K   = min(int(fit_max), N)
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed))[:K]
    X   = feats[idx]

    o   = -torch.linalg.pinv(W_cls) @ b_cls           # [feat_dim]
    Z   = (X - o).to(device)
    cov = (Z.t() @ Z) / max(K - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)
    U   = evecs[:, torch.argsort(evals, descending=True)][:, :D].contiguous()

    proj   = Z @ U
    resid  = torch.sqrt((Z * Z).sum(1) - (proj * proj).sum(1) + eps)
    logits = X.to(device) @ W_cls.t().to(device) + b_cls.to(device)
    alpha  = (logits.max(dim=1).values.sum() / (resid.sum() + eps)).detach()

    return {
        "o":     o.detach().cpu(),
        "U":     U.detach().cpu(),
        "alpha": alpha.detach().cpu(),
    }


@torch.no_grad()
def _vim_scores(
    feats: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    state: Dict[str, torch.Tensor],
    num_classes: int,
    eps: float = 1e-6,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """s(x) = alpha * ‖x_perp‖ − logsumexp(logits). Higher => more OOD."""
    f        = feats.detach().float().to(device)
    W_cls, b_cls = _ensure_Wb(W, b, int(f.shape[1]), num_classes)
    o      = state["o"].to(device).view(1, -1)
    U      = state["U"].to(device)
    alpha  = float(state["alpha"].item())

    Z      = f - o
    proj   = Z @ U
    resid  = torch.sqrt((Z * Z).sum(1) - (proj * proj).sum(1) + eps)
    energy = torch.logsumexp(f @ W_cls.t().to(device) + b_cls.to(device), dim=1)
    return (alpha * resid - energy).detach().cpu().numpy().astype(np.float64)


def run_method_vim(
    train_feats:  torch.Tensor,
    id_feats:     torch.Tensor,
    ood_feats:    torch.Tensor,
    W:            torch.Tensor,
    b:            torch.Tensor,
    num_classes:  int,
    vim_dim:      int = 0,
    vim_fit_max:  int = 200_000,
    seed:         int = 0,
    eps:          float = 1e-6,
    fit_device:   torch.device = torch.device("cpu"),
    score_device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """End-to-end ViM. Returns (id_scores, ood_scores), higher => more OOD."""
    state = _vim_fit(
        train_feats, W, b, num_classes,
        D=vim_dim, fit_max=vim_fit_max, seed=seed, eps=eps, device=fit_device,
    )
    kw = dict(W=W, b=b, state=state, num_classes=num_classes, eps=eps, device=score_device)
    return _vim_scores(id_feats, **kw), _vim_scores(ood_feats, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# NCI  (Neural Collapse for OOD detection)
# ══════════════════════════════════════════════════════════════════════════════

def compute_mu_global(train_feats: torch.Tensor) -> torch.Tensor:
    """Global mean feature vector. Returns [D]."""
    return train_feats.float().mean(dim=0)


def _vec_norm(x: torch.Tensor, ord: NormOrd, dim: int, eps: float) -> torch.Tensor:
    n = torch.linalg.vector_norm(
        x, ord=float("inf") if ord == "inf" else float(ord), dim=dim
    )
    return n.clamp_min(eps)


@torch.no_grad()
def _nci_batch(
    feats: torch.Tensor,
    W:     torch.Tensor,
    b:     Optional[torch.Tensor],
    mu_g:  torch.Tensor,
    *,
    alpha:      float   = 0.0,
    p_norm:     NormOrd = 1,
    use_bias:   bool    = True,
    eps:        float   = 1e-12,
    abs_pscore: bool    = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    pScore   = ((h − μ_G) · w_c) / ‖h − μ_G‖₂
    detScore = pScore + alpha * ‖h‖_p
    ood_score = −detScore  (higher => more OOD)
    """
    feats = feats.float()
    W     = W.float()
    mu_g  = mu_g.float()
    if b is not None:
        b = b.float()

    logits = feats @ W.t() + b if (use_bias and b is not None) else feats @ W.t()
    pred   = logits.argmax(dim=1)
    z      = feats - mu_g
    z_norm = _vec_norm(z, 2, dim=1, eps=eps)
    pscore = (z * W[pred]).sum(dim=1) / z_norm
    if abs_pscore:
        pscore = pscore.abs()

    det = pscore + float(alpha) * _vec_norm(feats, p_norm, dim=1, eps=eps)
    return det, -det


@torch.no_grad()
def nci_scores_batched(
    feats: torch.Tensor,
    W:     torch.Tensor,
    b:     Optional[torch.Tensor],
    mu_g:  torch.Tensor,
    *,
    alpha:      float              = 0.0,
    p_norm:     NormOrd            = 1,
    use_bias:   bool               = True,
    eps:        float              = 1e-12,
    abs_pscore: bool               = False,
    batch_size: int                = 65536,
    device:     Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked NCI scoring (avoids GPU OOM on large feature sets).
    Returns (det_score_cpu [N], ood_score_cpu [N]).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Wd  = W.to(device)
    bd  = b.to(device) if b is not None else None
    mud = mu_g.to(device)

    N   = feats.shape[0]
    det_out = torch.empty(N, dtype=torch.float32)
    ood_out = torch.empty(N, dtype=torch.float32)

    for i in range(0, N, batch_size):
        x = feats[i:i + batch_size].to(device)
        d, o = _nci_batch(x, Wd, bd, mud,
                          alpha=alpha, p_norm=p_norm, use_bias=use_bias,
                          eps=eps, abs_pscore=abs_pscore)
        det_out[i:i + d.shape[0]] = d.detach().cpu()
        ood_out[i:i + o.shape[0]] = o.detach().cpu()

    return det_out, ood_out


# ══════════════════════════════════════════════════════════════════════════════
# DECA  (Dual-Error Curve Analysis)
# ══════════════════════════════════════════════════════════════════════════════

def _fit_deca(
    train_feats: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Mean-centre features, run economy SVD, precompute cumulative statistics.

    Returns model dict: mean, V [D,r], S [r], norm_q_sq [r],
                        sigma_cumsum [r], r, eps.
    """
    feats = train_feats.float()
    N, D  = feats.shape
    mu    = feats.mean(dim=0)
    X     = feats - mu

    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh.t().contiguous()
    r = int(S.shape[0])

    print(f"[DECA fit] N={N} D={D} r={r}  top-5 σ: {S[:5].cpu().numpy().round(2)}")

    return {
        "mean":         mu.contiguous(),
        "V":            V,
        "S":            S.contiguous(),
        "norm_q_sq":    torch.cumsum(S ** 2, dim=0).contiguous(),
        "sigma_cumsum": torch.cumsum(S,      dim=0).contiguous(),
        "r":            torch.tensor(r, dtype=torch.long),
        "eps":          torch.tensor(eps),
    }


@torch.no_grad()
def _deca_curves(
    feats: torch.Tensor,
    model: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorised E_NPC(q) and E_WPC(q) for q = 1 … r. Returns ([N,r], [N,r])."""
    mu  = model["mean"]
    V   = model["V"]
    S   = model["S"]
    eps = float(model["eps"].item())
    nq2 = model["norm_q_sq"].unsqueeze(0).clamp_min(eps)

    z   = feats.float() - mu
    p   = z @ V                              # [N, r]
    p2  = p ** 2
    A   = torch.cumsum(p2,       dim=1)
    B   = torch.cumsum(S * p2,   dim=1)
    C   = torch.cumsum(S**2 * p2, dim=1)
    tot = p2.sum(dim=1, keepdim=True)

    npc = (tot - A).clamp_min(0.0)
    wpc = (A - 2.0 / nq2.sqrt() * B + C / nq2).clamp_min(0.0)
    return npc, wpc


@torch.no_grad()
def _deca_features(
    feats: torch.Tensor,
    model: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Extract 7 DECA features per sample. Returns [N, 7]."""
    npc, wpc = _deca_curves(feats, model)
    N, r     = npc.shape
    sig_cum  = model["sigma_cumsum"]
    tot_sig  = sig_cum[-1].clamp_min(float(model["eps"].item()))

    f0 = (npc[:, 0] - wpc[:, 0]).abs()
    f1 = npc.mean(dim=1)
    f2 = wpc.mean(dim=1)

    q_star = (npc - wpc).abs().argmin(dim=1)
    idx    = q_star.unsqueeze(1)
    f3     = npc.gather(1, idx).squeeze(1)
    f4     = wpc.gather(1, idx).squeeze(1)
    f5     = sig_cum[q_star] / tot_sig

    q_next = (q_star + 1).clamp(max=r - 1).unsqueeze(1)
    f6     = ((npc.gather(1, q_next) - npc.gather(1, idx)) -
              (wpc.gather(1, q_next) - wpc.gather(1, idx))).abs().squeeze(1)

    return torch.stack([f0, f1, f2, f3, f4, f5, f6], dim=1)


@torch.no_grad()
def _calibrate_deca(
    id_feats: torch.Tensor,
    model: Dict[str, torch.Tensor],
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """Store per-feature [min, max] from ID features for [0,1] normalisation."""
    F = _deca_features(id_feats, model)
    model["F_min"]    = F.min(dim=0).values
    model["F_max"]    = F.max(dim=0).values
    model["norm_eps"] = torch.tensor(eps)
    return model


@torch.no_grad()
def _score_deca(
    feats: torch.Tensor,
    model: Dict[str, torch.Tensor],
) -> np.ndarray:
    """Mean of normalised 7-feature vector. Higher => more OOD."""
    if "F_min" not in model:
        raise RuntimeError("Call _calibrate_deca(id_feats, model) before scoring.")
    F      = _deca_features(feats, model)
    scale  = (model["F_max"] - model["F_min"]).clamp_min(float(model["norm_eps"].item()))
    F_norm = ((F - model["F_min"]) / scale).clamp(0.0, 1.0)
    return (F_norm.sum(dim=1) / 7.0).cpu().numpy().astype(np.float64)


def run_deca(
    train_feats: torch.Tensor,
    id_feats:    torch.Tensor,
    ood_feats:   torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full DECA pipeline. Returns (id_scores, ood_scores), higher => more OOD."""
    model = _fit_deca(train_feats)
    _calibrate_deca(id_feats, model)
    id_scores  = _score_deca(id_feats,  model)
    ood_scores = _score_deca(ood_feats, model)
    print(
        f"[DECA] ID  mean={id_scores.mean():.6f}  std={id_scores.std():.6f}\n"
        f"       OOD mean={ood_scores.mean():.6f}  std={ood_scores.std():.6f}"
    )
    return id_scores, ood_scores
