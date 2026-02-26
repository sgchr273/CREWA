from typing import Tuple, List, Dict
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance
from numpy.linalg import norm, pinv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA

from models import DEVICE, forward_adapt

# ---------- Metrics & utils ----------
RECALL = 0.95

def logsumexp(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(arr, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(arr - m), axis=axis, keepdims=True))).squeeze(axis)

class Metrics:
    @staticmethod
    def auc(id_scores, ood_scores):
        from sklearn.metrics import roc_auc_score
        y = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
        s = np.concatenate([id_scores, ood_scores])
        return float(roc_auc_score(y, s))

    @staticmethod
    def fpr_recall(id_scores, ood_scores, target_recall=RECALL):
        from sklearn.metrics import roc_curve
        y = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
        s = np.concatenate([id_scores, ood_scores])
        fpr, tpr, _ = roc_curve(y, s)
        mask = tpr >= target_recall
        idx = np.where(mask)[0][0] if np.any(mask) else np.argmax(tpr)
        return float(fpr[idx]), float(tpr[idx])

# ---------- Collectors ----------
@torch.no_grad()
def collect_logits_softmax(model, loader, arch):
    all_logits, all_softmax = [], []
    for x, *_ in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits, *_ = forward_adapt(model, x, arch)
        sm = F.softmax(logits, dim=-1)
        all_logits.append(logits.float().cpu())
        all_softmax.append(sm.float().cpu())
    logits_np = torch.cat(all_logits, dim=0).numpy()
    softmax_np = torch.cat(all_softmax, dim=0).numpy()
    return logits_np, softmax_np

@torch.no_grad()
def collect_penultimate_and_labels(model, loader, arch):
    feats, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits, logits_aux, penult, _ = forward_adapt(model, x, arch)
        feats.append(penult.detach().cpu())
        labels.append(y)
    feats = torch.cat(feats, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return feats, labels

# ---------- Baselines (MSP, MaxLogit, Energy) ----------

def method_msp(id_sm, ood_sm, name):
    score_id = id_sm.max(axis=-1)
    score_ood = ood_sm.max(axis=-1)
    auc = Metrics.auc(-score_id, -score_ood)
    fpr, _ = Metrics.fpr_recall(-score_id, -score_ood)
    print(f"MSP: {name} auroc {auc:.2%}, fpr {fpr:.2%}")

def method_maxlogit(id_logits, ood_logits, name):
    score_id = id_logits.max(axis=-1)
    score_ood = ood_logits.max(axis=-1)
    auc = Metrics.auc(-score_id, -score_ood)
    fpr, _ = Metrics.fpr_recall(-score_id, -score_ood)
    print(f"MaxLogit: {name} auroc {auc:.2%}, fpr {fpr:.2%}")

def method_energy(id_logits, ood_logits, name):
    score_id = logsumexp(id_logits, axis=-1)
    score_ood = logsumexp(ood_logits, axis=-1)
    auc = Metrics.auc(-score_id, -score_ood)
    fpr, _ = Metrics.fpr_recall(-score_id, -score_ood)
    print(f"Energy: {name} auroc {auc:.2%}, fpr {fpr:.2%}")

# ---------- Mahalanobis ----------

def method_mahalanobis(feature_id_train, train_labels, feature_id_val, feature_ood, num_classes, name):
    print('[Mahalanobis] fitting class means/precision...')
    train_means, centered = [], []
    for c in range(num_classes):
        fs = feature_id_train[train_labels == c]
        m = fs.mean(axis=0)
        train_means.append(m)
        centered.append(fs - m)
    centered = np.concatenate(centered, axis=0)
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(centered.astype(np.float64))
    mean = torch.from_numpy(np.stack(train_means)).to(DEVICE).float()
    prec = torch.from_numpy(ec.precision_).to(DEVICE).float()

    def min_maha_sq(X):
        X = torch.from_numpy(X).to(DEVICE).float()
        XC = X[:, None, :] - mean[None, :, :]
        t = torch.matmul(XC, prec)
        d2 = (t * XC).sum(dim=-1)
        return d2.min(dim=1).values.detach().cpu().numpy()

    score_id = min_maha_sq(feature_id_val)
    score_ood = min_maha_sq(feature_ood)
    auc = Metrics.auc(score_id, score_ood)
    fpr, _ = Metrics.fpr_recall(score_id, score_ood)
    print(f"Mahalanobis: {name} auroc {auc:.2%}, fpr {fpr:.2%}")

# ---------- GradNorm ----------

def gradnorm(x, w, b, num_classes):
    fc = torch.nn.Linear(*w.shape[::-1], bias=True).to(DEVICE)
    with torch.no_grad():
        fc.weight.copy_(torch.from_numpy(w))
        fc.bias.copy_(torch.from_numpy(b))
    x = torch.from_numpy(x).float().to(DEVICE)
    logsoftmax = torch.nn.LogSoftmax(dim=-1).to(DEVICE)
    confs = []
    for i in x:
        targets = torch.ones((1, num_classes), device=DEVICE)
        fc.zero_grad(set_to_none=True)
        loss = torch.mean(torch.sum(-targets * logsoftmax(fc(i[None])), dim=-1))
        loss.backward()
        layer_grad_norm = torch.sum(torch.abs(fc.weight.grad)).detach().cpu().numpy()
        confs.append(layer_grad_norm)
    return np.array(confs)

# ---------- NNGuide ----------

def baseconf_msp_torch(logits: torch.Tensor):
    return logits.softmax(dim=-1).amax(dim=-1)

def baseconf_maxlogit_torch(logits: torch.Tensor):
    return logits.max(dim=-1).values

def baseconf_energy_torch(logits: torch.Tensor):
    m = logits.max(dim=-1, keepdim=True).values
    return (m + (logits - m).exp().sum(dim=-1, keepdim=True).log()).squeeze(-1)

@torch.no_grad()
def collect_feats_and_conf(model, loader, conf_fn, arch):
    feats, confs = [], []
    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits, logits_aux, penult, _ = forward_adapt(model, x, arch)
        feats.append(penult)
        confs.append(conf_fn(logits))
    Z = torch.cat(feats, dim=0)
    S = torch.cat(confs, dim=0).unsqueeze(1)
    return Z, S

@torch.no_grad()
def nnguide_scores(z, s, Z_bank, S_bank, k):
    z_norm = F.normalize(z, p=2, dim=-1)
    Z_bank_norm = F.normalize(Z_bank, p=2, dim=-1)
    Z_weighted = Z_bank_norm * S_bank
    sim = z_norm @ Z_weighted.t()
    g = sim.topk(k, dim=1).values.mean(dim=1)
    guided = s * g
    return guided, -guided

@torch.no_grad()
def nnguide_score_loader(model, loader, Z_bank, S_bank, k, conf_fn, arch, return_ood_higher=True):
    all_guided = []
    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits, _, penult, _ = forward_adapt(model, x, arch)
        s = conf_fn(logits)
        guided_id, guided_ood = nnguide_scores(penult, s, Z_bank, S_bank, k)
        all_guided.append(guided_ood if return_ood_higher else guided_id)
    return torch.cat(all_guided, dim=0).cpu().numpy()

# ---------- NECO ----------

def neco(feature_id_train, feature_id_val, feature_ood, logit_id_val, logit_ood, model_architecture_type, neco_dim):
    ss = StandardScaler()
    complete_vectors_train = ss.fit_transform(feature_id_train)
    complete_vectors_test = ss.transform(feature_id_val)
    complete_vectors_ood = ss.transform(feature_ood)
    pca_estimator = PCA(feature_id_train.shape[1])
    _ = pca_estimator.fit_transform(complete_vectors_train)
    cls_test_reduced_all = pca_estimator.transform(complete_vectors_test)
    cls_ood_reduced_all = pca_estimator.transform(complete_vectors_ood)
    score_id_maxlogit = logit_id_val.max(axis=-1)
    score_ood_maxlogit = logit_ood.max(axis=-1)
    if model_architecture_type in ['deit', 'swin']:
        complete_vectors_train = feature_id_train
        complete_vectors_test = feature_id_val
        complete_vectors_ood = feature_ood
    cls_test_reduced = cls_test_reduced_all[:, :neco_dim]
    cls_ood_reduced = cls_ood_reduced_all[:, :neco_dim]

    def _ratio_rows(Afull, Ared):
        out = []
        for i in range(Ared.shape[0]):
            sc_complet = LA.norm(Afull[i, :])
            sc = LA.norm(Ared[i, :])
            out.append(sc / sc_complet if sc_complet > 0 else 0.0)
        return np.array(out)

    l_ID = _ratio_rows(complete_vectors_test, cls_test_reduced)
    l_OOD = _ratio_rows(complete_vectors_ood, cls_ood_reduced)
    score_id, score_ood = l_ID, l_OOD
    if model_architecture_type != 'resnet':
        score_id *= score_id_maxlogit
        score_ood *= score_ood_maxlogit
    auc = Metrics.auc(-score_id, -score_ood)
    fpr, _ = Metrics.fpr_recall(-score_id, -score_ood, RECALL)
    print(f'NECO: auroc {auc:.2%}, fpr {fpr:.2%}')

# ---------- ReAct ----------

def react(feature_id_val, feature_ood, clip, w, b, name):
    logit_id_clip = np.clip(feature_id_val, a_min=None, a_max=clip) @ w.T + b
    logit_ood_clip = np.clip(feature_ood, a_min=None, a_max=clip) @ w.T + b
    score_id = logsumexp(logit_id_clip, axis=-1)
    score_ood = logsumexp(logit_ood_clip, axis=-1)
    auc = Metrics.auc(-score_id, -score_ood)
    fpr, _ = Metrics.fpr_recall(-score_id, -score_ood, RECALL)
    print(f"ReAct: {name} auroc {auc:.2%}, fpr {fpr:.2%}")

# ---------- ViM ----------

import numpy as np
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from numpy.linalg import pinv, norm
from scipy.special import logsumexp

EPS = 1e-12  # numeric guard

def vim(feature_id_train, feature_id_val, feature_ood,
        logit_id_train, logit_id_val, ood_logits,
        name, model_architecture_type, model_name, u):
    """
    ViM score (OOD-higher): score = v(x) - energy(x)
      - v(x): L2 norm of projection onto covariance null-space (low-variance ID directions)
      - energy(x): logsumexp(logits(x))
    """
    # ---- dimensions & keep rule ----
    D = int(feature_id_train.shape[-1])
    d_keep = max(1, min(int(0.6 * D), D - 1))   # <=— enforced rule
    tail = D - d_keep                            # null-space size >= 1

    # ---- center by u ----
    u = np.asarray(u).reshape(1, -1)
    Xtr = feature_id_train - u
    Xva = feature_id_val   - u
    Xoo = feature_ood      - u

    # ---- covariance & eigendecomp (stable) ----
    try:
        C = EmpiricalCovariance(assume_centered=True).fit(Xtr).covariance_
    except Exception:
        C = LedoitWolf(assume_centered=True).fit(Xtr).covariance_

    # symmetric → eigh; sort desc
    eig_vals, eig_vecs = np.linalg.eigh(C)
    idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, idx]

    # null-space basis = bottom eigenvectors
    NS = np.ascontiguousarray(eig_vecs[:, d_keep:])  # [D, tail], tail>=1

    def vnorm(Z):
        V = Z @ NS          # [N, tail]
        return norm(V, axis=-1)

    v_tr  = vnorm(Xtr)
    v_val = vnorm(Xva)
    v_ood = vnorm(Xoo)

    # ---- scale v by alpha (guarded) ----
    mean_v = max(float(v_tr.mean()), EPS)
    alpha  = float(logit_id_train.max(axis=-1).mean() / mean_v)
    v_val *= alpha
    v_ood *= alpha

    # ---- energies ----
    energy_id_val = logsumexp(logit_id_val, axis=-1)
    energy_ood    = logsumexp(ood_logits,    axis=-1)

    # ---- final scores (OOD-higher) ----
    score_id  = v_val - energy_id_val
    score_ood = v_ood - energy_ood

    # sanitize (just in case)
    score_id  = np.nan_to_num(score_id,  neginf=-1e9, posinf=1e9)
    score_ood = np.nan_to_num(score_ood, neginf=-1e9, posinf=1e9)

    auc = Metrics.auc(score_id, score_ood)
    fpr, _ = Metrics.fpr_recall(score_id, score_ood, RECALL)
    print(f"ViM: {name} auroc {auc:.2%}, fpr {fpr:.2%}")


# ---------- Cosine-prototype (cosine_layers) ----------
from models import _get_resnet_intermediates  # for taps

def _gap(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=(2, 3))

# @torch.no_grad()
# def _iter_feats_layers(model: torch.nn.Module, loader, arch: str, LAYERS: List[int]):
#      for x, y in loader:
#         x = x.to(DEVICE, non_blocking=True)
#         y = y.to(DEVICE, non_blocking=True)
#         logits, logits_aux, penult, extra = forward_adapt(model, x, arch)
#         # if arch.startswith("resnet-50"):
#         #     outs = _get_resnet_intermediates(model, x)
#         # else:
#         #     if extra is None:
#         #         raise RuntimeError("DenseNet path did not return intermediates for cosine_layers.")
#         outs = extra
#         feats = []
#         for li in LAYERS:  # 1->out2, 2->out3, 3->out4
#             feats.append(F.normalize(_gap(outs[li]), dim=1))
#         yield feats, y
@torch.no_grad()
def _iter_feats_layers(model, loader, arch: str, LAYERS: List[int]):
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits, logits_aux, penult, extra = forward_adapt(model, x, arch)

        outs = extra  # for resnet50 we just filled this dict
        feats = []
        for li in LAYERS:  # e.g., 2->out3, 3->out4, 5->penult
            z = outs[li]
            # If it's a feature map (N,C,H,W), GAP it; if it's already (N,C), keep it.
            if z.dim() == 4:
                z = _gap(z)
            feats.append(F.normalize(z, dim=1))
        yield feats, y

@torch.no_grad()
def cosine_build_prototypes(model, calib_loader, arch: str, LAYERS: List[int], num_classes: int):
    first_feats, first_y = next(_iter_feats_layers(model, calib_loader, arch, LAYERS))
    dims = [f.size(1) for f in first_feats]
    sums: Dict[int, List[torch.Tensor]] = {li: [torch.zeros(dims[k], device=DEVICE) for _ in range(num_classes)]
                                           for k, li in enumerate(LAYERS)}
    counts: Dict[int, torch.Tensor] = {li: torch.zeros(num_classes, device=DEVICE) for li in LAYERS}

    def update(feats_list, y):
        for k, li in enumerate(LAYERS):
            Fk = feats_list[k]
            for cls in range(num_classes):
                mask = (y == cls)
                if mask.any():
                    sums[li][cls] += Fk[mask].sum(dim=0)
                    counts[li][cls] += mask.sum()

    update(first_feats, first_y)
    for feats_list, y in _iter_feats_layers(model, calib_loader, arch, LAYERS):
        update(feats_list, y)

    prototypes: Dict[int, torch.Tensor] = {}
    for li in LAYERS:
        plist = []
        for cls in range(num_classes):
            if counts[li][cls] > 0:
                p = sums[li][cls] / counts[li][cls]
                p = F.normalize(p, dim=0)
            else:
                p = F.normalize(torch.randn_like(sums[li][0]), dim=0)
            plist.append(p)
        prototypes[li] = torch.stack(plist, dim=0)
    return prototypes

@torch.no_grad()
def cosine_score_loader(model, loader, arch: str, prototypes: Dict[int, torch.Tensor],
                        LAYERS: List[int], layer_weights: Dict[int, float]):
    wsum = sum(layer_weights[li] for li in LAYERS)
    scores_out = []
    for feats_list, _ in _iter_feats_layers(model, loader, arch, LAYERS):
        per_layer_max = []
        for k, li in enumerate(LAYERS):
            f = feats_list[k]
            P = prototypes[li].to(f.device)
            cos = f @ P.t()
            max_cos, _ = cos.max(dim=1)
            per_layer_max.append(layer_weights[li] * max_cos)
        avg_max_cos = per_layer_max[0]
        for t in per_layer_max[1:]:
            avg_max_cos = avg_max_cos + t
        avg_max_cos = avg_max_cos / wsum
        ood_score = 1.0 - avg_max_cos
        scores_out.append(ood_score.detach().cpu())
    return torch.cat(scores_out, dim=0).numpy()

def method_cosine_layers(model, arch: str,  id_val_loader, ood_loader,
                         num_classes: int, cos_layers: List[int], layer_weights: Dict[int, float],
                        calib_loader):
    # calib_subset = calib_subset_maker(id_train_set, calib_per_class, num_classes)
    # calib_loader = torch.utils.data.DataLoader(calib_subset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    print("[cosine_layers] Building prototypes...")
    protos = cosine_build_prototypes(model, calib_loader, arch, cos_layers, num_classes)
    print("[cosine_layers] Scoring ID and OOD...")
    id_scores = cosine_score_loader(model, id_val_loader, arch, protos, cos_layers, layer_weights)
    ood_scores = cosine_score_loader(model, ood_loader, arch, protos, cos_layers, layer_weights)
    auc = Metrics.auc(id_scores, ood_scores)
    fpr, _ = Metrics.fpr_recall(id_scores, ood_scores, RECALL)
    print(f"cosine_layers: AUROC {auc:.2%}, FPR@95 {fpr:.2%}")
    return fpr,auc