import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Style
# ----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Serif",
})

label_fontsize = 22
tick_fontsize = 20
legend_fontsize = 20

# Pastel palette similar to your reference figure (one color per variant)
PALETTE_10 = ["#8c8c8c", "#4b3fbf", "#4c86ff", "#7fd0ff", "#a7eadc", "#d8f2df",
              "#c6e6ff", "#b9f3ea", "#eef7c7", "#f2d7d7"]
PALETTE_6 = ["#8c8c8c", "#4b3fbf", "#4c86ff", "#7fd0ff", "#a7eadc", "#d8f2df"]
PALETTE_5 = ["#8c8c8c", "#4b3fbf", "#4c86ff", "#7fd0ff", "#a7eadc"]
PALETTE_4 = ["#8c8c8c", "#4b3fbf", "#4c86ff", "#7fd0ff"]
PALETTE_3 = ["#8c8c8c", "#4b3fbf", "#4c86ff"]
PALETTE_2 = ["#8c8c8c", "#4b3fbf"]

def _palette_for(n):
    if n <= 2:
        return PALETTE_2[:n]
    if n == 3:
        return PALETTE_3
    if n <= 4:
        return PALETTE_4[:n]
    if n <= 5:
        return PALETTE_5[:n]
    if n <= 6:
        return PALETTE_6[:n]
    return PALETTE_10[:n]  # enough for alpha sweeps like 0.1..0.7

# ----------------------------
# Data (from your summaries)
# ----------------------------
ABLATIONS = {
    "tiny-imagenet-200": {
        "alpha_sweep": [
            (0.1, 0.9050, 0.3919),
            (0.2, 0.9047, 0.3811),
            (0.3, 0.9069, 0.3738),
            (0.4, 0.9032, 0.3814),
            (0.5, 0.8936, 0.4133),
            (0.6, 0.8910, 0.4186),
            (0.7, 0.8928, 0.4141),
        ],
        "score_components": [
            ("s_res",           0.8936, 0.4133),
            ("a_only",          0.9473, 0.2263),
            ("s_ratio",         0.5216, 0.9080),
            ("res_plus_align",  0.9821, 0.0755),
            ("res_plus_ratio",  0.9125, 0.3755),
            ("full",            0.9760, 0.1053),
        ],
        "beta_calibration": [
            ("beta_0_no_align",   0.8936, 0.4133),
            ("beta_1_fixed",      0.9581, 0.1924),
            ("beta_median_ratio", 0.9274, 0.3015),
            ("beta_std_ratio",    0.9516, 0.2186),
        ],
        "subspace_direction": [
            ("discarded_subspace", 0.8936, 0.4133),
            ("kept_subspace",      0.8353, 0.5626),
        ],
    },

    "places": {
        "alpha_sweep": [
            (0.1, 0.9615, 0.1785),
            (0.2, 0.9690, 0.1511),
            (0.3, 0.9690, 0.1436),
            (0.4, 0.9646, 0.1652),
            (0.5, 0.9610, 0.1937),
            (0.6, 0.9551, 0.2145),
            (0.7, 0.9563, 0.2083),
        ],
        "score_components": [
            ("s_res",           0.9610, 0.1937),
            ("a_only",          0.9493, 0.2134),
            ("s_ratio",         0.6127, 0.8501),
            ("res_plus_align",  0.9986, 0.0049),
            ("res_plus_ratio",  0.9725, 0.1445),
            ("full",            0.9973, 0.0112),
        ],
        "beta_calibration": [
            ("beta_0_no_align",   0.9610, 0.1937),
            ("beta_1_fixed",      0.9782, 0.1004),
            ("beta_median_ratio", 0.9730, 0.1287),
            ("beta_std_ratio",    0.9799, 0.0932),
        ],
        "subspace_direction": [
            ("discarded_subspace", 0.9610, 0.1937),
            ("kept_subspace",      0.8804, 0.4388),
        ],
    },

    "textures": {
        "alpha_sweep": [
            (0.1, 0.9945, 0.0267),
            (0.2, 0.9945, 0.0281),
            (0.3, 0.9941, 0.0297),
            (0.4, 0.9939, 0.0323),
            (0.5, 0.9933, 0.0343),
            (0.6, 0.9933, 0.0346),
            (0.7, 0.9935, 0.0334),
        ],
        "score_components": [
            ("s_res",           0.9933, 0.0343),
            ("a_only",          0.9539, 0.1745),
            ("s_ratio",         0.8455, 0.6498),
            ("res_plus_align",  0.9982, 0.0032),
            ("res_plus_ratio",  0.9978, 0.0086),
            ("full",            1.0000, 0.0000),
        ],
        "beta_calibration": [
            ("beta_0_no_align",   0.9933, 0.0343),
            ("beta_1_fixed",      0.9942, 0.0275),
            ("beta_median_ratio", 0.9951, 0.0234),
            ("beta_std_ratio",    0.9957, 0.0193),
        ],
        "subspace_direction": [
            ("discarded_subspace", 0.9933, 0.0343),
            ("kept_subspace",      0.9530, 0.2097),
        ],
    },

    "cifar100": {
        "alpha_sweep": [
            (0.1, 0.8904, 0.4446),
            (0.2, 0.8821, 0.4789),
            (0.3, 0.8778, 0.4883),
            (0.4, 0.8767, 0.4947),
            (0.5, 0.8721, 0.5107),
            (0.6, 0.8708, 0.5176),
            (0.7, 0.8707, 0.5181),
        ],
        "score_components": [
            ("s_res",           0.8721, 0.5107),
            ("a_only",          0.8968, 0.4081),
            ("s_ratio",         0.5057, 0.9418),
            ("res_plus_align",  0.9748, 0.1178),
            ("res_plus_ratio",  0.8893, 0.5048),
            ("full",            0.9637, 0.1803),
        ],
        "beta_calibration": [
            ("beta_0_no_align",   0.8721, 0.5107),
            ("beta_1_fixed",      0.9229, 0.3514),
            ("beta_median_ratio", 0.9008, 0.4203),
            ("beta_std_ratio",    0.9209, 0.3555),
        ],
        "subspace_direction": [
            ("discarded_subspace", 0.8721, 0.5107),
            ("kept_subspace",      0.8339, 0.5642),
        ],
    },
}

DATASETS = ["tiny-imagenet-200", "places", "textures", "cifar100"]
DATASET_DISPLAY = {
    "tiny-imagenet-200": "TIN",
    "places": "Places",
    "textures": "Textures",
    "cifar100": "CIFAR-100",
}

# ----------------------------
# Variant selection / filtering
# ----------------------------
VARIANT_ORDER = {
    "score_components": ["s_res", "a_only", "res_plus_align"],
    "beta_calibration": ["beta_0_no_align", "beta_1_fixed", "beta_median_ratio", "beta_std_ratio"],
    "subspace_direction": ["discarded_subspace", "kept_subspace"],
}

# ----------------------------
# Pretty legend labels
# ----------------------------
LEGEND_LABELS = {
    "score_components": {
        "s_res": r"$s_{\perp}$",
        "a_only": r"$a$",
        "res_plus_align": r"$s_{CREWA}$",
    },
    "beta_calibration": {
        "beta_0_no_align": r"$\beta=0$",
        "beta_1_fixed": r"$\beta=1$",
        "beta_median_ratio": r"$\beta=\mathrm{median}(a)/\mathrm{median}(s_{\mathrm{res}})$",
        "beta_std_ratio": r"$\beta=\mathrm{std}(a)/\mathrm{std}(s_{\mathrm{res}})$",
    },
    "subspace_direction": {
        "kept_subspace": r"$V_{\parallel}$",
        "discarded_subspace": r"$V_{\perp}$",
    },
}

LEGEND_KWARGS_BY_ABLATION = {
    # Example placements (x,y are in axes fraction coords: (0,0)=bottom left, (1,1)=top right)
    "alpha_sweep":        dict(loc="center",      bbox_to_anchor=(0.62, 0.60)),
    "score_components":   dict(loc="center",      bbox_to_anchor=(0.50, 0.8)),
    "beta_calibration":   dict(loc="upper left",  bbox_to_anchor=(0.15, 0.95)),
    "subspace_direction": dict(loc="center",      bbox_to_anchor=(0.6, 0.8)),
}

# ----------------------------
# Helper: union of alpha values across datasets
# ----------------------------
def _collect_all_alphas():
    alphas = set()
    for ds in DATASETS:
        for (a, _, _) in ABLATIONS[ds]["alpha_sweep"]:
            alphas.add(float(a))
    return sorted(alphas)

def _alpha_to_str(a):
    # stable pretty formatting (0.1, 0.2, ...)
    return f"{a:.1f}".rstrip("0").rstrip(".")

# ----------------------------
# Plot function (no titles)
# ----------------------------
def plot_ablation_grouped_bars(ablation_key, outpath=None, as_percent=True):
    """
    One figure per ablation, two panels:
      left: FPR95, right: AUROC
    Grouped bars per dataset, one color per variant.

    Change requested:
    - alpha_sweep now uses the UNION of alpha values across ALL datasets,
      not just DATASETS[0].
      Missing (dataset, alpha) pairs are left as NaN and simply not drawn.
    """

    if ablation_key == "alpha_sweep":
        # union alpha grid across datasets
        alpha_grid = _collect_all_alphas()
        variants = [_alpha_to_str(a) for a in alpha_grid]

        # build dict per dataset: alpha -> (auroc, fpr)
        au = np.full((len(alpha_grid), len(DATASETS)), np.nan, dtype=float)
        fp = np.full((len(alpha_grid), len(DATASETS)), np.nan, dtype=float)

        for j, ds in enumerate(DATASETS):
            d = {float(a): (auroc, fpr95) for (a, auroc, fpr95) in ABLATIONS[ds]["alpha_sweep"]}
            for i, a in enumerate(alpha_grid):
                if a in d:
                    au[i, j] = d[a][0]
                    fp[i, j] = d[a][1]

        legend_labels = [f"α={v}" for v in variants]

    else:
        inferred = [name for (name, _, _) in ABLATIONS[DATASETS[0]][ablation_key]]
        variants = VARIANT_ORDER.get(ablation_key, inferred)

        au, fp = [], []
        for v in variants:
            au_row, fp_row = [], []
            for ds in DATASETS:
                d = {name: (auroc, fpr95) for (name, auroc, fpr95) in ABLATIONS[ds][ablation_key]}
                if v not in d:
                    raise KeyError(f"Variant '{v}' not found for dataset '{ds}' in ablation '{ablation_key}'.")
                au_row.append(d[v][0])
                fp_row.append(d[v][1])
            au.append(au_row)
            fp.append(fp_row)

        au = np.array(au, dtype=float)
        fp = np.array(fp, dtype=float)

        if ablation_key in LEGEND_LABELS:
            label_map = LEGEND_LABELS[ablation_key]
            legend_labels = [label_map.get(v, v) for v in variants]
        else:
            legend_labels = variants  # plain text keeps underscores safe

    # percent scaling
    if as_percent:
        au_plot = 100.0 * au
        fp_plot = 100.0 * fp
        ylab_au = "AUROC"
        ylab_fp = "FPR95"
    else:
        au_plot = au
        fp_plot = fp
        ylab_au = "AUROC"
        ylab_fp = "FPR95"

    n_var = len(variants)
    n_ds = len(DATASETS)
    colors = _palette_for(n_var)

    x = np.arange(n_ds)
    total_width = 0.78
    bar_w = total_width / n_var
    offsets = (np.arange(n_var) - (n_var - 1) / 2.0) * bar_w

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), dpi=150, constrained_layout=True)

    # Left: FPR95
    ax = axes[0]
    for i in range(n_var):
        ax.bar(
            x + offsets[i],
            fp_plot[i],
            width=bar_w,
            color=colors[i],
            edgecolor="none",
            label=legend_labels[i],
        )
    ax.set_ylabel(ylab_fp, fontsize=label_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_DISPLAY.get(ds, ds) for ds in DATASETS], fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    # Right: AUROC
    ax = axes[1]
    for i in range(n_var):
        ax.bar(
            x + offsets[i],
            au_plot[i],
            width=bar_w,
            color=colors[i],
            edgecolor="none",
            label=legend_labels[i],
        )
    ax.set_ylabel(ylab_au, fontsize=label_fontsize, labelpad=-10)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_DISPLAY.get(ds, ds) for ds in DATASETS], fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    lk = LEGEND_KWARGS_BY_ABLATION.get(ablation_key, dict(loc="upper right", bbox_to_anchor=(0.98, 0.98)))

    axes[0].legend(
        frameon=True,
        fontsize=legend_fontsize,
        bbox_transform=axes[0].transAxes,  # makes bbox_to_anchor use axes-fraction coords
        borderaxespad=0.0,
        **lk,
    )

    if outpath is None:
        outpath = f"{ablation_key}_groupedbars.png"

    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Make all figures
# ----------------------------
def make_all_figures(as_percent=True):
    plot_ablation_grouped_bars("alpha_sweep",        outpath="alpha_sweep_style.png",        as_percent=as_percent)
    plot_ablation_grouped_bars("score_components",   outpath="score_components_style.png",   as_percent=as_percent)
    plot_ablation_grouped_bars("beta_calibration",   outpath="beta_calibration_style.png",   as_percent=as_percent)
    plot_ablation_grouped_bars("subspace_direction", outpath="subspace_direction_style.png", as_percent=as_percent)

if __name__ == "__main__":
    make_all_figures(as_percent=True)