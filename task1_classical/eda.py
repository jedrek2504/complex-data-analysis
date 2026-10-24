"""Task 1: exploratory analysis of the datasets.

Outputs (results/task1/): eda_summary.csv, tables/eda_summary.tex,
figures/eda_pca.pdf, figures/eda_feature_shift.pdf, figures/eda_skewness.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import ROOT
from common.anomaly_types import anomaly_kind
from common.datasets import DISPLAY_NAMES, SPECS, TASK1_DATASETS, load_dataset
from common.latex import num, tabular
from common.plotting import BLUES, NORMAL_COLOR, TYPE_COLORS, TYPE_LABELS, save, setup

OUT = ROOT / "results" / "task1"
SOURCE_NAMES = {"adbench": "ADBench", "odds": "ODDS", "dami": "DAMI"}


def summarise(data):
    X, y = data.X, data.y
    Z = StandardScaler().fit_transform(X)
    corr = np.corrcoef(Z, rowvar=False)
    np.fill_diagonal(corr, 0)
    pca = PCA().fit(Z)
    kinds = anomaly_kind(data.name)
    return {
        "dataset": data.name,
        "source": SPECS[data.name].source,
        "n": len(X),
        "features": X.shape[1],
        "anomalies": int(y.sum()),
        "contamination": y.mean(),
        "global": int((kinds == "global").sum()),
        "local": int((kinds == "local").sum()),
        "missing": int(np.isnan(X).sum()),
        "integer_features": int(np.all(X == np.round(X), axis=0).sum()),
        "mean_abs_skew": float(np.mean(np.abs(skew(X, axis=0)))),
        "share_beyond_3sd": float(np.mean(np.abs(Z) > 3)),
        "max_abs_corr": float(np.abs(corr).max()),
        "pcs_90_variance": int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.9) + 1),
    }


def cohens_d(X, y):
    a, b = X[y == 1], X[y == 0]
    pooled = np.sqrt((a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1)) / 2)
    return (a.mean(axis=0) - b.mean(axis=0)) / np.maximum(pooled, 1e-12)


def main():
    setup()
    datasets = [load_dataset(name) for name in TASK1_DATASETS]
    summary = pd.DataFrame([summarise(d) for d in datasets])
    summary.to_csv(OUT / "eda_summary.csv", index=False)

    table = pd.DataFrame({
        "Zbiór": [DISPLAY_NAMES[n] for n in summary.dataset],
        "Źródło": [SOURCE_NAMES[s] for s in summary.source],
        "$n$": summary.n.astype(str),
        "$d$": summary.features.astype(str),
        "Anomalie": summary.anomalies.astype(str),
        "Udział [\\%]": [num(100 * c, 1) for c in summary.contamination],
        "Glob./lok.": [f"{g}/{loc}" for g, loc in zip(summary["global"], summary.local)],
        "Braki": summary.missing.astype(str),
        "$\\overline{|\\gamma_1|}$": [num(v, 2) for v in summary.mean_abs_skew],
        "$|z|>3$ [\\%]": [num(100 * v, 2) for v in summary.share_beyond_3sd],
        "max $|r|$": [num(v, 2) for v in summary.max_abs_corr],
        "PC 90\\%": summary.pcs_90_variance.astype(str),
    })
    tabular(table, OUT / "tables" / "eda_summary.tex", align="ll" + "r" * 10)

    # 2-D PCA projections
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.9))
    for ax, d in zip(axes, datasets):
        Z = PCA(2).fit_transform(StandardScaler().fit_transform(d.X))
        ax.scatter(*Z[d.y == 0].T, s=7, c=NORMAL_COLOR, lw=0, label="normalne")
        kinds = anomaly_kind(d.name)
        for t, color in TYPE_COLORS.items():
            m = kinds == t
            ax.scatter(*Z[m].T, s=14, c=color, lw=0.4, ec="white",
                       label=f"anomalie {TYPE_LABELS[t]}")
        ax.set_title(DISPLAY_NAMES[d.name])
        ax.set_xlabel("PC1")
        ax.set_xticks([]), ax.set_yticks([])
    axes[0].set_ylabel("PC2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    save(fig, OUT / "figures" / "eda_pca")

    # standardised mean difference of every feature (anomalies vs normal)
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.6), sharey=True)
    for ax, d in zip(axes, datasets):
        effect = np.sort(np.abs(cohens_d(d.X, d.y)))[::-1]
        ax.bar(np.arange(len(effect)), effect, color="#3987e5", width=0.8)
        ax.axhline(0.8, color="#52514e", lw=0.8, ls="--")
        ax.set_title(DISPLAY_NAMES[d.name])
        ax.set_xlabel("cechy (posortowane)")
        ax.set_xticks([])
    axes[0].set_ylabel("|d Cohena|")
    save(fig, OUT / "figures" / "eda_feature_shift")

    # per-feature skewness distribution
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    data = [skew(d.X, axis=0) for d in datasets]
    parts = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True,
                       medianprops={"color": "#0d366b"})
    for box in parts["boxes"]:
        box.set(facecolor=BLUES(0.25), edgecolor="#256abf")
    ax.set_yticklabels([DISPLAY_NAMES[d.name] for d in datasets])
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_xlabel("skośność cech $\\gamma_1$")
    save(fig, OUT / "figures" / "eda_skewness")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
