"""Task 1: description and exploratory analysis of the datasets.

Outputs (results/task1/):
  eda_summary.csv
  tables/datasets.tex        - one row per dataset (source, content, size)
  tables/sample_<name>.tex   - two example records of every dataset
  tables/eda_summary.tex     - shape indicators used later in the report
  figures/eda_pca.pdf        - 2-D PCA view of every dataset
  figures/eda_feature_shift.pdf - how strongly single features separate anomalies
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
from task1_classical.style import DATASET_STYLE, NORMAL, WIDTH, save, setup

OUT = ROOT / "results" / "task1"
SOURCE_NAMES = {"adbench": "ADBench", "odds": "ODDS", "dami": "DAMI"}
CONTENT = {
    "breastw": "biopsje piersi; anomalia = guz złośliwy",
    "mammography": "fragmenty mammogramów; anomalia = mikrozwapnienie",
    "wine": "analiza chemiczna win; anomalia = wino z odmiany 1",
    "heart": "badania kardiologiczne; anomalia = choroba serca",
    "cardio": "zapisy KTG płodu; anomalia = zapis patologiczny",
}
# Column names are given only where the column order is certain (it matches
# the original UCI files); the other files contain scaled, unnamed columns.
FEATURE_NAMES = {
    "breastw": ["grubość skupisk", "jednorodność rozmiaru", "jednorodność kształtu",
                "przyleganie brzeżne", "rozmiar komórki nabł."],
    "wine": ["alkohol", "kwas jabłkowy", "popiół", "zasadowość popiołu", "magnez"],
    "heart": ["wiek", "płeć", "typ bólu w klatce", "ciśnienie spoczynkowe",
              "cholesterol"],
}
SHOWN_FEATURES = 5


def summarise(data):
    Z = StandardScaler().fit_transform(data.X)
    corr = np.corrcoef(Z, rowvar=False)
    np.fill_diagonal(corr, 0)
    pca = PCA().fit(Z)
    return {
        "dataset": data.name,
        "n": len(data.X),
        "features": data.X.shape[1],
        "anomalies": int(data.y.sum()),
        "contamination": data.y.mean(),
        "missing": int(np.isnan(data.X).sum()),
        "mean_abs_skew": float(np.mean(np.abs(skew(data.X, axis=0)))),
        "share_beyond_3sd": float(np.mean(np.abs(Z) > 3)),
        "max_abs_corr": float(np.nanmax(np.abs(corr))),
        "pcs_90_variance": int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.9) + 1),
    }


def cohens_d(X, y):
    a, b = X[y == 1], X[y == 0]
    pooled = np.sqrt((a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1)) / 2)
    return (a.mean(axis=0) - b.mean(axis=0)) / np.maximum(pooled, 1e-12)


def table_datasets(summary):
    table = pd.DataFrame({
        "Zbiór danych": [DISPLAY_NAMES[n] for n in summary.dataset],
        "Źródło": [SOURCE_NAMES[SPECS[n].source] for n in summary.dataset],
        "Co zawiera": [CONTENT[n] for n in summary.dataset],
        "Obserwacje": summary.n.map(lambda v: f"{v:,}".replace(",", "\\,")),
        "Cechy": summary.features.astype(str),
        "Anomalie": [f"{a} ({num(100 * c, 1)}\\,\\%)"
                     for a, c in zip(summary.anomalies, summary.contamination)],
    })
    tabular(table, OUT / "tables" / "datasets.tex", align="llL{5.2cm}rrr")


def table_samples(data):
    """First normal and first anomalous record, the first few features."""
    rows = [np.flatnonzero(data.y == 0)[0], np.flatnonzero(data.y == 1)[0]]
    names = FEATURE_NAMES.get(data.name,
                              [f"cecha {i + 1}" for i in range(SHOWN_FEATURES)])
    names = ["\\makecell{" + n.replace(" ", "\\\\", 1) + "}" if " " in n else n
             for n in names]

    def digits(j):
        return 0 if np.all(data.X[:, j] == np.round(data.X[:, j])) else 2

    table = pd.DataFrame({
        name: [num(data.X[r, j], digits(j)) for r in rows] for j, name in enumerate(names)})
    table["\\dots"] = "\\dots"
    table["Anomalia"] = ["nie" if data.y[r] == 0 else "tak" for r in rows]
    tabular(table, OUT / "tables" / f"sample_{data.name}.tex",
            align="r" * (len(names) + 1) + "c")


def table_summary(summary):
    table = pd.DataFrame({
        "Zbiór danych": [DISPLAY_NAMES[n] for n in summary.dataset],
        "\\makecell[r]{Średnia\\\\skośność cech}": [num(v, 2) for v in summary.mean_abs_skew],
        "\\makecell[r]{Wartości\\\\skrajne [\\%]}": [
            num(100 * v, 2) for v in summary.share_beyond_3sd],
        "\\makecell[r]{Najsilniejsza\\\\korelacja}": [num(v, 2) for v in summary.max_abs_corr],
        "\\makecell[r]{Składowe PCA na\\\\90\\,\\% informacji}": [
            f"{k} z {d}" for k, d in zip(summary.pcs_90_variance, summary.features)],
    })
    tabular(table, OUT / "tables" / "eda_summary.tex", align="lrrrr")


def fig_pca(datasets):
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 5.2))
    axes = axes.ravel()
    for ax, d in zip(axes, datasets):
        Z = PCA(2).fit_transform(StandardScaler().fit_transform(d.X))
        kinds = anomaly_kind(d.name)
        ax.scatter(*Z[d.y == 0].T, s=8, c=NORMAL, lw=0, label="normalne")
        ax.scatter(*Z[kinds == "global"].T, s=26, c="#0072B2", marker="D",
                   ec="black", lw=0.4, label="anomalie globalne")
        ax.scatter(*Z[kinds == "local"].T, s=24, c="#D55E00", marker="o",
                   ec="black", lw=0.4, label="anomalie lokalne")
        ax.set_title(DISPLAY_NAMES[d.name])
        ax.set_xticks([]), ax.set_yticks([])
    for ax in axes[3:5]:
        ax.set_xlabel("1. składowa PCA")
    for ax in axes[::3]:
        ax.set_ylabel("2. składowa PCA")
    axes[5].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc="center", markerscale=1.4, fontsize=10.5)
    save(fig, OUT / "figures" / "eda_pca")


def fig_feature_shift(datasets):
    fig, ax = plt.subplots(figsize=(WIDTH, 3.1))
    for d in datasets:
        effect = np.sort(np.abs(cohens_d(d.X, d.y)))[::-1]
        st = DATASET_STYLE[d.name]
        ax.plot(np.arange(1, len(effect) + 1), effect, color=st["color"],
                marker=st["marker"], ls=st["ls"], label=DISPLAY_NAMES[d.name])
    ax.axhline(0.8, color="#555555", lw=1, ls="--")
    ax.text(21.3, 0.85, "duża różnica", ha="right", va="bottom", fontsize=10,
            color="#555555")
    ax.set_xlabel("cechy uporządkowane od najbardziej różnicującej")
    ax.set_ylabel("różnica anomalie – normalne\n(w odchyleniach standardowych)")
    ax.set_xticks([1, 5, 9, 13, 17, 21])
    ax.legend(ncol=2)
    save(fig, OUT / "figures" / "eda_feature_shift")


def main():
    setup()
    datasets = [load_dataset(name) for name in TASK1_DATASETS]
    summary = pd.DataFrame([summarise(d) for d in datasets])
    summary.to_csv(OUT / "eda_summary.csv", index=False)
    table_datasets(summary)
    for d in datasets:
        table_samples(d)
    table_summary(summary)
    fig_pca(datasets)
    fig_feature_shift(datasets)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
