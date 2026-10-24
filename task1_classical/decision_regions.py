"""Task 1: decision regions of the five detectors in a 2-D view of Cardio.

The training part of the first split of Cardio is standardised and projected
onto its first two principal components; every detector is fitted in this
plane. Each one flags 10 % of the training points (contamination = 0.1;
nu = 0.1 for OCSVM; for DBSCAN eps is the 0.9 quantile of the k-distance),
close to the 9.6 % of anomalies in Cardio, which makes the shapes of the
regions directly comparable. Anomalies are coloured by their global / local
label from the full feature space (common/anomaly_types.py), so the figure
also shows how much of that structure survives the projection.

Output: results/task1/figures/decision_regions_2d.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from common.anomaly_types import anomaly_kind
from common.detectors import build
from common.metrics import decision_metrics
from common.plotting import NORMAL_COLOR, TYPE_COLORS, TYPE_LABELS, save, setup
from common.protocol import make_experiment
from task1_classical.run_grid import OUT

DATASET = "cardio"
CONFIGS = [
    ("IForest", {"n_estimators": 200, "contamination": 0.1}),
    ("LOF", {"n_neighbors": 20, "contamination": 0.1}),
    ("COF", {"n_neighbors": 20, "contamination": 0.1}),
    ("OCSVM", {"kernel": "rbf", "nu": 0.1, "gamma_factor": 1.0}),
    ("DBSCAN", {"min_samples": 5, "eps_quantile": 0.9}),
]


def main():
    setup()
    exp = make_experiment(DATASET, seed=0)
    X = PCA(2, random_state=0).fit_transform(exp.X_train)
    y = exp.y_train
    types = anomaly_kind(DATASET)[exp.idx_train]
    lo, hi = np.quantile(X, 0.002, axis=0) - 0.5, np.quantile(X, 0.998, axis=0) + 0.5
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 300), np.linspace(lo[1], hi[1], 300))
    grid = np.c_[gx.ravel(), gy.ravel()]

    fig, axes = plt.subplots(1, 5, figsize=(13, 2.9), sharey=True)
    for ax, (method, params) in zip(axes, CONFIGS):
        det = build(method, **params).fit(X)
        region = det.predict(grid).reshape(gx.shape)
        ax.contourf(gx, gy, region, levels=[0.5, 1.5], colors=["#f6dcd0"])
        ax.contour(gx, gy, region, levels=[0.5], colors=["#c2410c"], linewidths=0.8)
        ax.scatter(*X[types == "normal"].T, s=4, c=NORMAL_COLOR, lw=0)
        for t in ("local", "global"):
            m = types == t
            ax.scatter(*X[m].T, s=14, c=TYPE_COLORS[t], ec="white", lw=0.4,
                       label=f"anomalie {TYPE_LABELS[t]}")
        pred = det.predict(X)
        rec = {t: pred[types == t].mean() for t in ("local", "global")}
        f1 = decision_metrics(y, pred)["f1"]
        ax.set_title(f"{method}\nF1 = {f1:.2f}\nR$_{{lok}}$ = {rec['local']:.2f},  "
                     f"R$_{{glob}}$ = {rec['global']:.2f}".replace(".", ","), fontsize=8.5)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlabel("PC1")
        ax.grid(False)
    axes[0].set_ylabel("PC2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    save(fig, OUT / "figures" / "decision_regions_2d")


if __name__ == "__main__":
    main()
