"""Task 1: what each detector found, shown in a 2-D PCA view of every dataset.

Split 0 of every dataset is used. The PCA projection is fitted on the
(standardised) training part and applied to the test part. For every detector
the configuration and the threshold chosen on the validation part are taken
from run_best (results/task1/test_predictions.csv.gz); the test part is only
drawn.

DBSCAN is also shown as a clustering method: test points are assigned to the
cluster of their nearest core point when it is closer than eps and to noise
otherwise (the detector's own rule).

The One-Class SVM region cannot be drawn in d dimensions, so it is
illustrated with the same kind of model fitted on the two PCA components:
(a) every dataset with the kernel, nu and gamma chosen on validation and the
threshold chosen on the validation part, (b) Cardio with the RBF kernel for
several gamma and nu values and the built-in threshold.

Outputs (results/task1/):
  figures/detected_<method>.pdf, figures/dbscan_clusters.pdf,
  figures/ocsvm_region.pdf, figures/ocsvm_region_params.pdf,
  tables/detected_counts.tex, tables/dbscan_clusters.tex
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import outcome_maps as maps
from common.datasets import DISPLAY_NAMES, TASK1_DATASETS
from common.detectors import build
from common.latex import num, tabular
from common.metrics import best_f1_threshold
from common.protocol import make_experiment
from task1_classical.config import METHODS
from task1_classical.run_grid import OUT
from task1_classical.style import FN_COLOR, WIDTH, save, setup

SEED = 0
REGION = "#cde2fb"
GAMMA_LABEL = {0.1: "0,1/$d$ (szerokie jądro)", 1.0: "1/$d$",
               10.0: "10/$d$ (wąskie jądro)"}
KERNEL_LABEL = {"rbf": "RBF", "sigmoid": "sigmoidalne", "poly": "wielomianowe",
                "linear": "liniowe"}


def panels(height=5.4):
    return maps.panels(len(TASK1_DATASETS), height)


def legend_cell(cax, ax, extra=()):
    maps.add_legend(None, cax, ax, extra)


def fig_detected(data, preds, method):
    fig, axes, cax = panels()
    rows = {}
    for ax, d in zip(axes, TASK1_DATASETS):
        exp, Z = data[d]["exp"], data[d]["Z_test"]
        p = preds[(preds.dataset == d) & (preds.method == method)].set_index("row")
        p = p.loc[exp.idx_test]
        assert np.array_equal(p.y.values, exp.y_test)
        pred = p.pred.values
        maps.outcome_points(ax, Z, exp.y_test, pred)
        c = maps.counts(exp.y_test, pred)
        ax.set_title(f"{DISPLAY_NAMES[d]}\nTP {c['TP']}, FN {c['FN']}, FP {c['FP']}",
                     fontsize=10.5)
        rows[d] = c
    legend_cell(cax, axes[0])
    save(fig, OUT / "figures" / f"detected_{method.lower()}")
    return rows


def table_counts(all_counts, data):
    rows = []
    for d in TASK1_DATASETS:
        cells = {"Zbiór danych": DISPLAY_NAMES[d],
                 "\\makecell[r]{Anomalie\\\\w teście}": str(int(data[d]["exp"].y_test.sum()))}
        for m in METHODS:
            c = all_counts[m][d]
            cells[m] = f"{c['TP']} / {c['FN']} / {c['FP']}"
        rows.append(cells)
    tabular(pd.DataFrame(rows), OUT / "tables" / "detected_counts.tex",
            align="lr" + "c" * len(METHODS))


def dbscan_clusters(data, best):
    fig, axes, cax = panels()
    palette = ["#56B4E9", "#009E73", "#CC79A7", "#F0E442", "#0072B2"]
    rows = []
    for ax, d in zip(axes, TASK1_DATASETS):
        exp, Z = data[d]["exp"], data[d]["Z_test"]
        params = json.loads(best[(best.dataset == d) & (best.method == "DBSCAN")].params.iloc[0])
        det = build("DBSCAN", **params).fit(exp.X_train)
        labels = det.model.labels_[det.model.core_sample_indices_]
        dist, idx = det.core_nn_.kneighbors(exp.X_test)
        cluster = np.where(dist[:, 0] <= det.eps, labels[idx[:, 0]], -1)
        y = exp.y_test
        groups = sorted(set(cluster) - {-1})
        for k in groups:
            m = cluster == k
            ax.scatter(*Z[m].T, s=10, c=palette[k % len(palette)], lw=0)
        noise = cluster == -1
        ax.scatter(*Z[noise].T, s=16, c="#555555", marker="s", lw=0)
        ax.scatter(*Z[y == 1].T, s=60, facecolors="none", edgecolors=FN_COLOR, lw=1.2)
        ax.set_title(f"{DISPLAY_NAMES[d]}\ngrupy: {len(groups)}, szum: {int(noise.sum())}",
                     fontsize=10.5)
        rows.append([DISPLAY_NAMES[d], len(groups), int((~noise).sum()),
                     int((~noise & (y == 1)).sum()), int(noise.sum()),
                     int((noise & (y == 1)).sum())])
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=7, mfc=palette[0], mec=palette[0],
                   label="punkt w grupie (kolor = grupa)"),
        plt.Line2D([], [], ls="", marker="s", ms=6, mfc="#555555", mec="#555555",
                   label="szum (poza grupami)"),
        plt.Line2D([], [], ls="", marker="o", ms=10, mfc="none", mec=FN_COLOR, mew=1.2,
                   label="prawdziwa anomalia"),
    ]
    cax.legend(handles=handles, loc="center", fontsize=10.5)
    save(fig, OUT / "figures" / "dbscan_clusters")
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
             " & & \\multicolumn{2}{c}{W grupach} & \\multicolumn{2}{c}{W szumie} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
             "Zbiór danych & Grupy & punkty & anomalie & punkty & anomalie \\\\",
             "\\midrule"]
    lines += [" & ".join(map(str, r)) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tables" / "dbscan_clusters.tex").write_text("\n".join(lines) + "\n")


def grid_around(*Zs, n=300):
    Z = np.vstack(Zs)
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    pad = 0.06 * (hi - lo)
    xx, yy = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], n),
                         np.linspace(lo[1] - pad[1], hi[1] + pad[1], n))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def draw_region(ax, xx, yy, score, threshold):
    """Shade {score < threshold} (the region accepted as normal) and its border."""
    f = threshold - score.reshape(xx.shape)
    if f.max() > 0:
        ax.contourf(xx, yy, f, levels=[0, f.max()], colors=[REGION])
        ax.contour(xx, yy, f, levels=[0], colors="black", linewidths=1)


def ocsvm_region(data, best, preds):
    """Every dataset: OCSVM with the validated setting fitted on two PCA components."""
    fig, axes, cax = panels()
    for ax, d in zip(axes, TASK1_DATASETS):
        exp, pca = data[d]["exp"], data[d]["pca"]
        Z_tr, Z_va, Z_te = (pca.transform(a) for a in (exp.X_train, exp.X_val, exp.X_test))
        params = json.loads(best[(best.dataset == d) & (best.method == "OCSVM")].params.iloc[0])
        det = build("OCSVM", **params).fit(Z_tr)
        threshold = best_f1_threshold(exp.y_val, det.score(Z_va))
        xx, yy, G = grid_around(Z_te)
        draw_region(ax, xx, yy, det.score(G), threshold)
        s_te = det.score(Z_te)
        maps.outcome_points(ax, Z_te, exp.y_test, (s_te >= threshold).astype(int))
        ax.set_xlim(xx.min(), xx.max()), ax.set_ylim(yy.min(), yy.max())
        pr2 = average_precision_score(exp.y_test, s_te)
        full = preds[(preds.dataset == d) & (preds.method == "OCSVM")]
        pr_full = average_precision_score(full.y, full.score)
        ax.set_title(f"{DISPLAY_NAMES[d]}, jądro {KERNEL_LABEL[params['kernel']]}\n"
                     f"PR-AUC {num(pr2, 2)} (pełny: {num(pr_full, 2)})",
                     fontsize=9.5)
    region = plt.Rectangle((0, 0), 1, 1, fc=REGION, ec="black", lw=1,
                           label="obszar uznany za normalny")
    legend_cell(cax, axes[0], extra=[region])
    save(fig, OUT / "figures" / "ocsvm_region")


def ocsvm_region_params(data):
    """Cardio, RBF kernel: how gamma and nu shape the region (built-in threshold)."""
    exp, pca = data["cardio"]["exp"], data["cardio"]["pca"]
    Z_tr = pca.transform(exp.X_train)
    y = exp.y_train
    xx, yy, G = grid_around(Z_tr)
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 5.0), sharex=True, sharey=True)
    for i, nu in enumerate((0.05, 0.5)):
        for j, gamma_factor in enumerate((0.1, 1.0, 10.0)):
            ax = axes[i, j]
            det = build("OCSVM", kernel="rbf", nu=nu, gamma_factor=gamma_factor).fit(Z_tr)
            draw_region(ax, xx, yy, det.score(G), 0.0)
            ax.scatter(*Z_tr[y == 0].T, s=6, c="#8f8f8f", lw=0, label="normalne")
            ax.scatter(*Z_tr[y == 1].T, s=26, c=FN_COLOR, marker="D", ec="black", lw=0.4,
                       label="anomalie")
            outside = np.mean(det.train_scores_ > 0)
            ax.text(0.03, 0.97, f"poza obszarem: {num(100 * outside, 0)} %",
                    transform=ax.transAxes, va="top", fontsize=9.5,
                    bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.5})
            if i == 0:
                ax.set_title(f"$\\gamma$ = {GAMMA_LABEL[gamma_factor]}", fontsize=10.5)
            if j == 0:
                ax.set_ylabel(f"$\\nu$ = {num(nu, 2)}", fontsize=11)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_xlim(xx.min(), xx.max()), ax.set_ylim(yy.min(), yy.max())
    region = plt.Rectangle((0, 0), 1, 1, fc=REGION, ec="black", lw=1,
                           label="obszar uznany za normalny")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend([region] + handles, [region.get_label()] + labels, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.04), markerscale=1.3, fontsize=10.5)
    fig.subplots_adjust(hspace=0.35, wspace=0.06)
    save(fig, OUT / "figures" / "ocsvm_region_params")


def main():
    setup()
    preds = pd.read_csv(OUT / "test_predictions.csv.gz")
    preds = preds[preds.seed == SEED]
    best = pd.read_csv(OUT / "best_configs.csv")
    data = {}
    for d in TASK1_DATASETS:
        exp = make_experiment(d, SEED)
        pca = maps.projection(exp)
        data[d] = {"exp": exp, "pca": pca, "Z_test": pca.transform(exp.X_test)}

    all_counts = {m: fig_detected(data, preds, m) for m in METHODS}
    table_counts(all_counts, data)
    print(pd.DataFrame(all_counts))
    dbscan_clusters(data, best)
    ocsvm_region(data, best, preds)
    ocsvm_region_params(data)


if __name__ == "__main__":
    main()
