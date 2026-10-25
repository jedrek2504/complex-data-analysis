"""Task 1: what each detector found on Cardio, shown in a 2-D PCA view.

Split 0 of Cardio is used. The PCA projection is fitted on the (standardised)
training part and applied to the test part. For every detector the
configuration and the threshold chosen on the validation part are taken from
run_best (results/task1/test_predictions.csv.gz); the test part is only drawn.

A second figure shows DBSCAN as a clustering method: test points are assigned
to the cluster of their nearest core point when it is closer than eps and
to noise otherwise (the detector's own rule).

Outputs (results/task1/):
  figures/detected_<method>.pdf, figures/dbscan_clusters.pdf,
  tables/detected_counts.tex (LaTeX macros with TP/FN/FP),
  tables/dbscan_clusters.tex
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common.detectors import build
from common.protocol import make_experiment
from task1_classical.config import METHODS
from task1_classical.run_grid import OUT
from task1_classical.style import FN_COLOR, FP_COLOR, NORMAL, TP_COLOR, WIDTH, save, setup

DATASET, SEED = "cardio", 0


def projection(exp):
    pca = PCA(2, random_state=0).fit(exp.X_train)
    return pca.transform(exp.X_test)


def counts(y, pred):
    return {"TP": int(((pred == 1) & (y == 1)).sum()), "FN": int(((pred == 0) & (y == 1)).sum()),
            "FP": int(((pred == 1) & (y == 0)).sum())}


def fig_detected(Z, y, pred, method):
    c = counts(y, pred)
    tn = (pred == 0) & (y == 0)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    ax.scatter(*Z[tn].T, s=14, c=NORMAL, lw=0, label=f"normalne, nieoznaczone ({int(tn.sum())})")
    ax.scatter(*Z[(pred == 1) & (y == 0)].T, s=48, c=FP_COLOR, marker="^", ec="black", lw=0.5,
               label=f"fałszywy alarm, FP ({c['FP']})")
    ax.scatter(*Z[(pred == 1) & (y == 1)].T, s=48, c=TP_COLOR, marker="o", ec="black", lw=0.5,
               label=f"wykryta anomalia, TP ({c['TP']})")
    ax.scatter(*Z[(pred == 0) & (y == 1)].T, s=70, c=FN_COLOR, marker="X", ec="black", lw=0.5,
               label=f"przeoczona anomalia, FN ({c['FN']})")
    ax.set_xlabel("1. składowa PCA")
    ax.set_ylabel("2. składowa PCA")
    ax.set_xticks([]), ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, markerscale=1.2)
    save(fig, OUT / "figures" / f"detected_{method.lower()}")
    return c


def dbscan_clusters(exp, Z, params):
    det = build("DBSCAN", **params).fit(exp.X_train)
    labels = det.model.labels_[det.model.core_sample_indices_]
    dist, idx = det.core_nn_.kneighbors(exp.X_test)
    cluster = np.where(dist[:, 0] <= det.eps, labels[idx[:, 0]], -1)
    y = exp.y_test

    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    palette = ["#56B4E9", "#009E73", "#CC79A7", "#F0E442", "#0072B2"]
    for k in sorted(set(cluster) - {-1}):
        m = cluster == k
        ax.scatter(*Z[m].T, s=16, c=palette[k % len(palette)], lw=0,
                   label=f"grupa {k + 1} ({int(m.sum())} punktów)")
    noise = cluster == -1
    ax.scatter(*Z[noise].T, s=26, c="#555555", marker="s", lw=0,
               label=f"szum ({int(noise.sum())} punktów)")
    ax.scatter(*Z[y == 1].T, s=90, facecolors="none", edgecolors=FN_COLOR, lw=1.4,
               label=f"prawdziwe anomalie ({int(y.sum())})")
    ax.set_xlabel("1. składowa PCA")
    ax.set_ylabel("2. składowa PCA")
    ax.set_xticks([]), ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    save(fig, OUT / "figures" / "dbscan_clusters")

    rows = [{"Grupa": f"grupa {k + 1}", "Punkty testowe": int((cluster == k).sum()),
             "w tym anomalie": int(((cluster == k) & (y == 1)).sum())}
            for k in sorted(set(cluster) - {-1})]
    rows.append({"Grupa": "szum (poza grupami)", "Punkty testowe": int(noise.sum()),
                 "w tym anomalie": int((noise & (y == 1)).sum())})
    table = pd.DataFrame(rows)
    lines = ["\\begin{tabular}{lrr}", "\\toprule",
             "Grupa & Punkty testowe & w tym prawdziwe anomalie \\\\", "\\midrule"]
    lines += [f"{r.Grupa} & {r['Punkty testowe']} & {r['w tym anomalie']} \\\\"
              for _, r in table.iterrows()]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tables" / "dbscan_clusters.tex").write_text("\n".join(lines) + "\n")
    return table


def main():
    setup()
    exp = make_experiment(DATASET, SEED)
    Z = projection(exp)
    preds = pd.read_csv(OUT / "test_predictions.csv.gz")
    preds = preds[(preds.dataset == DATASET) & (preds.seed == SEED)]
    macros = []
    for method in METHODS:
        p = preds[preds.method == method].set_index("row").loc[exp.idx_test]
        assert np.array_equal(p.y.values, exp.y_test)
        c = fig_detected(Z, exp.y_test, p.pred.values, method)
        macros.append(f"\\newcommand{{\\det{method}}}{{TP = {c['TP']}, FN = {c['FN']}, "
                      f"FP = {c['FP']}}}")
        print(method, c)
    (OUT / "tables" / "detected_counts.tex").write_text("\n".join(macros) + "\n")

    best = pd.read_csv(OUT / "best_configs.csv")
    params = json.loads(best[(best.dataset == DATASET) & (best.method == "DBSCAN")].params.iloc[0])
    print(dbscan_clusters(exp, Z, params))


if __name__ == "__main__":
    main()
