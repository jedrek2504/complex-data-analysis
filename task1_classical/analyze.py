"""Task 1: tables and figures for the report.

Reads results/task1/{grid.csv, best_metrics.csv, best_configs.csv,
test_predictions.csv.gz} and writes LaTeX tables to results/task1/tables and
figures to results/task1/figures.

Whenever a figure varies one parameter, the remaining parameters are chosen
for every value of that parameter on the validation part; the test part is
only plotted.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.agreement import jaccard_matrix
from common.datasets import DISPLAY_NAMES, SPECS, TASK1_DATASETS
from common.latex import num, pm, tabular
from task1_classical.config import METHODS
from task1_classical.run_grid import OUT
from task1_classical.style import (
    DATASET_STYLE,
    METHOD_STYLE,
    NAMES,
    WIDTH,
    line,
    save,
    setup,
)

FIG, TAB = OUT / "figures", OUT / "tables"
CONT_METHODS = ("IForest", "LOF", "COF")
GLOBAL_FAMILY = ("IForest", "OCSVM", "DBSCAN")
LOCAL_FAMILY = ("LOF", "COF")


def describe_params(method, params):
    p = json.loads(params) if isinstance(params, str) else params
    if method == "IForest":
        return f"$T$={p['n_estimators']}"
    if method in ("LOF", "COF"):
        return f"$k$={p['n_neighbors']}"
    if method == "OCSVM":
        g = (
            ""
            if p["kernel"] == "linear"
            else f", $\\gamma$={num(p['gamma_factor'], 1)}/$d$"
        )
        return f"{p['kernel']}, $\\nu$={num(p['nu'], 2)}{g}"
    return f"minPts={p['min_samples']}, $q_\\varepsilon$={num(p['eps_quantile'], 2)}"


def score_level(grid):
    """One row per fitted model (contamination only moves the threshold)."""
    return grid.drop_duplicates(["dataset", "seed", "method", "params"])


def best_by_validation(df, keys, metric="val_pr_auc"):
    """Per group of ``keys``: the configuration with the best mean validation
    metric, returned with the mean of all test columns."""
    means = df.groupby(keys + ["params"]).mean(numeric_only=True).reset_index()
    return means.loc[means.groupby(keys)[metric].idxmax()].reset_index(drop=True)


# ---------------------------------------------------------------- tables


def table_best(best):
    agg = (
        best.groupby(["dataset", "method"])
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            f1_sd=("f1", "std"),
            roc=("roc_auc", "mean"),
            pr=("pr_auc", "mean"),
            pr_sd=("pr_auc", "std"),
            time=("fit_time", "mean"),
            stime=("score_time", "mean"),
            params=("params", "first"),
        )
        .reset_index()
    )
    lines = [
        "\\begin{tabular}{llL{3.2cm}rrrrrr}",
        "\\toprule",
        " & ".join(
            [
                "Zbiór danych",
                "Metoda",
                "\\makecell[l]{Ustawienie\\\\(z walidacji)}",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
                "PR-AUC",
                "\\makecell[r]{Czas\\\\{}[ms]}",
            ]
        )
        + " \\\\",
        "\\midrule",
    ]
    for j, d in enumerate(TASK1_DATASETS):
        sub = agg[agg.dataset == d].set_index("method").loc[list(METHODS)]
        best_pr, best_f1 = sub.pr.max(), sub.f1.max()
        for i, (m, r) in enumerate(sub.iterrows()):
            f1, pr = pm(r.f1, r.f1_sd), pm(r.pr, r.pr_sd)
            f1 = f"\\textbf{{{f1}}}" if np.isclose(r.f1, best_f1) else f1
            pr = f"\\textbf{{{pr}}}" if np.isclose(r.pr, best_pr) else pr
            lines.append(
                " & ".join(
                    [
                        DISPLAY_NAMES[d] if i == 0 else "",
                        m,
                        describe_params(m, r.params),
                        num(r.precision),
                        num(r.recall),
                        f1,
                        num(r.roc),
                        pr,
                        num(1000 * (r.time + r.stime), 1),
                    ]
                )
                + " \\\\"
            )
        if j < len(TASK1_DATASETS) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "best_results.tex").write_text("\n".join(lines) + "\n")
    return agg


def table_accuracy_trap(best, preds):
    rows = []
    for d in TASK1_DATASETS:
        share = preds[(preds.dataset == d) & (preds.method == "IForest")].y.mean()
        sub = best[best.dataset == d].groupby("method")[["accuracy", "f1"]].mean()
        top = sub.f1.idxmax()
        rows.append(
            {
                "Zbiór danych": DISPLAY_NAMES[d],
                "\\makecell[r]{Udział anomalii\\\\w teście}": f"{num(100 * share, 1)}\\,\\%",
                "\\makecell[r]{Accuracy\\\\,,wszystko normalne''}": num(1 - share),
                "\\makecell[l]{Najlepszy\\\\detektor (F1)}": top,
                "\\makecell[r]{Jego\\\\Accuracy}": num(sub.accuracy[top]),
                "\\makecell[r]{Jego\\\\F1}": num(sub.f1[top]),
            }
        )
    tabular(pd.DataFrame(rows), TAB / "accuracy_trap.tex", align="lrrlrr")


def table_kinds(preds):
    """Recall of global and local anomalies at the validation threshold, with
    the share of flagged test points (a higher share buys a higher recall)."""
    sub = preds[preds.y == 1]
    rec = (
        sub.groupby(["dataset", "kind", "method"]).pred.mean().unstack()[list(METHODS)]
    )
    flagged = preds.groupby(["dataset", "method"]).pred.mean().unstack()[list(METHODS)]
    counts = sub[sub.method == "IForest"].groupby(["dataset", "kind"]).size()
    rows = []
    for d in TASK1_DATASETS:
        for i, k in enumerate(("global", "local")):
            r = rec.loc[(d, k)]
            rows.append(
                {
                    "Zbiór danych": DISPLAY_NAMES[d] if i == 0 else "",
                    "Miara [\\%]": {"global": "Recall: globalne", "local": "Recall: lokalne"}[k],
                    "Liczba": str(counts[(d, k)]),
                    **{m: num(100 * v, 0) for m, v in r.items()},
                }
            )
        rows.append(
            {
                "Zbiór danych": "",
                "Miara [\\%]": "zgłoszone punkty",
                "Liczba": "--",
                **{m: num(100 * v, 0) for m, v in flagged.loc[d].items()},
            }
        )
    tabular(pd.DataFrame(rows), TAB / "recall_by_kind.tex", align="llr" + "r" * 5)
    return rec


def auc_by_kind(preds):
    """Threshold-free view: ROC-AUC of the anomalies of one type against all
    normal test points (mean over the splits that contain that type)."""
    rows = []
    for (d, m, seed), g in preds.groupby(["dataset", "method", "seed"]):
        normal = g[g.y == 0].score.values
        for k in ("global", "local"):
            a = g[g.kind == k].score.values
            if len(a):
                y = np.r_[np.zeros(len(normal)), np.ones(len(a))]
                rows.append({"dataset": d, "method": m, "kind": k,
                             "auc": roc_auc_score(y, np.r_[normal, a])})
    auc = (pd.DataFrame(rows).groupby(["dataset", "kind", "method"]).auc.mean()
           .unstack()[list(METHODS)])
    out = []
    for d in TASK1_DATASETS:
        for i, k in enumerate(("global", "local")):
            r = auc.loc[(d, k)]
            out.append(
                {
                    "Zbiór danych": DISPLAY_NAMES[d] if i == 0 else "",
                    "Typ": {"global": "globalne", "local": "lokalne"}[k],
                    **{
                        m: f"\\textbf{{{num(v, 2)}}}" if np.isclose(v, r.max()) else num(v, 2)
                        for m, v in r.items()
                    },
                }
            )
    tabular(pd.DataFrame(out), TAB / "auc_by_kind.tex", align="ll" + "r" * 5)
    return auc


def table_param_effect(grid, method):
    """Setting chosen on validation and the spread of test PR-AUC over the grid."""
    sl = score_level(grid)
    rows = []
    for d in TASK1_DATASETS:
        s = sl[(sl.dataset == d) & (sl.method == method)]
        means = s.groupby("params")[["val_pr_auc", "test_pr_auc"]].mean()
        chosen = means.val_pr_auc.idxmax()
        rows.append(
            {
                "Zbiór danych": DISPLAY_NAMES[d],
                "\\makecell[l]{Ustawienie wybrane\\\\na walidacji}": describe_params(
                    method, chosen
                ),
                "\\makecell[r]{PR-AUC przy\\\\tym ustawieniu}": num(
                    means.test_pr_auc[chosen]
                ),
                "\\makecell[r]{PR-AUC: najsłabsze --\\\\najlepsze ustawienie}": f"{num(means.test_pr_auc.min())} -- {num(means.test_pr_auc.max())}",
            }
        )
    tabular(pd.DataFrame(rows), TAB / f"param_{method}.tex", align="llrr")


def table_contamination(grid):
    """For IForest/LOF/COF: contamination chosen by validation F1."""
    sub = grid[grid.method.isin(["IForest", "LOF", "COF"])]
    configs = best_by_validation(score_level(sub), ["dataset", "method"])
    sub = sub.merge(configs[["dataset", "method", "params"]])
    rows = []
    for d in TASK1_DATASETS:
        cells = {
            "Zbiór danych": DISPLAY_NAMES[d],
            "Prawdziwy udział": f"{num(100 * SPECS[d].anomalies / SPECS[d].n, 1)}\\,\\%",
        }
        for m in ("IForest", "LOF", "COF"):
            g = sub[(sub.dataset == d) & (sub.method == m)].groupby("contamination")
            c = g.val_f1.mean().idxmax()
            cells[m] = f"$c$={num(c, 2)} (F1 {num(g.test_f1.mean()[c], 2)})"
        rows.append(cells)
    tabular(pd.DataFrame(rows), TAB / "contamination.tex", align="lrlll")


def tables_full_grid(grid):
    """Appendix: mean test results of every configuration (all experiments)."""
    agg = (
        grid.groupby(["method", "params", "contamination", "dataset"], dropna=False)[
            ["test_pr_auc", "test_f1"]
        ]
        .mean()
        .reset_index()
    )
    for method in METHODS:
        sub = agg[agg.method == method]
        keys = sub[["params", "contamination"]].drop_duplicates()
        lines = [
            "\\begin{longtable}{l" + "rr" * len(TASK1_DATASETS) + "}",
            "\\toprule",
            "Ustawienie & "
            + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for n in NAMES)
            + " \\\\",
            " & " + " & ".join(["PR-AUC & F1"] * len(NAMES)) + " \\\\",
            "\\midrule",
            "\\endhead",
        ]
        for _, k in keys.iterrows():
            label = describe_params(method, k.params)
            if not pd.isna(k.contamination):
                label += f", $c$={num(k.contamination, 2)}"
            cells = []
            for d in TASK1_DATASETS:
                r = sub[
                    (sub.params == k.params)
                    & (sub.dataset == d)
                    & (
                        (sub.contamination == k.contamination)
                        | (pd.isna(sub.contamination) & pd.isna(k.contamination))
                    )
                ].iloc[0]
                cells += [num(r.test_pr_auc), num(r.test_f1)]
            lines.append(label + " & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{longtable}"]
        (TAB / f"grid_{method}.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- figures


def _dataset_curves(ax, sub, key, metric="test_pr_auc"):
    for d in TASK1_DATASETS:
        g = sub[sub.dataset == d].groupby(key)[metric].mean()
        line(ax, g.index, g.values, DATASET_STYLE[d], DISPLAY_NAMES[d])


def fig_iforest(grid):
    sub = score_level(grid[grid.method == "IForest"])
    fig, ax = plt.subplots(figsize=(WIDTH, 3.1))
    _dataset_curves(ax, sub, "n_estimators")
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 500], ["50", "100", "200", "500"])
    ax.set_xlabel("liczba drzew $T$")
    ax.set_ylabel("PR-AUC (test)")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    save(fig, FIG / "iforest_trees")


def fig_neighbors(grid, method):
    sub = score_level(grid[grid.method == method])
    fig, ax = plt.subplots(figsize=(WIDTH, 3.1))
    _dataset_curves(ax, sub, "n_neighbors")
    ax.set_xticks([5, 10, 20, 35, 50])
    ax.set_xlabel("liczba sąsiadów $k$")
    ax.set_ylabel("PR-AUC (test)")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    save(fig, FIG / f"{method.lower()}_neighbors")


def fig_contamination(grid):
    """Every dataset: precision and recall of the contamination threshold,
    averaged over IForest, LOF and COF (their settings chosen on validation)."""
    sub = grid[grid.method.isin(CONT_METHODS)]
    configs = best_by_validation(score_level(sub), ["dataset", "method"])
    sub = sub.merge(configs[["dataset", "method", "params"]])
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.1), sharey=True)
    for ax, col, title in zip(
        axes,
        ("test_precision", "test_recall"),
        ("Precision (trafność zgłoszeń)", "Recall (wykryte anomalie)"),
    ):
        _dataset_curves(ax, sub, "contamination", col)
        mean = sub.groupby(["dataset", "contamination"])[col].mean()
        mean = mean.groupby("contamination").mean()
        ax.plot(mean.index, mean.values, color="black", lw=3.2, label="średnio")
        ax.set_title(title)
        ax.set_xlabel("contamination $c$")
        ax.set_ylim(0, 1.02)
    axes[0].legend(ncol=6, fontsize=9.5, loc="upper center", bbox_to_anchor=(1.05, 1.3))
    save(fig, FIG / "contamination")


KERNELS = ("rbf", "poly", "sigmoid", "linear")
KERNEL_STYLE = {
    "rbf": ("#0072B2", ""),
    "poly": ("#E69F00", "//"),
    "sigmoid": ("#009E73", "\\\\"),
    "linear": ("#CC79A7", "xx"),
}
KERNEL_LABEL = {
    "rbf": "RBF",
    "poly": "wielomianowe",
    "sigmoid": "sigmoidalne",
    "linear": "liniowe",
}


def fig_ocsvm(grid):
    sub = grid[grid.method == "OCSVM"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDTH, 5.6))
    width = 0.2
    for i, k in enumerate(KERNELS):
        chosen = best_by_validation(sub[sub.kernel == k], ["dataset"]).set_index(
            "dataset"
        )
        vals = chosen.loc[list(TASK1_DATASETS), "test_pr_auc"]
        color, hatch = KERNEL_STYLE[k]
        a1.bar(
            np.arange(5) + (i - 1.5) * width,
            vals,
            width * 0.92,
            color=color,
            hatch=hatch,
            edgecolor="white",
            lw=0.5,
            label=KERNEL_LABEL[k],
        )
    a1.set_xticks(range(5), NAMES)
    a1.set_ylabel("PR-AUC (test)")
    a1.set_ylim(0, 1.02)
    a1.set_title("(a) rodzaj jądra ($\\nu$ i $\\gamma$ wybrane na walidacji)")
    a1.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    rbf = sub[sub.kernel == "rbf"]
    chosen = best_by_validation(rbf, ["dataset", "gamma_factor"])
    _dataset_curves(a2, chosen, "gamma_factor")
    a2.set_xscale("log")
    a2.set_xticks([0.1, 1, 10], ["0,1/$d$", "1/$d$", "10/$d$"])
    a2.set_xlabel("parametr $\\gamma$ (od szerokiego do wąskiego jądra)")
    a2.set_ylabel("PR-AUC (test)")
    a2.set_title("(b) jądro RBF: wpływ $\\gamma$ ($\\nu$ wybrane na walidacji)")
    a2.set_ylim(0, 1.02)
    a2.legend(ncol=5, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.subplots_adjust(hspace=0.75)
    save(fig, FIG / "ocsvm_params")


def fig_dbscan(grid):
    sub = grid[grid.method == "DBSCAN"]
    chosen = best_by_validation(sub, ["dataset", "eps_quantile"], metric="val_f1")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharex=True)
    _dataset_curves(axes[0], chosen, "eps_quantile", "test_f1")
    _dataset_curves(axes[1], chosen, "eps_quantile", "test_flagged")
    axes[0].set_title("F1 (test)")
    axes[1].set_title("odsetek punktów uznanych za szum")
    for ax in axes:
        ax.set_xlabel("kwantyl $q_\\varepsilon$ (wyznacza promień $\\varepsilon$)")
        ax.set_xticks([0.5, 0.75, 0.9, 0.99], ["0,5", "0,75", "0,9", "0,99"])
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(ncol=5, fontsize=9.5, loc="upper center", bbox_to_anchor=(1.1, -0.2))
    save(fig, FIG / "dbscan_params")


def fig_best(best):
    fig, ax = plt.subplots(figsize=(WIDTH, 3.1))
    width = 0.16
    for i, m in enumerate(METHODS):
        g = best[best.method == m].groupby("dataset").pr_auc.agg(["mean", "std"])
        g = g.loc[list(TASK1_DATASETS)]
        st = METHOD_STYLE[m]
        ax.bar(
            np.arange(5) + (i - 2) * width,
            g["mean"],
            width * 0.92,
            yerr=g["std"],
            color=st["color"],
            hatch=st["hatch"],
            edgecolor="white",
            lw=0.5,
            label=m,
            error_kw={"elinewidth": 0.9, "ecolor": "#333333", "capsize": 2},
        )
    for i, d in enumerate(TASK1_DATASETS):
        share = SPECS[d].anomalies / SPECS[d].n
        ax.plot([i - 0.45, i + 0.45], [share, share], color="black", lw=1.2, ls=":")
    ax.set_xticks(range(5), NAMES)
    ax.set_ylabel("PR-AUC (test)")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save(fig, FIG / "best_comparison")


def fig_kinds(auc):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True)
    width = 0.16
    for ax, kind, title in (
        (axes[0], "global", "anomalie globalne"),
        (axes[1], "local", "anomalie lokalne"),
    ):
        for i, m in enumerate(METHODS):
            st = METHOD_STYLE[m]
            vals = [auc.loc[(d, kind), m] for d in TASK1_DATASETS]
            ax.bar(
                np.arange(5) + (i - 2) * width,
                vals,
                width * 0.92,
                color=st["color"],
                hatch=st["hatch"],
                edgecolor="white",
                lw=0.5,
                label=m,
            )
        ax.axhline(0.5, color="black", lw=1.2, ls=":")
        ax.set_xticks(range(5), NAMES, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("ROC-AUC typu (test)")
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(1.05, 1.25))
    save(fig, FIG / "auc_by_kind")


def jaccard_by_dataset(preds):
    """Mean Jaccard matrix of the methods' test detections per dataset."""
    out = {}
    for d in TASK1_DATASETS:
        mats = []
        for _, g in preds[preds.dataset == d].groupby("seed"):
            P = g.pivot(index="row", columns="method", values="pred")[list(METHODS)]
            mats.append(jaccard_matrix(P.values.T))
        out[d] = np.nanmean(mats, axis=0)
    return out


def fig_jaccard(preds):
    mats = jaccard_by_dataset(preds)
    M = np.mean([mats[d] for d in TASK1_DATASETS], axis=0)
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(WIDTH, 3.4), gridspec_kw={"width_ratios": [1, 1.35]},
        constrained_layout=True,
    )
    a1.imshow(M, cmap="Greys", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(M):
        a1.text(j, i, num(v, 2), ha="center", va="center", fontsize=10,
                color="white" if v > 0.55 else "black")
    a1.set_xticks(range(5), METHODS, rotation=40, ha="right")
    a1.set_yticks(range(5), METHODS)
    a1.set_title("(a) średnio na 5 zbiorach danych")
    a1.grid(False)

    idx = {m: i for i, m in enumerate(METHODS)}
    pairs = {
        "IForest, OCSVM, DBSCAN\nmiędzy sobą": [
            (a, b) for i, a in enumerate(GLOBAL_FAMILY) for b in GLOBAL_FAMILY[i + 1:]],
        "LOF z COF": [("LOF", "COF")],
        "między rodzinami": [(a, b) for a in GLOBAL_FAMILY for b in LOCAL_FAMILY],
    }
    styles = (("#0072B2", ""), ("#D55E00", "//"), ("#999999", ".."))
    width = 0.26
    for k, ((label, pp), (color, hatch)) in enumerate(zip(pairs.items(), styles)):
        vals = [np.mean([mats[d][idx[a], idx[b]] for a, b in pp]) for d in TASK1_DATASETS]
        a2.bar(np.arange(5) + (k - 1) * width, vals, width * 0.92, color=color,
               hatch=hatch, edgecolor="white", lw=0.5, label=label)
    a2.set_xticks(range(5), NAMES, rotation=30, ha="right")
    a2.set_ylim(0, 1.0)
    a2.set_ylabel("indeks Jaccarda")
    a2.set_title("(b) zgodność w rodzinach i między nimi")
    a2.legend(fontsize=9, loc="upper right")
    save(fig, FIG / "jaccard_heatmap")


def time_summary(best):
    """Per dataset and method: mean test PR-AUC, its rank among the methods and
    the mean time of fitting plus scoring the test part [ms]."""
    b = best.assign(t=1000 * (best.fit_time + best.score_time))
    g = (
        b.groupby(["dataset", "method"])
        .agg(t=("t", "mean"), pr=("pr_auc", "mean"))
        .reset_index()
    )
    g["rank"] = g.groupby("dataset").pr.rank(ascending=False)
    return g


def table_time(tq):
    rows = []
    for m in METHODS:
        s = tq[tq.method == m]
        rows.append(
            {
                "Metoda": m,
                "\\makecell[r]{PR-AUC\\\\(średnio)}": num(s.pr.mean()),
                "\\makecell[r]{Miejsce\\\\(średnio)}": num(s["rank"].mean(), 1),
                "\\makecell[r]{Czas średnio\\\\{}[ms]}": num(s.t.mean(), 0),
                "\\makecell[r]{Czas: najkrótszy --\\\\najdłuższy [ms]}":
                    f"{num(s.t.min(), 0)} -- {num(s.t.max(), 0)}",
            }
        )
    tabular(pd.DataFrame(rows), TAB / "time_quality.tex", align="lrrrr")


def fig_time(tq):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 3.3))
    n_train = {d: SPECS[d].n // 2 for d in TASK1_DATASETS}
    order = sorted(TASK1_DATASETS, key=n_train.get)
    for m in METHODS:
        s = tq[tq.method == m].set_index("dataset")
        st = METHOD_STYLE[m]
        a1.plot([s.t.min(), s.t.max()], [s.pr.mean()] * 2, color=st["color"], lw=1.2,
                alpha=0.6)
        a1.scatter(s.t.mean(), s.pr.mean(), s=70, color=st["color"], marker=st["marker"],
                   edgecolor="black", lw=0.5, zorder=3)
        a1.annotate(m, (s.t.mean(), s.pr.mean()), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=10)
        line(a2, [n_train[d] for d in order], s.t.loc[order].values, st, m)
    a1.set_xscale("log")
    a1.set_xlabel("czas uczenia i oceny [ms]\n(punkt: średnio, kreska: od--do)")
    a1.set_ylabel("PR-AUC (test), średnio")
    a1.set_ylim(0.2, 0.75)
    a1.set_title("(a) jakość a czas, 5 zbiorów danych")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("liczba punktów uczących")
    a2.set_ylabel("czas uczenia i oceny [ms]")
    a2.set_title("(b) czas a wielkość zbioru danych")
    a2.legend(fontsize=9.5, ncol=2, loc="lower right")
    fig.subplots_adjust(wspace=0.35)
    save(fig, FIG / "time_vs_quality")


def main():
    setup()
    grid = pd.read_csv(OUT / "grid.csv")
    best = pd.read_csv(OUT / "best_metrics.csv")
    preds = pd.read_csv(OUT / "test_predictions.csv.gz")

    agg = table_best(best)
    agg.to_csv(OUT / "best_summary.csv", index=False)
    table_accuracy_trap(best, preds)
    table_kinds(preds)
    auc = auc_by_kind(preds)
    for method in METHODS:
        table_param_effect(grid, method)
    table_contamination(grid)
    tables_full_grid(grid)

    fig_iforest(grid)
    fig_neighbors(grid, "LOF")
    fig_neighbors(grid, "COF")
    fig_contamination(grid)
    fig_ocsvm(grid)
    fig_dbscan(grid)
    fig_best(best)
    fig_kinds(auc)
    fig_jaccard(preds)
    tq = time_summary(best)
    table_time(tq)
    fig_time(tq)


if __name__ == "__main__":
    main()
