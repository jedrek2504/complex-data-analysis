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

from common.agreement import jaccard_matrix
from common.datasets import DISPLAY_NAMES, SPECS, TASK1_DATASETS
from common.latex import num, pm, tabular
from task1_classical.config import METHODS
from task1_classical.run_grid import OUT
from task1_classical.style import (DATASET_STYLE, METHOD_STYLE, NAMES, WIDTH, line, save,
                                   setup)

FIG, TAB = OUT / "figures", OUT / "tables"
CARDIO = "cardio"


def describe_params(method, params):
    p = json.loads(params) if isinstance(params, str) else params
    if method == "IForest":
        return f"$T$={p['n_estimators']}"
    if method in ("LOF", "COF"):
        return f"$k$={p['n_neighbors']}"
    if method == "OCSVM":
        g = "" if p["kernel"] == "linear" else f", $\\gamma$={num(p['gamma_factor'], 1)}/$d$"
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
    agg = best.groupby(["dataset", "method"]).agg(
        precision=("precision", "mean"), recall=("recall", "mean"),
        f1=("f1", "mean"), f1_sd=("f1", "std"),
        roc=("roc_auc", "mean"), pr=("pr_auc", "mean"), pr_sd=("pr_auc", "std"),
        time=("fit_time", "mean"), stime=("score_time", "mean"),
        params=("params", "first")).reset_index()
    lines = ["\\begin{tabular}{llL{3.2cm}rrrrrr}", "\\toprule",
             " & ".join(["Zbiór danych", "Metoda", "\\makecell[l]{Ustawienie\\\\(z walidacji)}",
                         "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC",
                         "\\makecell[r]{Czas\\\\{}[ms]}"]) + " \\\\",
             "\\midrule"]
    for j, d in enumerate(TASK1_DATASETS):
        sub = agg[agg.dataset == d].set_index("method").loc[list(METHODS)]
        best_pr, best_f1 = sub.pr.max(), sub.f1.max()
        for i, (m, r) in enumerate(sub.iterrows()):
            f1, pr = pm(r.f1, r.f1_sd), pm(r.pr, r.pr_sd)
            f1 = f"\\textbf{{{f1}}}" if np.isclose(r.f1, best_f1) else f1
            pr = f"\\textbf{{{pr}}}" if np.isclose(r.pr, best_pr) else pr
            lines.append(" & ".join([
                DISPLAY_NAMES[d] if i == 0 else "", m, describe_params(m, r.params),
                num(r.precision), num(r.recall), f1, num(r.roc), pr,
                num(1000 * (r.time + r.stime), 1)]) + " \\\\")
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
        rows.append({
            "Zbiór danych": DISPLAY_NAMES[d],
            "\\makecell[r]{Udział anomalii\\\\w teście}": f"{num(100 * share, 1)}\\,\\%",
            "\\makecell[r]{Accuracy\\\\,,wszystko normalne''}": num(1 - share),
            "\\makecell[l]{Najlepszy\\\\detektor (F1)}": top,
            "\\makecell[r]{Jego\\\\Accuracy}": num(sub.accuracy[top]),
            "\\makecell[r]{Jego\\\\F1}": num(sub.f1[top]),
        })
    tabular(pd.DataFrame(rows), TAB / "accuracy_trap.tex", align="lrrlrr")


def table_kinds(preds):
    sub = preds[preds.y == 1]
    rec = sub.groupby(["dataset", "kind", "method"]).pred.mean().unstack()[list(METHODS)]
    counts = sub[sub.method == "IForest"].groupby(["dataset", "kind"]).size()
    rows = []
    for d in TASK1_DATASETS:
        for i, k in enumerate(("global", "local")):
            r = rec.loc[(d, k)]
            cells = {m: (f"\\textbf{{{num(100 * v, 0)}}}" if np.isclose(v, r.max())
                         else num(100 * v, 0)) for m, v in r.items()}
            rows.append({"Zbiór danych": DISPLAY_NAMES[d] if i == 0 else "",
                         "Typ": {"global": "globalne", "local": "lokalne"}[k],
                         "Liczba": str(counts[(d, k)]), **cells})
    tabular(pd.DataFrame(rows), TAB / "recall_by_kind.tex", align="llr" + "r" * 5)
    return rec


def table_param_effect(grid, method):
    """Setting chosen on validation and the spread of test PR-AUC over the grid."""
    sl = score_level(grid)
    rows = []
    for d in TASK1_DATASETS:
        s = sl[(sl.dataset == d) & (sl.method == method)]
        means = s.groupby("params")[["val_pr_auc", "test_pr_auc"]].mean()
        chosen = means.val_pr_auc.idxmax()
        rows.append({
            "Zbiór danych": DISPLAY_NAMES[d],
            "\\makecell[l]{Ustawienie wybrane\\\\na walidacji}": describe_params(method, chosen),
            "\\makecell[r]{PR-AUC przy\\\\tym ustawieniu}": num(means.test_pr_auc[chosen]),
            "\\makecell[r]{PR-AUC: najsłabsze --\\\\najlepsze ustawienie}":
                f"{num(means.test_pr_auc.min())} -- {num(means.test_pr_auc.max())}",
        })
    tabular(pd.DataFrame(rows), TAB / f"param_{method}.tex", align="llrr")


def table_contamination(grid):
    """For IForest/LOF/COF: contamination chosen by validation F1."""
    sub = grid[grid.method.isin(["IForest", "LOF", "COF"])]
    configs = best_by_validation(score_level(sub), ["dataset", "method"])
    sub = sub.merge(configs[["dataset", "method", "params"]])
    rows = []
    for d in TASK1_DATASETS:
        cells = {"Zbiór danych": DISPLAY_NAMES[d],
                 "Prawdziwy udział": f"{num(100 * SPECS[d].anomalies / SPECS[d].n, 1)}\\,\\%"}
        for m in ("IForest", "LOF", "COF"):
            g = sub[(sub.dataset == d) & (sub.method == m)].groupby("contamination")
            c = g.val_f1.mean().idxmax()
            cells[m] = f"$c$={num(c, 2)} (F1 {num(g.test_f1.mean()[c], 2)})"
        rows.append(cells)
    tabular(pd.DataFrame(rows), TAB / "contamination.tex", align="lrlll")


def tables_full_grid(grid):
    """Appendix: mean test results of every configuration (all experiments)."""
    agg = (grid.groupby(["method", "params", "contamination", "dataset"], dropna=False)
           [["test_pr_auc", "test_f1"]].mean().reset_index())
    for method in METHODS:
        sub = agg[agg.method == method]
        keys = sub[["params", "contamination"]].drop_duplicates()
        lines = ["\\begin{longtable}{l" + "rr" * len(TASK1_DATASETS) + "}", "\\toprule",
                 "Ustawienie & " + " & ".join(
                     f"\\multicolumn{{2}}{{c}}{{{n}}}" for n in NAMES) + " \\\\",
                 " & " + " & ".join(["PR-AUC & F1"] * len(NAMES)) + " \\\\",
                 "\\midrule", "\\endhead"]
        for _, k in keys.iterrows():
            label = describe_params(method, k.params)
            if not pd.isna(k.contamination):
                label += f", $c$={num(k.contamination, 2)}"
            cells = []
            for d in TASK1_DATASETS:
                r = sub[(sub.params == k.params) & (sub.dataset == d)
                        & ((sub.contamination == k.contamination)
                           | (pd.isna(sub.contamination) & pd.isna(k.contamination)))].iloc[0]
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
    """Cardio: precision and recall of the contamination threshold."""
    sub = grid[grid.method.isin(["IForest", "LOF", "COF"]) & (grid.dataset == CARDIO)]
    configs = best_by_validation(score_level(sub), ["method"])
    sub = sub.merge(configs[["method", "params"]])
    true_share = SPECS[CARDIO].anomalies / SPECS[CARDIO].n
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True)
    for m in ("IForest", "LOF", "COF"):
        g = sub[sub.method == m].groupby("contamination")
        for ax, col in zip(axes, ("test_precision", "test_recall")):
            line(ax, g[col].mean().index, g[col].mean().values, METHOD_STYLE[m], m)
    for ax, title in zip(axes, ("Precision (trafność zgłoszeń)", "Recall (wykryte anomalie)")):
        ax.axvline(true_share, color="#555555", lw=1.2, ls="--")
        ax.set_title(title)
        ax.set_xlabel("contamination $c$")
        ax.set_ylim(0, 1.02)
    axes[0].text(true_share + 0.007, 0.66, "prawdziwy\nudział", fontsize=9.5, color="#555555")
    axes[0].legend(loc="upper right")
    save(fig, FIG / "contamination_cardio")


KERNELS = ("rbf", "poly", "sigmoid", "linear")
KERNEL_STYLE = {"rbf": ("#0072B2", ""), "poly": ("#E69F00", "//"),
                "sigmoid": ("#009E73", "\\\\"), "linear": ("#CC79A7", "xx")}
KERNEL_LABEL = {"rbf": "RBF", "poly": "wielomianowe", "sigmoid": "sigmoidalne",
                "linear": "liniowe"}


def fig_ocsvm(grid):
    sub = grid[grid.method == "OCSVM"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDTH, 5.6))
    width = 0.2
    for i, k in enumerate(KERNELS):
        chosen = best_by_validation(sub[sub.kernel == k], ["dataset"]).set_index("dataset")
        vals = chosen.loc[list(TASK1_DATASETS), "test_pr_auc"]
        color, hatch = KERNEL_STYLE[k]
        a1.bar(np.arange(5) + (i - 1.5) * width, vals, width * 0.92, color=color,
               hatch=hatch, edgecolor="white", lw=0.5, label=KERNEL_LABEL[k])
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
        ax.bar(np.arange(5) + (i - 2) * width, g["mean"], width * 0.92, yerr=g["std"],
               color=st["color"], hatch=st["hatch"], edgecolor="white", lw=0.5, label=m,
               error_kw={"elinewidth": 0.9, "ecolor": "#333333", "capsize": 2})
    for i, d in enumerate(TASK1_DATASETS):
        share = SPECS[d].anomalies / SPECS[d].n
        ax.plot([i - 0.45, i + 0.45], [share, share], color="black", lw=1.2, ls=":")
    ax.set_xticks(range(5), NAMES)
    ax.set_ylabel("PR-AUC (test)")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save(fig, FIG / "best_comparison")


def fig_kinds(rec):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True)
    width = 0.16
    for ax, kind, title in ((axes[0], "global", "anomalie globalne"),
                            (axes[1], "local", "anomalie lokalne")):
        for i, m in enumerate(METHODS):
            st = METHOD_STYLE[m]
            vals = [rec.loc[(d, kind), m] for d in TASK1_DATASETS]
            ax.bar(np.arange(5) + (i - 2) * width, vals, width * 0.92, color=st["color"],
                   hatch=st["hatch"], edgecolor="white", lw=0.5, label=m)
        ax.set_xticks(range(5), NAMES, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Recall (test)")
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(1.05, 1.25))
    save(fig, FIG / "recall_by_kind")


def fig_jaccard(preds):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.1), constrained_layout=True)
    for ax, d in zip(axes, ("breastw", CARDIO)):
        mats = []
        for _, g in preds[preds.dataset == d].groupby("seed"):
            P = g.pivot(index="row", columns="method", values="pred")[list(METHODS)]
            mats.append(jaccard_matrix(P.values.T))
        M = np.nanmean(mats, axis=0)
        im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, num(v, 2), ha="center", va="center", fontsize=10,
                    color="white" if v > 0.55 else "black")
        ax.set_xticks(range(5), METHODS, rotation=40, ha="right")
        ax.set_yticks(range(5), METHODS if d == "breastw" else [""] * 5)
        ax.set_title(DISPLAY_NAMES[d])
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.9)
    cbar.set_label("indeks Jaccarda (1 = te same punkty)")
    save(fig, FIG / "jaccard_heatmap")


def fig_time(grid):
    sl = score_level(grid)
    agg = sl.groupby(["dataset", "method", "params"]).agg(
        t=("fit_time", "mean"), s=("score_time", "mean"), pr=("test_pr_auc", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.1), sharey=True)
    for ax, d in zip(axes, ("wine", "mammography")):
        for m in METHODS:
            s = agg[(agg.dataset == d) & (agg.method == m)]
            st = METHOD_STYLE[m]
            ax.scatter(1000 * (s.t + s.s), s.pr, s=30, color=st["color"], marker=st["marker"],
                       edgecolor="black", lw=0.3, label=m)
        ax.set_xscale("log")
        ax.set_xlabel("czas uczenia i oceny [ms]")
        n_train = SPECS[d].n // 2
        ax.set_title(f"{DISPLAY_NAMES[d]}: {n_train} punktów uczących")
    axes[0].set_ylabel("PR-AUC (test)")
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(1.05, -0.2), markerscale=1.3)
    save(fig, FIG / "time_vs_quality")


def main():
    setup()
    grid = pd.read_csv(OUT / "grid.csv")
    best = pd.read_csv(OUT / "best_metrics.csv")
    preds = pd.read_csv(OUT / "test_predictions.csv.gz")

    agg = table_best(best)
    agg.to_csv(OUT / "best_summary.csv", index=False)
    table_accuracy_trap(best, preds)
    rec = table_kinds(preds)
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
    fig_kinds(rec)
    fig_jaccard(preds)
    fig_time(grid)


if __name__ == "__main__":
    main()
