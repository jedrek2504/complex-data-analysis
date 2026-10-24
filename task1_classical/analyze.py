"""Task 1: tables and figures for the report.

Reads results/task1/{grid.csv, best_metrics.csv, best_configs.csv,
test_predictions.csv.gz} and writes LaTeX tables to results/task1/tables and
figures to results/task1/figures.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.agreement import jaccard_matrix
from common.datasets import DISPLAY_NAMES, SPECS, TASK1_DATASETS
from common.latex import num, pm, tabular
from common.plotting import BLUES, METHOD_COLORS, TYPE_LABELS, save, setup
from task1_classical.config import CONTAMINATIONS, METHODS
from task1_classical.run_grid import OUT

FIG, TAB = OUT / "figures", OUT / "tables"
NAMES = [DISPLAY_NAMES[d] for d in TASK1_DATASETS]


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


# ---------------------------------------------------------------- tables

def table_best(best):
    agg = best.groupby(["dataset", "method"]).agg(
        precision=("precision", "mean"), recall=("recall", "mean"),
        f1=("f1", "mean"), f1_sd=("f1", "std"),
        roc=("roc_auc", "mean"), pr=("pr_auc", "mean"), pr_sd=("pr_auc", "std"),
        time=("fit_time", "mean"), stime=("score_time", "mean"),
        params=("params", "first")).reset_index()
    rows = []
    for d in TASK1_DATASETS:
        sub = agg[agg.dataset == d].set_index("method").loc[list(METHODS)]
        best_pr, best_f1 = sub.pr.max(), sub.f1.max()
        for i, (m, r) in enumerate(sub.iterrows()):
            bold = lambda v, b, s: f"\\textbf{{{s}}}" if np.isclose(v, b) else s  # noqa: E731
            rows.append({
                "Zbiór": DISPLAY_NAMES[d] if i == 0 else "",
                "Metoda": m,
                "Konfiguracja": describe_params(m, r.params),
                "Precision": num(r.precision), "Recall": num(r.recall),
                "F1": bold(r.f1, best_f1, pm(r.f1, r.f1_sd)),
                "ROC-AUC": num(r.roc),
                "PR-AUC": bold(r.pr, best_pr, pm(r.pr, r.pr_sd)),
                "Czas [ms]": num(1000 * (r.time + r.stime), 1),
            })
        rows.append(None)
    lines = ["\\begin{tabular}{llp{3.3cm}rrrrrr}", "\\toprule",
             " & ".join(["Zbiór", "Metoda", "Konfiguracja", "Prec.", "Rec.", "F1",
                         "ROC-AUC", "PR-AUC", "Czas [ms]"]) + " \\\\", "\\midrule"]
    for r in rows[:-1]:
        lines.append("\\midrule" if r is None else " & ".join(r.values()) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "best_results.tex").write_text("\n".join(lines) + "\n")
    return agg


def table_accuracy_trap(best, preds):
    rows = []
    for d in TASK1_DATASETS:
        y = preds[(preds.dataset == d) & (preds.method == "IForest")].y
        sub = best[best.dataset == d].groupby("method")[["accuracy", "f1", "pr_auc"]].mean()
        top = sub.f1.idxmax()
        worst = sub.f1.idxmin()
        rows.append({
            "Zbiór": DISPLAY_NAMES[d],
            "Udział anomalii": num(y.mean(), 3),
            "Acc. ``zawsze normalny''": num(1 - y.mean(), 3),
            "F1 / Recall": "0,000 / 0,000",
            "PR-AUC losowy": num(y.mean(), 3),
            f"Acc. najlepszy (F1)": f"{num(sub.accuracy[top], 3)} ({top})",
            "F1 najlepszy": num(sub.f1[top], 3),
            f"Acc. najsłabszy (F1)": f"{num(sub.accuracy[worst], 3)} ({worst})",
            "F1 najsłabszy": num(sub.f1[worst], 3),
        })
    tabular(pd.DataFrame(rows), TAB / "accuracy_trap.tex", align="lrrrrrrrr")


def table_kinds(preds):
    sub = preds[preds.y == 1]
    rec = sub.groupby(["dataset", "kind", "method"]).pred.mean().unstack()[list(METHODS)]
    counts = sub[sub.method == "IForest"].groupby(["dataset", "kind"]).size()
    rows = []
    for d in TASK1_DATASETS:
        kinds = [k for k in ("global", "local") if (d, k) in rec.index]
        for i, k in enumerate(kinds):
            r = rec.loc[(d, k)]
            best_v = r.max()
            cells = {m: (f"\\textbf{{{num(v)}}}" if np.isclose(v, best_v) else num(v))
                     for m, v in r.items()}
            rows.append({"Zbiór": DISPLAY_NAMES[d] if i == 0 else "",
                         "Typ": TYPE_LABELS[k],
                         "$n$ (suma pow.)": str(counts[(d, k)]), **cells})
    tabular(pd.DataFrame(rows), TAB / "recall_by_kind.tex", align="llr" + "r" * 5)
    return rec


def tables_full_grid(grid):
    """Appendix: mean test results of every configuration (all experiments)."""
    agg = (grid.groupby(["method", "params", "contamination", "dataset"], dropna=False)
           [["test_pr_auc", "test_f1", "test_precision", "test_recall"]].mean()
           .reset_index())
    for method in METHODS:
        sub = agg[agg.method == method]
        keys = sub[["params", "contamination"]].drop_duplicates()
        lines = ["\\begin{longtable}{l" + "rr" * len(TASK1_DATASETS) + "}", "\\toprule",
                 "Konfiguracja & " + " & ".join(
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

def _panel_axes(n=5, h=2.5, sharey=True):
    fig, axes = plt.subplots(1, n, figsize=(13, h), sharey=sharey)
    for ax, name in zip(axes, NAMES):
        ax.set_title(name)
    return fig, axes


def fig_neighbors(grid):
    """PR-AUC of LOF and COF as a function of k."""
    sub = grid.drop_duplicates(["dataset", "seed", "method", "params"])
    fig, axes = _panel_axes()
    for ax, d in zip(axes, TASK1_DATASETS):
        for m in ("LOF", "COF"):
            s = sub[(sub.dataset == d) & (sub.method == m)]
            g = s.groupby("n_neighbors").test_pr_auc.agg(["mean", "std"])
            ax.plot(g.index, g["mean"], marker="o", color=METHOD_COLORS[m], label=m)
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                            color=METHOD_COLORS[m], alpha=0.12, lw=0)
        ax.set_xlabel("liczba sąsiadów $k$")
    axes[0].set_ylabel("PR-AUC (test)")
    axes[0].legend()
    save(fig, FIG / "neighbors_pr_auc")


def fig_iforest(grid):
    sub = grid[grid.method == "IForest"].drop_duplicates(["dataset", "seed", "params"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 2.7))
    for i, d in enumerate(TASK1_DATASETS):
        g = sub[sub.dataset == d].groupby("n_estimators")
        color = BLUES(0.35 + 0.15 * i)
        a1.plot(g.test_pr_auc.mean(), marker="o", color=color, label=DISPLAY_NAMES[d])
        a2.plot(g.test_pr_auc.std(), marker="o", color=color)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xticks([50, 100, 200, 500], ["50", "100", "200", "500"])
        ax.set_xlabel("liczba drzew $T$")
    a1.set_ylabel("PR-AUC (średnia)")
    a2.set_ylabel("PR-AUC (odch. std. między podziałami)")
    a1.legend(ncol=2, fontsize=7)
    save(fig, FIG / "iforest_estimators")


def fig_contamination(grid):
    """Precision and recall of contamination-thresholded detectors."""
    sub = grid[grid.method.isin(["IForest", "LOF", "COF"])]
    best = sub.groupby(["dataset", "method", "params"]).val_pr_auc.mean().reset_index()
    best = best.loc[best.groupby(["dataset", "method"]).val_pr_auc.idxmax()]
    sub = sub.merge(best[["dataset", "method", "params"]])
    fig, axes = plt.subplots(2, 5, figsize=(13, 4.6), sharex=True, sharey="row")
    for j, d in enumerate(TASK1_DATASETS):
        axes[0, j].set_title(DISPLAY_NAMES[d])
        for m in ("IForest", "LOF", "COF"):
            g = sub[(sub.dataset == d) & (sub.method == m)].groupby("contamination")
            axes[0, j].plot(g.test_precision.mean(), marker="o", color=METHOD_COLORS[m], label=m)
            axes[1, j].plot(g.test_recall.mean(), marker="o", color=METHOD_COLORS[m])
        true_rate = SPECS[d].anomalies / SPECS[d].n
        for ax in axes[:, j]:
            ax.axvline(true_rate, color="#52514e", lw=0.8, ls="--")
        axes[1, j].set_xlabel("contamination $c$")
    axes[0, 0].set_ylabel("Precision (test)")
    axes[1, 0].set_ylabel("Recall (test)")
    axes[0, 0].legend()
    save(fig, FIG / "contamination_effect")


def fig_ocsvm(grid):
    sub = grid[grid.method == "OCSVM"]
    # (a) rbf: nu x gamma heat maps of test PR-AUC
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.6))
    nus = sorted(sub.nu.unique())
    gammas = [0.1, 1.0, 10.0]
    for ax, d in zip(axes, TASK1_DATASETS):
        s = sub[(sub.dataset == d) & (sub.kernel == "rbf")]
        M = s.groupby(["nu", "gamma_factor"]).test_pr_auc.mean().unstack()[gammas].loc[nus]
        ax.imshow(M.values, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(M.values):
            ax.text(j, i, num(v, 2), ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 else "#0b0b0b")
        ax.set_xticks(range(3), ["0,1/d", "1/d", "10/d"])
        ax.set_yticks(range(len(nus)), [num(n, 2) for n in nus])
        ax.set_xlabel("$\\gamma$")
        ax.set_title(DISPLAY_NAMES[d])
        ax.grid(False)
    axes[0].set_ylabel("$\\nu$")
    save(fig, FIG / "ocsvm_rbf_heatmap")

    # (b) every kernel with nu and gamma chosen on validation data
    fig, ax = plt.subplots(figsize=(8, 2.7))
    kernels = ["rbf", "poly", "sigmoid", "linear"]
    shades = [BLUES(v) for v in (0.85, 0.6, 0.4, 0.2)]
    width = 0.2
    for i, k in enumerate(kernels):
        vals = []
        for d in TASK1_DATASETS:
            g = sub[(sub.dataset == d) & (sub.kernel == k)].groupby("params")
            vals.append(g.test_pr_auc.mean()[g.val_pr_auc.mean().idxmax()])
        ax.bar(np.arange(5) + (i - 1.5) * width, vals, width * 0.92, color=shades[i], label=k)
    ax.set_xticks(range(5), NAMES)
    ax.set_ylabel("PR-AUC (test)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, FIG / "ocsvm_kernels")

    # (c) nu controls the fraction of flagged points
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6))
    s = sub[(sub.kernel == "rbf") & (sub.gamma_factor == 1.0)]
    for i, d in enumerate(TASK1_DATASETS):
        g = s[s.dataset == d].groupby("nu")
        color = BLUES(0.35 + 0.15 * i)
        axes[0].plot(g.test_flagged.mean(), marker="o", color=color, label=DISPLAY_NAMES[d])
        axes[1].plot(g.test_recall.mean(), g.test_precision.mean(), marker="o", color=color)
    axes[0].plot([0, 0.5], [0, 0.5], color="#52514e", lw=0.8, ls="--")
    axes[0].set_xlabel("$\\nu$")
    axes[0].set_ylabel("odsetek oznaczonych (test)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[0].legend(fontsize=7)
    save(fig, FIG / "ocsvm_nu")


def fig_dbscan(grid):
    sub = grid[grid.method == "DBSCAN"]
    fig, axes = plt.subplots(2, 5, figsize=(13, 4.8))
    qs = sorted(sub.eps_quantile.unique())
    ms = sorted(int(m) for m in sub.min_samples.unique())
    for j, d in enumerate(TASK1_DATASETS):
        s = sub[sub.dataset == d]
        for i, (col, label) in enumerate((("test_f1", "F1"), ("test_flagged", "odsetek szumu"))):
            M = s.groupby(["min_samples", "eps_quantile"])[col].mean().unstack().loc[ms, qs]
            ax = axes[i, j]
            ax.imshow(M.values, cmap=BLUES, vmin=0, vmax=1 if i == 0 else 0.6, aspect="auto")
            for (a, b), v in np.ndenumerate(M.values):
                ax.text(b, a, num(v, 2), ha="center", va="center", fontsize=7,
                        color="white" if v > (0.6 if i == 0 else 0.36) else "#0b0b0b")
            ax.set_xticks(range(len(qs)), [num(q, 2) for q in qs], fontsize=7)
            ax.set_yticks(range(len(ms)), ms)
            ax.grid(False)
            if i == 0:
                ax.set_title(DISPLAY_NAMES[d])
            else:
                ax.set_xlabel("kwantyl $q_\\varepsilon$")
            if j == 0:
                ax.set_ylabel(f"{label}\nminPts")
    save(fig, FIG / "dbscan_heatmap")


def fig_best(best):
    fig, axes = plt.subplots(1, 2, figsize=(13, 2.9), sharey=True)
    width = 0.16
    for ax, metric, label in ((axes[0], "pr_auc", "PR-AUC"), (axes[1], "f1", "F1")):
        for i, m in enumerate(METHODS):
            g = best[best.method == m].groupby("dataset")[metric].agg(["mean", "std"])
            g = g.loc[list(TASK1_DATASETS)]
            ax.bar(np.arange(5) + (i - 2) * width, g["mean"], width * 0.9,
                   yerr=g["std"], color=METHOD_COLORS[m], label=m,
                   error_kw={"elinewidth": 0.8, "ecolor": "#52514e", "capsize": 1.5})
        ax.set_xticks(range(5), NAMES)
        ax.set_ylabel(f"{label} (test)")
        ax.set_ylim(0, 1.05)
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(1.0, 1.2))
    save(fig, FIG / "best_comparison")


def fig_time_quality(grid):
    sub = grid.drop_duplicates(["dataset", "seed", "method", "params"])
    agg = sub.groupby(["dataset", "method", "params"]).agg(
        t=("fit_time", "mean"), s=("score_time", "mean"), pr=("test_pr_auc", "mean")).reset_index()
    fig, axes = _panel_axes(h=2.7)
    for ax, d in zip(axes, TASK1_DATASETS):
        for m in METHODS:
            s = agg[(agg.dataset == d) & (agg.method == m)]
            ax.scatter(1000 * (s.t + s.s), s.pr, s=12, color=METHOD_COLORS[m],
                       alpha=0.8, lw=0, label=m)
        ax.set_xscale("log")
        ax.set_xlabel("czas uczenia + oceny [ms]")
    axes[0].set_ylabel("PR-AUC (test)")
    axes[0].legend(ncol=1, fontsize=7, markerscale=1.5)
    save(fig, FIG / "time_vs_quality")


def fig_agreement(preds):
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.8))
    for ax, d in zip(axes, TASK1_DATASETS):
        mats = []
        for _, g in preds[preds.dataset == d].groupby("seed"):
            P = g.pivot(index="row", columns="method", values="pred")[list(METHODS)]
            mats.append(jaccard_matrix(P.values.T))
        M = np.nanmean(mats, axis=0)
        ax.imshow(M, cmap=BLUES, vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, num(v, 2), ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.6 else "#0b0b0b")
        ax.set_xticks(range(5), METHODS, rotation=60, fontsize=7)
        ax.set_yticks(range(5), METHODS if d == TASK1_DATASETS[0] else [""] * 5, fontsize=7)
        ax.set_title(DISPLAY_NAMES[d])
        ax.grid(False)
    save(fig, FIG / "jaccard_heatmap")


def fig_overlap(preds):
    """How many detectors flag each true anomaly and each false alarm."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.8), sharey=True)
    shades = [BLUES(v) for v in np.linspace(0.12, 0.95, 6)]
    for ax, label_value, title in ((axes[0], 1, "prawdziwe anomalie"),
                                   (axes[1], 0, "fałszywe alarmy (co najmniej 1 detektor)")):
        bottoms = np.zeros(5)
        shares = []
        for d in TASK1_DATASETS:
            g = preds[(preds.dataset == d) & (preds.y == label_value)]
            votes = g.groupby(["seed", "row"]).pred.sum()
            if label_value == 0:
                votes = votes[votes > 0]
            shares.append(np.bincount(votes, minlength=6)[:6] / len(votes))
        shares = np.array(shares)
        start = 0 if label_value == 1 else 1
        for v in range(start, 6):
            ax.bar(range(5), shares[:, v], 0.6, bottom=bottoms, color=shades[v],
                   edgecolor="white", lw=1, label=f"{v}")
            bottoms += shares[:, v]
        ax.set_xticks(range(5), NAMES, fontsize=7.5)
        ax.set_title(title)
    axes[0].set_ylabel("udział obserwacji")
    axes[0].legend(title="liczba detektorów", ncol=6, loc="upper center",
                   bbox_to_anchor=(1.05, 1.32))
    save(fig, FIG / "detector_overlap")


def fig_kinds(rec):
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.6), sharey=True)
    for ax, d in zip(axes, TASK1_DATASETS):
        kinds = [k for k in ("global", "local") if (d, k) in rec.index]
        width = 0.8 / len(kinds)
        shades = {"global": BLUES(0.85), "local": BLUES(0.45)}
        for i, k in enumerate(kinds):
            ax.bar(np.arange(5) + (i - (len(kinds) - 1) / 2) * width, rec.loc[(d, k)].values,
                   width * 0.9, color=shades[k], label=TYPE_LABELS[k])
        ax.set_xticks(range(5), METHODS, rotation=45, fontsize=7)
        ax.set_title(DISPLAY_NAMES[d])
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUES(v)) for v in (0.85, 0.45)]
    fig.legend(handles, [TYPE_LABELS[k] for k in ("global", "local")],
               ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.1))
    axes[0].set_ylabel("Recall wg typu anomalii")
    save(fig, FIG / "recall_by_kind")


def main():
    setup()
    grid = pd.read_csv(OUT / "grid.csv")
    best = pd.read_csv(OUT / "best_metrics.csv")
    preds = pd.read_csv(OUT / "test_predictions.csv.gz")

    agg = table_best(best)
    agg.to_csv(OUT / "best_summary.csv", index=False)
    table_accuracy_trap(best, preds)
    rec = table_kinds(preds)
    tables_full_grid(grid)

    fig_neighbors(grid)
    fig_iforest(grid)
    fig_contamination(grid)
    fig_ocsvm(grid)
    fig_dbscan(grid)
    fig_best(best)
    fig_time_quality(grid)
    fig_agreement(preds)
    fig_overlap(preds)
    fig_kinds(rec)


if __name__ == "__main__":
    main()
