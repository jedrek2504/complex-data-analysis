"""Task 3: tables and figures, including the synthesis of all three tasks."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT
from common.datasets import DISPLAY_NAMES, TASK1_DATASETS, TASK3_DATASETS
from common.latex import num, pm, tabular
from common.style import DIVERGING, WIDTH, bar, save, setup, styles_for
from task3_ensemble.ensemble import normalise
from task3_ensemble.run_base import BASE, OUT

FIG, TAB = OUT / "figures", OUT / "tables"
NORMAL_HIST = "#bdbdbd"
ANOMALY_HIST = "#D55E00"
NAMES = [DISPLAY_NAMES[d] for d in TASK3_DATASETS]
BASE_LABEL = {"IForest": "IForest", "LOF": "LOF", "OCSVM": "OCSVM", "PCA": "PCA (rekonstr.)"}
BASE_STYLE = {
    "IForest": {"color": "#0072B2", "marker": "o", "ls": "-", "hatch": ""},
    "LOF": {"color": "#D55E00", "marker": "s", "ls": "--", "hatch": "//"},
    "OCSVM": {"color": "#E69F00", "marker": "D", "ls": ":", "hatch": ".."},
    "PCA": {"color": "#009E73", "marker": "^", "ls": "-.", "hatch": "\\\\"},
}
DATASET_STYLE = styles_for(TASK3_DATASETS)
KEY_ROWS = ["best-single", "MV 2/4", "Fusion mean", "Fusion skill", "DSF", "CBF"]
KEY_STYLE = styles_for(KEY_ROWS)
MAIN_ROWS = ["IForest", "LOF", "OCSVM", "PCA", "best-single", "MV 1/4", "MV 2/4", "MV 3/4",
             "MV 4/4", "max rank/val", "Fusion mean", "Fusion AP", "Fusion skill", "DSF", "CBF"]
ROW_LABEL = {"best-single": "najlepszy pojedynczy", "max rank/val": "fuzja: maksimum",
             "Fusion mean": "fuzja: średnia", "Fusion AP": "fuzja: wagi PR-AUC",
             "Fusion skill": "fuzja: wagi skill", "DSF": "DSF (własna)",
             "CBF": "CBF (własna)", "PCA": "PCA (rekonstr.)"}
SCHEMES = ["uniform", "ap", "skill", "diverse"]
NORMS = ["minmax", "zscore", "robust", "rank"]
REP_LABEL = {"Original": "oryginalna", "PCA": "PCA", "UMAP": "UMAP", "AE": "autoenkoder"}


def signed(v, digits=3):
    return ("+" if v >= 0 else "") + num(v, digits)


def summary(variants):
    g = variants.groupby(["dataset", "variant"])
    agg = g.mean(numeric_only=True)
    sd = g.std(numeric_only=True)
    agg["pr_auc_sd"], agg["f1_sd"] = sd.pr_auc, sd.f1
    return agg.reset_index()


# ------------------------------------------------------------------ tables

def table_base_configs():
    cfg = pd.read_csv(OUT / "base_configs.csv")
    rows = []
    for d in TASK3_DATASETS:
        for i, b in enumerate(BASE):
            r = cfg[(cfg.dataset == d) & (cfg.base == b)].iloc[0]
            rp, dp = json.loads(r.rep_params), json.loads(r.det_params)
            rep = REP_LABEL[r.rep]
            if r.rep == "PCA":
                rep += f" ($k$={rp['n_components']})"
            elif r.rep == "UMAP":
                rep += f" ($m$={rp['n_components']}, nn={rp['n_neighbors']}, md={num(rp['min_dist'], 1)})"
            elif r.rep == "AE":
                rep += f" ($z$={rp['latent_dim']})"
            det = {"LOF": lambda: f"$k$={dp['n_neighbors']}",
                   "OCSVM": lambda: f"$\\nu$={num(dp['nu'], 2)}, $\\gamma$={num(dp['gamma_factor'], 1)}/$d$",
                   "IForest": lambda: f"$T$={dp.get('n_estimators', '')}",
                   "Reconstruction": lambda: "błąd rekonstrukcji"}[r.detector]()
            rows.append({"Zbiór danych": DISPLAY_NAMES[d] if i == 0 else "",
                         "Detektor": BASE_LABEL[b], "Reprezentacja": rep, "Parametry": det,
                         "\\makecell[r]{PR-AUC na walidacji\\\\(zadanie 2)}": r.val_pr_auc_task2})
    tabular(pd.DataFrame(rows), TAB / "base_configs.tex", align="llllr")


def table_main(summ):
    lines = ["\\begin{longtable}{llrrrrrr}", "\\toprule",
             "Zbiór danych & Wariant & Precision & Recall & F1 & ROC-AUC & PR-AUC & Czas [ms] \\\\",
             "\\midrule", "\\endhead"]
    for d in TASK3_DATASETS:
        sub = summ[summ.dataset == d].set_index("variant")
        best_f1 = sub.loc[MAIN_ROWS, "f1"].max()
        best_pr = sub.loc[MAIN_ROWS, "pr_auc"].max()
        for i, v in enumerate(MAIN_ROWS):
            r = sub.loc[v]
            f1 = pm(r.f1, r.f1_sd)
            pr = pm(r.pr_auc, r.pr_auc_sd)
            if np.isclose(r.f1, best_f1):
                f1 = f"\\textbf{{{f1}}}"
            if np.isclose(r.pr_auc, best_pr):
                pr = f"\\textbf{{{pr}}}"
            cells = [DISPLAY_NAMES[d] if i == 0 else "", ROW_LABEL.get(v, v),
                     num(r.precision), num(r.recall), f1, num(r.roc_auc), pr,
                     num(1000 * r.time, 0)]
            lines.append(" & ".join(cells) + " \\\\")
            if v in ("PCA", "best-single", "MV 4/4"):
                lines.append("\\cmidrule(l){2-8}")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{longtable}")
    (TAB / "main_results.tex").write_text("\n".join(lines) + "\n")


def table_ablation(variants):
    """Mean over datasets of PR-AUC and F1 gain over the best single detector."""
    fus = variants[(variants.family == "fusion") & variants.weights.isin(SCHEMES)
                   & (variants.fit_on == "val")]
    best = variants[variants.variant == "best-single"][["dataset", "seed", "pr_auc", "f1"]]
    fus = fus.merge(best, on=["dataset", "seed"], suffixes=("", "_best"))
    fus["d_pr"] = fus.pr_auc - fus.pr_auc_best
    fus["d_f1"] = fus.f1 - fus.f1_best
    rows = []
    for (norm, scheme, sel), g in fus.groupby(["norm", "weights", "selection"]):
        per_ds = g.groupby(["dataset", "threshold"])[["d_pr", "d_f1", "pr_auc", "f1"]].mean()
        f1_rule = per_ds.xs("f1", level="threshold")
        c_rule = per_ds.xs("contamination", level="threshold")
        rows.append({"norm": norm, "weights": scheme, "selection": sel,
                     "pr_auc": f1_rule.pr_auc.mean(), "d_pr": f1_rule.d_pr.mean(),
                     "f1_val": f1_rule.f1.mean(), "d_f1_val": f1_rule.d_f1.mean(),
                     "f1_c": c_rule.f1.mean(), "d_f1_c": c_rule.d_f1.mean(),
                     "wins": int((f1_rule.d_pr > 0).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ablation.csv", index=False)
    order = {"minmax": 0, "zscore": 1, "robust": 2, "rank": 3}
    sch = {"uniform": "równe", "ap": "PR-AUC", "skill": "skill", "diverse": "skill$\\times$różn."}
    w_order = {w: i for i, w in enumerate(SCHEMES)}
    df = df.sort_values(["norm", "weights", "selection"],
                        key=lambda c: c.map(order) if c.name == "norm" else
                        (c.map(w_order) if c.name == "weights" else c))
    out = pd.DataFrame({
        "Normalizacja": df.norm.map({"minmax": "min-max", "zscore": "z-score",
                                     "robust": "odporny z", "rank": "rangi"}),
        "Wagi": df.weights.map(sch),
        "Detektory": df.selection.map({"all": "wszystkie", "greedy": "zachłanny wybór"}),
        "PR-AUC": [num(v) for v in df.pr_auc],
        "\\makecell[r]{Zmiana\\\\PR-AUC}": [("+" if v >= 0 else "") + num(v) for v in df.d_pr],
        "\\makecell[r]{F1 (próg\\\\z walidacji)}": [num(v) for v in df.f1_val],
        "\\makecell[r]{Zmiana\\\\F1}": [("+" if v >= 0 else "") + num(v) for v in df.d_f1_val],
        "\\makecell[r]{F1 (próg\\\\z udziału $c$)}": [num(v) for v in df.f1_c],
        "\\makecell[r]{Wygrane\\\\(z 4)}": df.wins.astype(str)})
    tabular(out, TAB / "ablation.tex", align="lllrrrrrr")
    return df


def table_cases(dec):
    rows = []
    for d in TASK3_DATASETS:
        sub = dec[dec.dataset == d]
        for ens in ("MV 2/4", "MV 3/4", "Fusion mean", "DSF", "CBF"):
            a, n = sub[sub.y == 1], sub[sub.y == 0]
            rows.append({
                "Zbiór danych": DISPLAY_NAMES[d] if ens == "MV 2/4" else "",
                "Zespół": ROW_LABEL.get(ens, ens).replace("\\textbf{", "").replace("}", ""),
                "Anomalie": str(len(a)),
                "\\makecell[r]{Anomalie\\\\uratowane}":
                    str(int(((a[ens] == 1) & (a["best-single"] == 0)).sum())),
                "\\makecell[r]{Anomalie\\\\utracone}":
                    str(int(((a[ens] == 0) & (a["best-single"] == 1)).sum())),
                "\\makecell[r]{Fałszywe alarmy\\\\najlepszego pojed.}":
                    str(int(n["best-single"].sum())),
                "\\makecell[r]{Fałszywe alarmy\\\\usunięte}":
                    str(int(((n[ens] == 0) & (n["best-single"] == 1)).sum())),
                "\\makecell[r]{Fałszywe alarmy\\\\dodane}":
                    str(int(((n[ens] == 1) & (n["best-single"] == 0)).sum())),
            })
    tabular(pd.DataFrame(rows), TAB / "cases.tex", align="llrrrrrr")


def table_normalisation(variants):
    """Normaliser fitted on in-sample training or out-of-sample validation scores."""
    rows = []
    label = {"minmax": "min-max", "zscore": "z-score", "robust": "odporny z", "rank": "rangi"}
    for norm in NORMS:
        def mean_of(name, metric="pr_auc"):
            return variants[variants.variant == name].groupby("dataset")[metric].mean().mean()
        mean_tr = mean_of(f"fusion {norm}/train/uniform/all/f1")
        mean_va = mean_of(f"fusion {norm}/val/uniform/all/f1") if norm != "rank" else mean_of("Fusion mean")
        cbf = "CBF" if norm == "robust" else f"CBF {norm}"
        rows.append({"Normalizacja": label[norm],
                     "\\makecell[r]{Średnia:\\\\na uczących}": num(mean_tr),
                     "\\makecell[r]{Średnia:\\\\na walidacji}": num(mean_va),
                     "\\makecell[r]{Maksimum:\\\\na uczących}": num(mean_of(f"max {norm}/train")),
                     "\\makecell[r]{Maksimum:\\\\na walidacji}": num(mean_of(f"max {norm}/val")),
                     "\\makecell[r]{CBF:\\\\PR-AUC}": num(mean_of(cbf)),
                     "\\makecell[r]{CBF:\\\\F1}": num(mean_of(cbf, "f1"))})
    tabular(pd.DataFrame(rows), TAB / "normalisation.tex", align="lrrrrrr")


def table_confirmation(variants, confirm):
    rows = []
    names = ["best-single", "MV 2/4", "Fusion mean", "Fusion skill", "DSF", "CBF"]
    for v in names:
        r = {"Wariant": ROW_LABEL.get(v, v).replace("\\textbf{", "").replace("}", "")}
        for label, df in (("projektowe", variants), ("nowe", confirm)):
            sub = df[df.variant == v].groupby("dataset")[["pr_auc", "f1"]].mean()
            r[f"\\makecell[r]{{PR-AUC\\\\({label})}}"] = num(sub.pr_auc.mean())
            r[f"\\makecell[r]{{F1\\\\({label})}}"] = num(sub.f1.mean())
            best = df[df.variant == "best-single"].groupby("dataset")[["pr_auc", "f1"]].mean()
            r[f"\\makecell[r]{{Wygrane\\\\({label})}}"] = str(
                int((sub.pr_auc > best.pr_auc + 1e-9).sum()))
        rows.append(r)
    tabular(pd.DataFrame(rows), TAB / "confirmation.tex", align="lrrrrrr")
    per_ds = []
    for label, df in (("design", variants), ("confirm", confirm)):
        t = df[df.variant.isin(names)].groupby(["variant", "dataset"])[["pr_auc", "f1"]].mean()
        per_ds.append(t.assign(seeds=label))
    pd.concat(per_ds).to_csv(OUT / "confirmation.csv")


# ------------------------------------------------------------------ figures

def grid_axes(height=4.8, sharey=True, sharex=False):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, height), sharey=sharey, sharex=sharex)
    return fig, axes.ravel()


def table_key(summ):
    """Main variants: PR-AUC per dataset and on average, mean F1 and time."""
    rows = []
    for v in KEY_ROWS:
        sub = summ[summ.variant == v].set_index("dataset").loc[list(TASK3_DATASETS)]
        r = {"Wariant": ROW_LABEL.get(v, v)}
        for d in TASK3_DATASETS:
            r[DISPLAY_NAMES[d]] = sub.loc[d, "pr_auc"]
        r["\\makecell[r]{PR-AUC\\\\średnio}"] = sub.pr_auc.mean()
        r["\\makecell[r]{F1\\\\średnio}"] = sub.f1.mean()
        r["\\makecell[r]{Czas\\\\średnio [s]}"] = num(sub.time.mean(), 1)
        rows.append(r)
    df = pd.DataFrame(rows)
    tabular(df, TAB / "key_results.tex", align="lrrrrrrr",
            bold_max=[DISPLAY_NAMES[d] for d in TASK3_DATASETS]
            + ["\\makecell[r]{PR-AUC\\\\średnio}", "\\makecell[r]{F1\\\\średnio}"])


def fig_key(summ):
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
    width = 0.13
    n = len(NAMES) + 1
    for i, v in enumerate(KEY_ROWS):
        sub = summ[summ.variant == v].set_index("dataset").loc[list(TASK3_DATASETS)]
        vals = list(sub.pr_auc) + [sub.pr_auc.mean()]
        bar(ax, np.arange(n) + (i - (len(KEY_ROWS) - 1) / 2) * width, vals, width * 0.92,
            KEY_STYLE[v], ROW_LABEL.get(v, v))
    ax.set_xticks(range(n), NAMES + ["średnio"])
    ax.set_ylabel("PR-AUC (test)")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.3))
    save(fig, FIG / "key_comparison")


def fig_jaccard():
    jac = pd.read_csv(OUT / "jaccard.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 6.0), constrained_layout=True)
    labels = ["IForest", "LOF", "OCSVM", "PCA"]
    for ax, d in zip(axes.ravel(), TASK3_DATASETS):
        M = (jac[jac.dataset == d].groupby(["a", "b"]).jaccard.mean().unstack()
             .loc[BASE, BASE].values)
        im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, num(v, 2), ha="center", va="center", fontsize=11,
                    color="white" if v > 0.55 else "black")
        ax.set_xticks(range(4), labels, rotation=30, ha="right")
        ax.set_yticks(range(4), labels)
        ax.set_title(DISPLAY_NAMES[d])
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.6)
    cbar.set_label("indeks Jaccarda (1 = te same punkty)")
    save(fig, FIG / "jaccard_heatmap")


def fig_voting(summ):
    fig, axes = grid_axes()
    metrics = {"precision": "Precision", "recall": "Recall", "f1": "F1"}
    styles = styles_for(metrics)
    m = [1, 2, 3, 4]
    for ax, d in zip(axes, TASK3_DATASETS):
        sub = summ[summ.dataset == d].set_index("variant")
        for metric, label in metrics.items():
            st = styles[metric]
            ax.plot(m, [sub.loc[f"MV {i}/4", metric] for i in m], color=st["color"],
                    marker=st["marker"], ls="-", label=label)
            ax.plot(m, [sub.loc[f"MV {i}/4 (c)", metric] for i in m], color=st["color"],
                    marker=st["marker"], ls="--", lw=1.2, alpha=0.7)
        ax.axhline(sub.loc["best-single", "f1"], color="#555555", lw=1.2, ls=":")
        ax.set_xticks(m, ["1/4", "2/4", "3/4", "4/4"])
        ax.set_title(DISPLAY_NAMES[d])
        ax.set_ylim(0, 1.02)
    for ax in axes[2:]:
        ax.set_xlabel("wymagana liczba głosów")
    for ax in axes[::2]:
        ax.set_ylabel("wartość (test)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="#555555", ls="--", lw=1.2),
                plt.Line2D([], [], color="#555555", ls=":", lw=1.2)]
    labels += ["progi z udziału anomalii", "F1 najlepszego pojedynczego"]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.98))
    fig.subplots_adjust(hspace=0.35)
    save(fig, FIG / "voting_threshold")


def fig_ablation_heatmap(abl):
    order_n = ["minmax", "zscore", "robust", "rank"]
    order_w = ["uniform", "ap", "skill", "diverse"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6), constrained_layout=True)
    lim = max(0.05, np.abs(abl.d_pr).max())
    for ax, sel, title in ((axes[0], "all", "wszystkie detektory"),
                           (axes[1], "greedy", "zachłanny wybór detektorów")):
        sub = abl[abl.selection == sel].set_index(["norm", "weights"])
        M = np.array([[sub.loc[(n, w), "d_pr"] for w in order_w] for n in order_n])
        im = ax.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim)
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, signed(v), ha="center", va="center", fontsize=10)
        ax.set_xticks(range(4), ["równe", "PR-AUC", "skill", "skill×różn."], rotation=30,
                      ha="right")
        ax.set_yticks(range(4), ["min-max", "z-score", "odporny z", "rangi"]
                      if sel == "all" else [""] * 4)
        ax.set_xlabel("wagi")
        ax.set_title(title)
        ax.grid(False)
    axes[0].set_ylabel("normalizacja")
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label("zmiana PR-AUC względem\nnajlepszego pojedynczego")
    save(fig, FIG / "ablation_heatmap")


def fig_complementarity(variants):
    comp = pd.read_csv(OUT / "complementarity.csv")
    best = variants[variants.variant == "best-single"][["dataset", "seed", "pr_auc", "f1"]]
    fig, axes = grid_axes(height=5.4, sharey=False)
    panels = ((axes[0], "Fusion mean", "mean_jaccard", "średni indeks Jaccarda detektorów"),
              (axes[1], "Fusion mean", "unique_hits", "anomalie wykryte przez 1 detektor"),
              (axes[2], "MV 2/4", "false_alarms_single", "fałszywe alarmy 1 detektora"),
              (axes[3], "Fusion mean", "val_quality_gap",
               "przewaga najlepszego (PR-AUC wal.)"))
    for ax, ens, x, xlabel in panels:
        e = variants[variants.variant == ens][["dataset", "seed", "pr_auc", "f1"]]
        df = comp.merge(e, on=["dataset", "seed"]).merge(best, on=["dataset", "seed"],
                                                          suffixes=("", "_best"))
        metric = "f1" if ens.startswith("MV") else "pr_auc"
        df["gain"] = df[metric] - df[f"{metric}_best"]
        for d in TASK3_DATASETS:
            sub, st = df[df.dataset == d], DATASET_STYLE[d]
            ax.scatter(sub[x], sub.gain, s=26, color=st["color"], marker=st["marker"],
                       label=DISPLAY_NAMES[d], edgecolor="white", lw=0.4)
        r = np.corrcoef(df[x], df.gain)[0, 1]
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"zysk {'F1' if metric == 'f1' else 'PR-AUC'}")
        name = "MV 2/4" if ens.startswith("MV") else "fuzja średnia"
        ax.set_title(f"{name}, $r$ = {num(r, 2)}", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.98),
               markerscale=1.4)
    fig.subplots_adjust(hspace=0.45, wspace=0.3)
    save(fig, FIG / "complementarity_gain")


def fig_weights(variants):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.4), sharey=True)
    for ax, v, title in ((axes[0], "Fusion skill", "fuzja: wagi skill"),
                         (axes[1], "DSF", "DSF"),
                         (axes[2], "CBF", "CBF")):
        sub = variants[variants.variant == v]
        W = np.array([np.array(json.loads(w)) for w in sub.w])
        mean_w = pd.DataFrame(W, columns=BASE).assign(dataset=sub.dataset.values).groupby(
            "dataset").mean().loc[list(TASK3_DATASETS)]
        bottom = np.zeros(len(mean_w))
        for b in BASE:
            st = BASE_STYLE[b]
            ax.bar(range(len(mean_w)), mean_w[b], 0.65, bottom=bottom, color=st["color"],
                   hatch=st["hatch"], label=BASE_LABEL[b], edgecolor="white", lw=1)
            bottom += mean_w[b].values
        ax.set_xticks(range(len(mean_w)), NAMES, rotation=35, ha="right")
        ax.set_title(title)
    axes[0].set_ylabel("średnia waga")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.97))
    fig.subplots_adjust(wspace=0.3)
    save(fig, FIG / "weights")


def fig_normalisation():
    """Why normalisers should be fitted on out-of-sample scores (one split)."""
    scores = pd.read_csv(OUT / "base_scores.csv.gz")
    sub = scores[(scores.dataset == "campaign") & (scores.seed == 0) & (scores.base == "OCSVM")]
    parts = {p: sub[sub.part == p].score.values for p in ("train", "val", "test")}
    y_test = sub[sub.part == "test"].y.values
    fig, axes = grid_axes(height=4.8, sharey=False)
    styles = {"train": ("#0072B2", "-", "uczące (in-sample)"),
              "val": ("#E69F00", "--", "walidacyjne"),
              "test": ("#000000", ":", "testowe")}
    bins = np.linspace(min(map(np.min, parts.values())), max(map(np.max, parts.values())), 30)
    for p_, v in parts.items():
        color, ls, label = styles[p_]
        axes[0].hist(v, bins=bins, histtype="step", density=True, lw=2, color=color, ls=ls,
                     label=label)
    axes[0].set_title("(a) surowe wyniki OCSVM")
    axes[0].legend(fontsize=9.5)
    panels = ((axes[1], "rank", parts["train"], "(b) rangi dopasowane na uczących"),
              (axes[2], "rank", parts["val"], "(c) rangi dopasowane na walidacji"),
              (axes[3], "robust", parts["val"], "(d) odporny z dopasowany na walidacji"))
    for ax, method, fit, title in panels:
        z = normalise(fit[None], parts["test"][None], method=method)[0][0]
        ax.hist(z[y_test == 0], bins=25, color=NORMAL_HIST, label="normalne (test)")
        ax.hist(z[y_test == 1], bins=25, color=ANOMALY_HIST, alpha=0.8, hatch="//",
                edgecolor="white", label="anomalie (test)")
        ax.set_title(title, fontsize=11)
    axes[1].legend(fontsize=9.5)
    for ax in axes:
        ax.set_yticks([])
    fig.subplots_adjust(hspace=0.4)
    save(fig, FIG / "normalisation")


# --------------------------------------------------------------- synthesis

def synthesis(summ):
    """Effect of algorithm, representation and ensemble on test PR-AUC."""
    t1 = pd.read_csv(ROOT / "results" / "task1" / "best_metrics.csv")
    t1_mean = t1.groupby(["dataset", "method"]).pr_auc.mean()
    # task 1 has its own datasets: the spread between its five methods
    t1_spread = pd.DataFrame({"dataset": list(TASK1_DATASETS), "algo_spread_t1": [
        t1_mean.loc[d].max() - t1_mean.loc[d].min() for d in TASK1_DATASETS]})
    t1_spread.to_csv(OUT / "synthesis_task1_spread.csv", index=False)
    t2 = pd.read_csv(ROOT / "results" / "task2" / "selected_configs.csv")
    rows, ladder = [], []
    for d in TASK3_DATASETS:
        orig = t2[(t2.dataset == d) & (t2.rep == "Original") & (t2.detector != "Reconstruction")]
        reps = t2[(t2.dataset == d) & t2.detector.isin(["IForest", "LOF", "OCSVM"])]
        rep_spread = reps.groupby("detector").test_pr_auc.agg(lambda s: s.max() - s.min()).mean()
        best_t2 = t2[t2.dataset == d].loc[t2[t2.dataset == d].val_pr_auc.idxmax()]
        s = summ[summ.dataset == d].set_index("variant")
        rows.append({
            "dataset": d,
            "algo_spread_t2": orig.test_pr_auc.max() - orig.test_pr_auc.min(),
            "rep_spread": rep_spread,
            "ens_gain_cbf": s.loc["CBF", "pr_auc"] - s.loc["best-single", "pr_auc"],
            "ens_gain_mean": s.loc["Fusion mean", "pr_auc"] - s.loc["best-single", "pr_auc"],
        })
        base = orig[orig.detector == "IForest"].test_pr_auc.iloc[0]
        algo = orig.loc[orig.val_pr_auc.idxmax()]
        ladder.append({"dataset": d,
                       "IForest (oryg.)": base,
                       "Zad. 2: wybór algorytmu (oryg.)": algo.test_pr_auc,
                       "Zad. 2: wybór repr. i algorytmu": best_t2.test_pr_auc,
                       "Zad. 3: najl. pojedynczy": s.loc["best-single", "pr_auc"],
                       "Zad. 3: fuzja średnia": s.loc["Fusion mean", "pr_auc"],
                       "Zad. 3: CBF": s.loc["CBF", "pr_auc"],
                       "Zad. 3: MV 2/4 (F1)": s.loc["MV 2/4", "f1"],
                       "Zad. 3: CBF (F1)": s.loc["CBF", "f1"],
                       "Zad. 3: najl. pojed. (F1)": s.loc["best-single", "f1"],
                       "algo_t2": algo.detector, "rep_t2": f"{best_t2.rep}/{best_t2.detector}"})
    eff = pd.DataFrame(rows)
    lad = pd.DataFrame(ladder)
    eff.to_csv(OUT / "synthesis_effects.csv", index=False)
    lad.to_csv(OUT / "synthesis_ladder.csv", index=False)

    out = pd.DataFrame({
        "Zbiór danych": [DISPLAY_NAMES[d] for d in eff.dataset],
        "\\makecell[r]{Rozrzut: algorytm\\\\(dane oryginalne)}": [num(v) for v in eff.algo_spread_t2],
        "\\makecell[r]{Rozrzut:\\\\reprezentacja}": [num(v) for v in eff.rep_spread],
        "\\makecell[r]{Zysk zespołu:\\\\fuzja średnia}": [signed(v) for v in eff.ens_gain_mean],
        "\\makecell[r]{Zysk zespołu:\\\\CBF}": [signed(v) for v in eff.ens_gain_cbf]})
    mean_row = dict(zip(out.columns, ["\\textit{średnio}", num(eff.algo_spread_t2.mean()),
                                      num(eff.rep_spread.mean()), signed(eff.ens_gain_mean.mean()),
                                      signed(eff.ens_gain_cbf.mean())]))
    out = pd.concat([out, pd.DataFrame([mean_row])])
    tabular(out, TAB / "synthesis_effects.tex", align="lrrrr")

    cols = ["IForest (oryg.)", "Zad. 2: wybór algorytmu (oryg.)",
            "Zad. 2: wybór repr. i algorytmu", "Zad. 3: najl. pojedynczy",
            "Zad. 3: fuzja średnia", "Zad. 3: CBF"]
    stage_label = dict(zip(cols, ["IForest, dane oryginalne", "wybór algorytmu (zad. 2)",
                                  "wybór reprezentacji i algorytmu (zad. 2)",
                                  "najlepszy pojedynczy (zad. 3)", "fuzja średnia (zad. 3)",
                                  "CBF (zad. 3)"]))
    t = pd.DataFrame({"Etap": [stage_label[c] for c in cols]})
    for _, r in lad.iterrows():
        t[DISPLAY_NAMES[r.dataset]] = [num(r[c]) for c in cols]
    t["Średnio"] = [num(lad[c].mean()) for c in cols]
    tabular(t, TAB / "synthesis_ladder.tex", align="lrrrrr")

    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    width = 0.13
    stage = styles_for(cols)
    n = len(NAMES) + 1
    for i, c in enumerate(cols):
        vals = list(lad[c]) + [lad[c].mean()]
        bar(ax, np.arange(n) + (i - (len(cols) - 1) / 2) * width, vals, width * 0.92,
            stage[c], stage_label[c])
    ax.set_xticks(range(n), NAMES + ["średnio"])
    ax.set_ylabel("PR-AUC (test)")
    ax.set_ylim(0, 1)
    ax.legend(ncol=2, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, 1.42))
    save(fig, FIG / "synthesis_ladder")
    return eff, lad


def main():
    setup()
    TAB.mkdir(parents=True, exist_ok=True)
    variants = pd.read_csv(OUT / "variants.csv")
    dec = pd.read_csv(OUT / "test_decisions.csv.gz")
    summ = summary(variants)
    summ.to_csv(OUT / "summary.csv", index=False)

    table_base_configs()
    table_main(summ)
    abl = table_ablation(variants)
    table_cases(dec)
    table_normalisation(variants)
    table_confirmation(variants, pd.read_csv(OUT / "confirm" / "variants.csv"))
    table_key(summ)
    fig_key(summ)
    fig_jaccard()
    fig_voting(summ)
    fig_ablation_heatmap(abl)
    fig_complementarity(variants)
    fig_weights(variants)
    fig_normalisation()
    eff, lad = synthesis(summ)
    print(summ[summ.variant.isin(MAIN_ROWS)].pivot(index="variant", columns="dataset",
                                                   values="pr_auc").loc[MAIN_ROWS].round(3))
    print(eff.round(3))


if __name__ == "__main__":
    main()
