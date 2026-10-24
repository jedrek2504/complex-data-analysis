"""Task 3: tables and figures, including the synthesis of all three tasks."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT
from common.datasets import DISPLAY_NAMES, HIGHDIM_DATASETS
from common.latex import num, pm, tabular
from common.plotting import BLUES, DIVERGING, METHOD_COLORS, save, setup
from task3_ensemble.ensemble import normalise
from task3_ensemble.run_base import BASE, OUT

FIG, TAB = OUT / "figures", OUT / "tables"
NORMAL_HIST = "#b4b2aa"
NAMES = [DISPLAY_NAMES[d] for d in HIGHDIM_DATASETS]
BASE_LABEL = {"IForest": "IForest", "LOF": "LOF", "OCSVM": "OCSVM", "PCA": "PCA (rekonstr.)"}
BASE_COLORS = {"IForest": METHOD_COLORS["IForest"], "LOF": METHOD_COLORS["LOF"],
               "OCSVM": METHOD_COLORS["OCSVM"], "PCA": METHOD_COLORS["PCA"]}
MAIN_ROWS = ["IForest", "LOF", "OCSVM", "PCA", "best-single", "MV 1/4", "MV 2/4", "MV 3/4",
             "MV 4/4", "max rank/val", "Fusion mean", "Fusion AP", "Fusion skill", "DSF", "CBF"]
ROW_LABEL = {"best-single": "Najlepszy pojedynczy (wal.)", "max rank/val": "Fuzja: maksimum",
             "Fusion mean": "Fuzja: średnia", "Fusion AP": "Fuzja: wagi PR-AUC",
             "Fusion skill": "Fuzja: wagi skill", "DSF": "DSF (własna, wersja 1)",
             "CBF": "\\textbf{CBF (własna)}", "PCA": "PCA (rekonstr.)"}
SCHEMES = ["uniform", "ap", "skill", "diverse"]
NORMS = ["minmax", "zscore", "robust", "rank"]
REP_LABEL = {"Original": "oryginalna", "PCA": "PCA", "UMAP": "UMAP", "AE": "autoenkoder"}


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
    for d in HIGHDIM_DATASETS:
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
            rows.append({"Zbiór": DISPLAY_NAMES[d] if i == 0 else "", "Detektor": BASE_LABEL[b],
                         "Reprezentacja": rep, "Parametry": det,
                         "PR-AUC wal. (Zad. 2)": r.val_pr_auc_task2})
    tabular(pd.DataFrame(rows), TAB / "base_configs.tex", align="llllr")


def table_main(summ):
    lines = ["\\begin{longtable}{llrrrrrr}", "\\toprule",
             "Zbiór & Wariant & Prec. & Rec. & F1 & ROC-AUC & PR-AUC & Czas [ms] \\\\",
             "\\midrule", "\\endhead"]
    for d in HIGHDIM_DATASETS:
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
        "$\\Delta$PR-AUC": [("+" if v >= 0 else "") + num(v) for v in df.d_pr],
        "F1 (próg wal.)": [num(v) for v in df.f1_val],
        "$\\Delta$F1": [("+" if v >= 0 else "") + num(v) for v in df.d_f1_val],
        "F1 (próg $c$)": [num(v) for v in df.f1_c],
        "Wygrane (z 4)": df.wins.astype(str)})
    tabular(out, TAB / "ablation.tex", align="lllrrrrrr")
    return df


def table_cases(dec):
    rows = []
    for d in HIGHDIM_DATASETS:
        sub = dec[dec.dataset == d]
        for ens in ("MV 2/4", "MV 3/4", "Fusion mean", "DSF", "CBF"):
            a, n = sub[sub.y == 1], sub[sub.y == 0]
            rows.append({
                "Zbiór": DISPLAY_NAMES[d] if ens == "MV 2/4" else "",
                "Zespół": ROW_LABEL.get(ens, ens).replace("\\textbf{", "").replace("}", ""),
                "Anomalie": str(len(a)),
                "Uratowane": str(int(((a[ens] == 1) & (a["best-single"] == 0)).sum())),
                "Utracone": str(int(((a[ens] == 0) & (a["best-single"] == 1)).sum())),
                "FP najl. pojed.": str(int(n["best-single"].sum())),
                "FP usunięte": str(int(((n[ens] == 0) & (n["best-single"] == 1)).sum())),
                "FP dodane": str(int(((n[ens] == 1) & (n["best-single"] == 0)).sum())),
            })
    tabular(pd.DataFrame(rows), TAB / "cases.tex", align="llrrrrrr")


def table_normalisation(variants):
    """Normaliser fitted on in-sample training or out-of-sample validation scores."""
    rows = []
    label = {"minmax": "min-max", "zscore": "z-score", "robust": "odporny z", "rank": "rangi (ECDF)"}
    for norm in NORMS:
        def mean_of(name, metric="pr_auc"):
            return variants[variants.variant == name].groupby("dataset")[metric].mean().mean()
        mean_tr = mean_of(f"fusion {norm}/train/uniform/all/f1")
        mean_va = mean_of(f"fusion {norm}/val/uniform/all/f1") if norm != "rank" else mean_of("Fusion mean")
        cbf = "CBF" if norm == "robust" else f"CBF {norm}"
        rows.append({"Normalizacja": label[norm],
                     "Średnia: dopas. train": num(mean_tr), "Średnia: dopas. wal.": num(mean_va),
                     "Maksimum: train": num(mean_of(f"max {norm}/train")),
                     "Maksimum: wal.": num(mean_of(f"max {norm}/val")),
                     "CBF: PR-AUC": num(mean_of(cbf)), "CBF: F1": num(mean_of(cbf, "f1"))})
    tabular(pd.DataFrame(rows), TAB / "normalisation.tex", align="lrrrrrr")


def table_confirmation(variants, confirm):
    rows = []
    names = ["best-single", "MV 2/4", "Fusion mean", "Fusion skill", "DSF", "CBF"]
    for v in names:
        r = {"Wariant": ROW_LABEL.get(v, v).replace("\\textbf{", "").replace("}", "")}
        for label, df in (("0--9", variants), ("10--19", confirm)):
            sub = df[df.variant == v].groupby("dataset")[["pr_auc", "f1"]].mean()
            r[f"PR-AUC ({label})"] = num(sub.pr_auc.mean())
            r[f"F1 ({label})"] = num(sub.f1.mean())
            best = df[df.variant == "best-single"].groupby("dataset")[["pr_auc", "f1"]].mean()
            r[f"Lepszy niż najl. ({label})"] = str(int((sub.pr_auc > best.pr_auc + 1e-9).sum()))
        rows.append(r)
    tabular(pd.DataFrame(rows), TAB / "confirmation.tex", align="lrrrrrr")
    per_ds = []
    for label, df in (("0--9", variants), ("10--19", confirm)):
        t = df[df.variant.isin(names)].groupby(["variant", "dataset"])[["pr_auc", "f1"]].mean()
        per_ds.append(t.assign(seeds=label))
    pd.concat(per_ds).to_csv(OUT / "confirmation.csv")


# ------------------------------------------------------------------ figures

def fig_jaccard():
    jac = pd.read_csv(OUT / "jaccard.csv")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0))
    labels = ["IForest", "LOF", "OCSVM", "PCA"]
    for ax, d in zip(axes, HIGHDIM_DATASETS):
        M = (jac[jac.dataset == d].groupby(["a", "b"]).jaccard.mean().unstack()
             .loc[BASE, BASE].values)
        ax.imshow(M, cmap=BLUES, vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, num(v, 2), ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.6 else "#0b0b0b")
        ax.set_xticks(range(4), labels, rotation=40, fontsize=7.5)
        ax.set_yticks(range(4), labels if d == HIGHDIM_DATASETS[0] else [""] * 4, fontsize=7.5)
        ax.set_title(DISPLAY_NAMES[d])
        ax.grid(False)
    save(fig, FIG / "jaccard_heatmap")


def fig_voting(summ):
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.8), sharey=True)
    styles = {"precision": ("Precision", "#2a78d6", "-"), "recall": ("Recall", "#eb6834", "-"),
              "f1": ("F1", "#0b0b0b", "-")}
    for ax, d in zip(axes, HIGHDIM_DATASETS):
        sub = summ[summ.dataset == d].set_index("variant")
        m = [1, 2, 3, 4]
        for metric, (label, color, ls) in styles.items():
            ax.plot(m, [sub.loc[f"MV {i}/4", metric] for i in m], marker="o", color=color,
                    ls=ls, label=label)
            ax.plot(m, [sub.loc[f"MV {i}/4 (c)", metric] for i in m], marker="o", color=color,
                    ls="--", lw=1.2, alpha=0.7)
        ax.axhline(sub.loc["best-single", "f1"], color="#52514e", lw=0.8, ls=":")
        ax.set_xticks(m, ["1/4", "2/4", "3/4", "4/4"])
        ax.set_xlabel("wymagana liczba głosów")
        ax.set_title(DISPLAY_NAMES[d])
    axes[0].set_ylabel("wartość metryki (test)")
    axes[0].legend(fontsize=7)
    save(fig, FIG / "voting_threshold")


def fig_ablation_heatmap(abl):
    order_n = ["minmax", "zscore", "robust", "rank"]
    order_w = ["uniform", "ap", "skill", "diverse"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    for ax, sel, title in ((axes[0], "all", "wszystkie detektory"),
                           (axes[1], "greedy", "zachłanny wybór detektorów")):
        sub = abl[abl.selection == sel].set_index(["norm", "weights"])
        M = np.array([[sub.loc[(n, w), "d_pr"] for w in order_w] for n in order_n])
        lim = max(0.05, np.abs(abl.d_pr).max())
        ax.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim)
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, ("+" if v >= 0 else "") + num(v, 3), ha="center", va="center",
                    fontsize=7.5)
        ax.set_xticks(range(4), ["równe", "PR-AUC", "skill", "skill×różn."], fontsize=8)
        ax.set_yticks(range(4), ["min-max", "z-score", "odporny z", "rangi"], fontsize=8)
        ax.set_title(title)
        ax.grid(False)
    save(fig, FIG / "ablation_heatmap")


def fig_complementarity(variants):
    comp = pd.read_csv(OUT / "complementarity.csv")
    best = variants[variants.variant == "best-single"][["dataset", "seed", "pr_auc", "f1"]]
    shades = {d: BLUES(0.3 + 0.2 * i) for i, d in enumerate(HIGHDIM_DATASETS)}
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.0))
    for ax, ens, x, xlabel in ((axes[0], "Fusion mean", "mean_jaccard", "średni Jaccard detektorów"),
                               (axes[1], "Fusion mean", "unique_hits", "udział anomalii wykrytych przez 1 det."),
                               (axes[2], "MV 2/4", "false_alarms_single", "udział FP zgłoszonych przez 1 det."),
                               (axes[3], "Fusion mean", "val_quality_gap",
                                "przewaga najlepszego det. (PR-AUC wal.)")):
        e = variants[variants.variant == ens][["dataset", "seed", "pr_auc", "f1"]]
        df = comp.merge(e, on=["dataset", "seed"]).merge(best, on=["dataset", "seed"],
                                                          suffixes=("", "_best"))
        metric = "f1" if ens.startswith("MV") else "pr_auc"
        df["gain"] = df[metric] - df[f"{metric}_best"]
        for d in HIGHDIM_DATASETS:
            s = df[df.dataset == d]
            ax.scatter(s[x], s.gain, s=20, color=shades[d], label=DISPLAY_NAMES[d],
                       edgecolor="white", lw=0.4)
        r = np.corrcoef(df[x], df.gain)[0, 1]
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"zysk {'F1' if metric == 'f1' else 'PR-AUC'} ({ens})", fontsize=8)
        ax.set_title(f"$r$ = {num(r, 2)}")
    axes[0].legend(fontsize=7)
    save(fig, FIG / "complementarity_gain")


def fig_weights(variants):
    fig, axes = plt.subplots(1, 3, figsize=(13, 2.8), sharey=True)
    for ax, v, title in ((axes[0], "Fusion skill", "wagi skill"),
                         (axes[1], "DSF", "DSF: skill × różnorodność + wybór"),
                         (axes[2], "CBF", "CBF: detektory kompetentne")):
        sub = variants[variants.variant == v]
        W = np.array([np.array(json.loads(w)) for w in sub.w])
        mean_w = pd.DataFrame(W, columns=BASE).assign(dataset=sub.dataset.values).groupby(
            "dataset").mean().loc[list(HIGHDIM_DATASETS)]
        bottom = np.zeros(len(mean_w))
        for b in BASE:
            ax.bar(range(len(mean_w)), mean_w[b], 0.6, bottom=bottom, color=BASE_COLORS[b],
                   label=BASE_LABEL[b], edgecolor="white", lw=1)
            bottom += mean_w[b].values
        ax.set_xticks(range(len(mean_w)), NAMES, fontsize=8)
        ax.set_title(title)
    axes[0].set_ylabel("średnia waga (10 podziałów)")
    axes[2].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, FIG / "weights")


def fig_normalisation():
    """Why normalisers should be fitted on out-of-sample scores (one split)."""
    scores = pd.read_csv(OUT / "base_scores.csv.gz")
    sub = scores[(scores.dataset == "ionosphere") & (scores.seed == 0) & (scores.base == "OCSVM")]
    parts = {p: sub[sub.part == p].score.values for p in ("train", "val", "test")}
    y_test = sub[sub.part == "test"].y.values
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.6))
    colors = {"train": "#2a78d6", "val": "#eb6834", "test": "#0b0b0b"}
    bins = np.linspace(min(map(np.min, parts.values())), max(map(np.max, parts.values())), 30)
    for p, v in parts.items():
        axes[0].hist(v, bins=bins, histtype="step", density=True, lw=1.5, color=colors[p],
                     label={"train": "uczące (in-sample)", "val": "walidacyjne", "test": "testowe"}[p])
    axes[0].set_title("OCSVM, surowe wyniki")
    axes[0].legend(fontsize=7)
    panels = ((axes[1], "rank", parts["train"], "rangi dopasowane na train"),
              (axes[2], "rank", parts["val"], "rangi dopasowane na wal."),
              (axes[3], "robust", parts["val"], "odporny z dopasowany na wal."))
    for ax, method, fit, title in panels:
        z = normalise(fit[None], parts["test"][None], method=method)[0][0]
        ax.hist(z[y_test == 0], bins=25, color=NORMAL_HIST, alpha=0.9, label="normalne")
        ax.hist(z[y_test == 1], bins=25, color="#e34948", alpha=0.85, label="anomalie")
        ax.set_title(title)
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_yticks([])
    save(fig, FIG / "normalisation")


# --------------------------------------------------------------- synthesis

def synthesis(summ):
    """Effect of algorithm, representation and ensemble on test PR-AUC."""
    t1 = pd.read_csv(ROOT / "results" / "task1" / "best_metrics.csv")
    t1_mean = t1.groupby(["dataset", "method"]).pr_auc.mean()
    t1_cfg = pd.read_csv(ROOT / "results" / "task1" / "best_configs.csv")
    t2 = pd.read_csv(ROOT / "results" / "task2" / "selected_configs.csv")
    rows, ladder = [], []
    for d in HIGHDIM_DATASETS:
        a1 = t1_mean.loc[d]
        sel1 = t1_cfg[t1_cfg.dataset == d].set_index("method").val_pr_auc.idxmax()
        orig = t2[(t2.dataset == d) & (t2.rep == "Original") & (t2.detector != "Reconstruction")]
        reps = t2[(t2.dataset == d) & t2.detector.isin(["IForest", "LOF", "OCSVM"])]
        rep_spread = reps.groupby("detector").test_pr_auc.agg(lambda s: s.max() - s.min()).mean()
        best_t2 = t2[t2.dataset == d].loc[t2[t2.dataset == d].val_pr_auc.idxmax()]
        s = summ[summ.dataset == d].set_index("variant")
        rows.append({
            "dataset": d,
            "algo_spread_t1": a1.max() - a1.min(),
            "algo_spread_t2": orig.test_pr_auc.max() - orig.test_pr_auc.min(),
            "rep_spread": rep_spread,
            "ens_gain_cbf": s.loc["CBF", "pr_auc"] - s.loc["best-single", "pr_auc"],
            "ens_gain_mean": s.loc["Fusion mean", "pr_auc"] - s.loc["best-single", "pr_auc"],
        })
        base = orig[orig.detector == "IForest"].test_pr_auc.iloc[0]
        algo = orig.loc[orig.val_pr_auc.idxmax()]
        ladder.append({"dataset": d,
                       "IForest (oryg.)": base,
                       "Zad. 1: wybór algorytmu": a1[sel1],
                       "Zad. 2: wybór algorytmu (oryg.)": algo.test_pr_auc,
                       "Zad. 2: wybór repr. i algorytmu": best_t2.test_pr_auc,
                       "Zad. 3: najl. pojedynczy": s.loc["best-single", "pr_auc"],
                       "Zad. 3: fuzja średnia": s.loc["Fusion mean", "pr_auc"],
                       "Zad. 3: CBF": s.loc["CBF", "pr_auc"],
                       "Zad. 3: MV 2/4 (F1)": s.loc["MV 2/4", "f1"],
                       "Zad. 3: CBF (F1)": s.loc["CBF", "f1"],
                       "Zad. 3: najl. pojed. (F1)": s.loc["best-single", "f1"],
                       "algo_t2": algo.detector, "rep_t2": f"{best_t2.rep}/{best_t2.detector}",
                       "algo_t1": sel1})
    eff = pd.DataFrame(rows)
    lad = pd.DataFrame(ladder)
    eff.to_csv(OUT / "synthesis_effects.csv", index=False)
    lad.to_csv(OUT / "synthesis_ladder.csv", index=False)

    out = pd.DataFrame({
        "Zbiór": [DISPLAY_NAMES[d] for d in eff.dataset],
        "Algorytm (Zad. 1, 5 metod)": [num(v) for v in eff.algo_spread_t1],
        "Algorytm (Zad. 2, oryg.)": [num(v) for v in eff.algo_spread_t2],
        "Reprezentacja": [num(v) for v in eff.rep_spread],
        "Zespół: średnia": [("+" if v >= 0 else "") + num(v) for v in eff.ens_gain_mean],
        "Zespół: CBF": [("+" if v >= 0 else "") + num(v) for v in eff.ens_gain_cbf]})
    mean_row = {"Zbiór": "\\textit{średnio}",
                "Algorytm (Zad. 1, 5 metod)": num(eff.algo_spread_t1.mean()),
                "Algorytm (Zad. 2, oryg.)": num(eff.algo_spread_t2.mean()),
                "Reprezentacja": num(eff.rep_spread.mean()),
                "Zespół: średnia": ("+" if eff.ens_gain_mean.mean() >= 0 else "") + num(eff.ens_gain_mean.mean()),
                "Zespół: CBF": ("+" if eff.ens_gain_cbf.mean() >= 0 else "") + num(eff.ens_gain_cbf.mean())}
    out = pd.concat([out, pd.DataFrame([mean_row])])
    tabular(out, TAB / "synthesis_effects.tex", align="lrrrrr")

    cols = ["IForest (oryg.)", "Zad. 1: wybór algorytmu", "Zad. 2: wybór algorytmu (oryg.)",
            "Zad. 2: wybór repr. i algorytmu", "Zad. 3: najl. pojedynczy",
            "Zad. 3: fuzja średnia", "Zad. 3: CBF"]
    t = pd.DataFrame({"Etap": cols})
    for _, r in lad.iterrows():
        t[DISPLAY_NAMES[r.dataset]] = [num(r[c]) for c in cols]
    t["Średnio"] = [num(lad[c].mean()) for c in cols]
    tabular(t, TAB / "synthesis_ladder.tex", align="lrrrrr")

    fig, ax = plt.subplots(figsize=(11, 3.0))
    width = 0.11
    shades = [BLUES(v) for v in np.linspace(0.15, 0.95, len(cols))]
    for i, c in enumerate(cols):
        vals = list(lad[c]) + [lad[c].mean()]
        ax.bar(np.arange(5) + (i - 3) * width, vals, width * 0.92, color=shades[i], label=c)
    ax.set_xticks(range(5), NAMES + ["średnio"])
    ax.set_ylabel("PR-AUC (test)")
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.28))
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
