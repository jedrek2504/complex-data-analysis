"""Task 2: tables and figures for the report.

Selection rule (validation only): for every dataset, representation family
and detector, the pair (representation parameters, detector parameters) with
the highest mean validation PR-AUC over the 10 seeds is chosen. Sweeps over
one representation parameter (e.g. the number of PCA components) choose the
detector parameters on validation separately for every value of the swept
parameter.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT
from common.datasets import DISPLAY_NAMES, HIGHDIM_DATASETS
from common.latex import num, pm, tabular
from common.plotting import (ANOMALY_COLOR, BLUES, METHOD_COLORS, NORMAL_COLOR,
                             TYPE_COLORS, TYPE_LABELS, save, setup)

OUT = ROOT / "results" / "task2"
FIG, TAB = OUT / "figures", OUT / "tables"
REPS = ["Original", "PCA", "UMAP", "AE"]
DETECTORS = ["IForest", "LOF", "OCSVM", "Reconstruction"]
DET_LABEL = {"IForest": "IForest", "LOF": "LOF", "OCSVM": "OCSVM",
             "Reconstruction": "Rekonstrukcja"}
DET_COLORS = {**METHOD_COLORS, "Reconstruction": METHOD_COLORS["PCA"]}
REP_LABEL = {"Original": "Oryginalna", "PCA": "PCA", "UMAP": "UMAP", "AE": "Autoenkoder"}
MAIN_AE = '"hidden": [64, 32]'


def load():
    runs = pd.read_csv(OUT / "runs.csv")
    structure = pd.read_csv(OUT / "structure.csv")
    runs["main"] = (runs.rep != "AE") | runs.rep_params.str.contains(MAIN_AE, regex=False)
    structure["main"] = (structure.rep != "AE") | structure.rep_params.str.contains(
        MAIN_AE, regex=False)
    runs["total_time"] = (runs.rep_fit_time + runs.rep_transform_time
                          + runs.det_fit_time + runs.det_score_time)
    return runs, structure


def select(df, keys):
    """Best (rep_params, det_params) per group by mean validation PR-AUC.

    Returns one row per group with mean/std of test metrics of the winner.
    """
    cfg = [c for c in ("rep_params", "detector", "det_params") if c not in keys]
    means = df.groupby(keys + cfg, dropna=False).mean(numeric_only=True).reset_index()
    stds = df.groupby(keys + cfg, dropna=False).std(numeric_only=True).reset_index()
    best_idx = means.groupby(keys, dropna=False).val_pr_auc.idxmax()
    best = means.loc[best_idx].reset_index(drop=True)
    sd = stds.loc[best_idx].reset_index(drop=True)
    for c in ("test_pr_auc", "test_f1", "test_roc_auc"):
        best[c + "_sd"] = sd[c]
    return best


def describe(rep, rep_params, det, det_params):
    rp, dp = json.loads(rep_params), json.loads(det_params)
    parts = []
    if rep == "PCA":
        parts.append(f"$k$={rp['n_components']}")
    elif rep == "UMAP":
        parts.append(f"$m$={rp['n_components']}, nn={rp['n_neighbors']}, "
                     f"md={num(rp['min_dist'], 1)}")
    elif rep == "AE":
        parts.append(f"$z$={rp['latent_dim']}")
    if det == "LOF":
        parts.append(f"$k_{{LOF}}$={dp['n_neighbors']}")
    elif det == "OCSVM":
        parts.append(f"$\\nu$={num(dp['nu'], 2)}, $\\gamma$={num(dp['gamma_factor'], 1)}/$d$")
    return "; ".join(parts) if parts else "--"


# ------------------------------------------------------------------ tables

def table_configs(best):
    lines = ["\\begin{longtable}{lllp{4.2cm}rrrrrrr}", "\\toprule",
             "Zbiór & Repr. & Detektor & Konfiguracja & Prec. & Rec. & F1 & ROC-AUC & "
             "PR-AUC & $t_{repr}$ [ms] & $t_{det}$ [ms] \\\\", "\\midrule", "\\endhead"]
    for d in HIGHDIM_DATASETS:
        sub = best[best.dataset == d]
        top_pr = sub.test_pr_auc.max()
        first = True
        for rep in REPS:
            for det in DETECTORS:
                r = sub[(sub.rep == rep) & (sub.detector == det)]
                if r.empty:
                    continue
                r = r.iloc[0]
                pr = pm(r.test_pr_auc, r.test_pr_auc_sd)
                if np.isclose(r.test_pr_auc, top_pr):
                    pr = f"\\textbf{{{pr}}}"
                cells = [DISPLAY_NAMES[d] if first else "", REP_LABEL[rep], DET_LABEL[det],
                         describe(rep, r.rep_params, det, r.det_params),
                         num(r.test_precision), num(r.test_recall), num(r.test_f1),
                         num(r.test_roc_auc), pr,
                         num(1000 * (r.rep_fit_time + r.rep_transform_time), 1),
                         num(1000 * (r.det_fit_time + r.det_score_time), 1)]
                lines.append(" & ".join(cells) + " \\\\")
                first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{longtable}")
    (TAB / "configs_full.tex").write_text("\n".join(lines) + "\n")


def table_task1_comparison(best):
    t1 = pd.read_csv(ROOT / "results" / "task1" / "best_metrics.csv")
    t1 = t1.groupby(["dataset", "method"]).pr_auc.mean()
    rows = []
    for d in HIGHDIM_DATASETS:
        for det in ("IForest", "LOF", "OCSVM"):
            r = {"Zbiór": DISPLAY_NAMES[d] if det == "IForest" else "", "Detektor": det,
                 "Zad. 1": t1[(d, det)]}
            for rep in REPS:
                r[REP_LABEL[rep]] = best[(best.dataset == d) & (best.rep == rep)
                                         & (best.detector == det)].test_pr_auc.iloc[0]
            best_rep = max(REPS, key=lambda x: r[REP_LABEL[x]])
            r["Najlepsza repr."] = REP_LABEL[best_rep]
            r["Zysk"] = r[REP_LABEL[best_rep]] - r["Zad. 1"]
            rows.append(r)
    df = pd.DataFrame(rows)
    out = df.copy()
    for c in ["Zad. 1"] + [REP_LABEL[x] for x in REPS]:
        out[c] = [num(v) for v in df[c]]
    out["Zysk"] = [("+" if v >= 0 else "") + num(v) for v in df["Zysk"]]
    tabular(out, TAB / "task1_comparison.tex", align="llrrrrrlr")
    return df


def table_ae_architecture(runs, structure):
    ae = runs[(runs.rep == "AE") & runs.rep_params.str.contains('"latent_dim": 8', regex=False)]
    st = structure[(structure.rep == "AE")
                   & structure.rep_params.str.contains('"latent_dim": 8', regex=False)]
    rows = []
    for d in HIGHDIM_DATASETS:
        for hidden, label in (("[32]", "$d$-32-8"), ("[64, 32]", "$d$-64-32-8"),
                              ("[128, 64]", "$d$-128-64-8")):
            key = f'"hidden": {hidden}'
            sub = ae[(ae.dataset == d) & ae.rep_params.str.contains(key, regex=False)]
            rec = sub[sub.detector == "Reconstruction"]
            per_det = select(sub[sub.detector != "Reconstruction"], ["dataset", "detector"])
            det = per_det.loc[per_det.val_pr_auc.idxmax()]
            s = st[(st.dataset == d) & st.rep_params.str.contains(key, regex=False)]
            rows.append({"Zbiór": DISPLAY_NAMES[d] if hidden == "[32]" else "",
                         "Architektura": label,
                         "PR-AUC rekonstr.": num(rec.test_pr_auc.mean()),
                         "PR-AUC detektor w $z$": f"{num(det.test_pr_auc)} ({det.detector})",
                         "Epoki": num(s.ae_epochs.mean(), 0),
                         "Czas uczenia [s]": num(s.rep_fit_time.mean(), 2)})
    tabular(pd.DataFrame(rows), TAB / "ae_architecture.tex", align="llrrrr")


def table_types(best):
    sub = best[best.dataset == "synthetic"]
    rows = []
    for rep in REPS:
        for det in DETECTORS:
            r = sub[(sub.rep == rep) & (sub.detector == det)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({"Reprezentacja": REP_LABEL[rep] if det == "IForest" else "",
                         "Detektor": DET_LABEL[det],
                         "globalne": r.test_recall_global, "lokalne": r.test_recall_local,
                         "podprzestrzenne": r.test_recall_subspace, "PR-AUC": r.test_pr_auc})
    tabular(pd.DataFrame(rows), TAB / "recall_by_type_synthetic.tex", align="llrrrr")


# ------------------------------------------------------------------ figures

def fig_matrix(best):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
    for ax, d in zip(axes, HIGHDIM_DATASETS):
        M = np.full((4, 4), np.nan)
        for i, rep in enumerate(REPS):
            for j, det in enumerate(DETECTORS):
                r = best[(best.dataset == d) & (best.rep == rep) & (best.detector == det)]
                if not r.empty:
                    M[i, j] = r.test_pr_auc.iloc[0]
        ax.imshow(np.ma.masked_invalid(M), cmap=BLUES, vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(M):
            if not np.isnan(v):
                ax.text(j, i, num(v, 2), ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 else "#0b0b0b")
            else:
                ax.text(j, i, "n/d", ha="center", va="center", fontsize=7, color="#52514e")
        ax.set_xticks(range(4), [DET_LABEL[x] for x in DETECTORS], rotation=30, fontsize=7.5)
        ax.set_yticks(range(4), [REP_LABEL[x] for x in REPS] if d == HIGHDIM_DATASETS[0]
                      else [""] * 4)
        ax.set_title(DISPLAY_NAMES[d])
        ax.grid(False)
    save(fig, FIG / "pr_auc_matrix")


def fig_pca(runs, structure):
    pca = runs[runs.rep == "PCA"].copy()
    pca["k"] = pca.rep_params.map(lambda p: json.loads(p)["n_components"])
    orig = runs[runs.rep == "Original"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 4.8), sharex="col")
    for j, d in enumerate(HIGHDIM_DATASETS):
        ax = axes[0, j]
        for det in DETECTORS:
            sub = pca[(pca.dataset == d) & (pca.detector == det)]
            best = select(sub, ["k"]).sort_values("k")
            ax.plot(best.k, best.test_pr_auc, marker="o", color=DET_COLORS[det],
                    label=DET_LABEL[det])
            if det != "Reconstruction":
                ref = select(orig[(orig.dataset == d) & (orig.detector == det)], ["dataset"])
                ax.axhline(ref.test_pr_auc.iloc[0], color=DET_COLORS[det], lw=0.8, ls=":")
        ax.set_title(DISPLAY_NAMES[d])
        ax.set_ylim(0, 1)
        st = structure[(structure.dataset == d) & (structure.rep == "PCA")].copy()
        st["k"] = st.rep_params.map(lambda p: json.loads(p)["n_components"])
        ev = st.groupby("k").explained_variance.mean()
        axes[1, j].plot(ev.index, ev.values, marker="o", color="#256abf")
        axes[1, j].axhline(0.9, color="#52514e", lw=0.8, ls="--")
        axes[1, j].set_ylim(0, 1.02)
        axes[1, j].set_xlabel("liczba składowych $k$")
    axes[0, 0].set_ylabel("PR-AUC (test)")
    axes[1, 0].set_ylabel("zachowana wariancja")
    axes[0, 0].legend(fontsize=7)
    save(fig, FIG / "pca_components")


def fig_umap(runs):
    um = runs[runs.rep == "UMAP"].copy()
    for key in ("n_components", "n_neighbors", "min_dist"):
        um[key] = um.rep_params.map(lambda p, k=key: json.loads(p)[k])
    cols = [(n, md) for n in (5, 15, 50) for md in (0.0, 0.5)]
    fig, axes = plt.subplots(3, 4, figsize=(13, 5.6))
    for i, det in enumerate(["IForest", "LOF", "OCSVM"]):
        for j, d in enumerate(HIGHDIM_DATASETS):
            sub = um[(um.dataset == d) & (um.detector == det)]
            best = select(sub, ["n_components", "n_neighbors", "min_dist"])
            M = np.array([[best[(best.n_components == c) & (best.n_neighbors == n)
                                & (best.min_dist == md)].test_pr_auc.iloc[0]
                           for n, md in cols] for c in (2, 5, 10)])
            ax = axes[i, j]
            ax.imshow(M, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
            for (a, b), v in np.ndenumerate(M):
                ax.text(b, a, num(v, 2), ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else "#0b0b0b")
            ax.set_xticks(range(6), [f"{n}/{num(md, 1)}" for n, md in cols], fontsize=6.5)
            ax.set_yticks(range(3), [2, 5, 10])
            ax.grid(False)
            if i == 0:
                ax.set_title(DISPLAY_NAMES[d])
            if j == 0:
                ax.set_ylabel(f"{det}\nwymiar $m$")
            if i == 2:
                ax.set_xlabel("n_neighbors / min_dist")
    save(fig, FIG / "umap_params")


def fig_umap_structure(structure):
    st = structure[structure.rep == "UMAP"].copy()
    for key in ("n_components", "n_neighbors", "min_dist"):
        st[key] = st.rep_params.map(lambda p, k=key: json.loads(p)[k])
    fig, axes = plt.subplots(2, 4, figsize=(13, 4.6), sharex=True)
    shades = {2: BLUES(0.4), 5: BLUES(0.65), 10: BLUES(0.95)}
    for j, d in enumerate(HIGHDIM_DATASETS):
        sub = st[st.dataset == d]
        for c in (2, 5, 10):
            for md, ls in ((0.0, "-"), (0.5, "--")):
                g = sub[(sub.n_components == c) & (sub.min_dist == md)].groupby("n_neighbors")
                label = f"$m$={c}, md={num(md, 1)}"
                axes[0, j].plot(g.trustworthiness.mean(), ls=ls, marker="o",
                                color=shades[c], label=label)
                axes[1, j].plot(g.silhouette.mean(), ls=ls, marker="o", color=shades[c])
        orig = structure[(structure.dataset == d) & (structure.rep == "Original")].silhouette.mean()
        axes[1, j].axhline(orig, color="#52514e", lw=0.8, ls=":")
        axes[0, j].set_title(DISPLAY_NAMES[d])
        axes[1, j].set_xscale("log")
        axes[1, j].set_xticks([5, 15, 50], ["5", "15", "50"])
        axes[1, j].set_xlabel("n_neighbors")
    axes[0, 0].set_ylabel("trustworthiness ($k$=10)")
    axes[1, 0].set_ylabel("silhouette (normalne/anomalie)")
    axes[0, 0].legend(fontsize=6.5, ncol=1)
    save(fig, FIG / "umap_structure")


def fig_ae(runs):
    ae = runs[(runs.rep == "AE") & runs.main].copy()
    ae["z"] = ae.rep_params.map(lambda p: json.loads(p)["latent_dim"])
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.7), sharey=True)
    for ax, d in zip(axes, HIGHDIM_DATASETS):
        for det in DETECTORS:
            best = select(ae[(ae.dataset == d) & (ae.detector == det)], ["z"]).sort_values("z")
            ax.plot(best.z, best.test_pr_auc, marker="o", color=DET_COLORS[det],
                    label=DET_LABEL[det])
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16], ["2", "4", "8", "16"])
        ax.set_xlabel("wymiar przestrzeni latentnej")
        ax.set_title(DISPLAY_NAMES[d])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("PR-AUC (test)")
    axes[0].legend(fontsize=7)
    save(fig, FIG / "ae_latent")

    hist = pd.read_csv(OUT / "ae_history.csv")
    hist = hist[hist.rep_params.str.contains('"latent_dim": 8', regex=False)
                & hist.rep_params.str.contains(MAIN_AE, regex=False)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.5))
    for ax, d in zip(axes, HIGHDIM_DATASETS):
        h = hist[hist.dataset == d]
        ax.plot(h.epoch, h.train_loss, color="#2a78d6", lw=1.5, label="uczenie")
        ax.plot(h.epoch, h.holdout_loss, color="#eb6834", lw=1.5, label="odłożone 15\\%")
        ax.set_yscale("log")
        ax.set_xlabel("epoka")
        ax.set_title(DISPLAY_NAMES[d])
    axes[0].set_ylabel("MSE")
    axes[0].legend(fontsize=7)
    save(fig, FIG / "ae_training")


def fig_embeddings():
    emb = pd.read_csv(OUT / "embeddings_2d.csv.gz")
    emb = emb[emb.part == "test"]
    panels = [("PCA", '{"n_components": 2}', "PCA ($k$=2)"),
              ("UMAP", '{"min_dist": 0.0, "n_components": 2, "n_neighbors": 15}',
               "UMAP ($m$=2, nn=15, md=0)"),
              ("AE", '{"hidden": [64, 32], "latent_dim": 2}', "Autoenkoder ($z$=2)")]
    fig, axes = plt.subplots(4, 3, figsize=(9.5, 11.5))
    for i, d in enumerate(HIGHDIM_DATASETS):
        for j, (rep, params, title) in enumerate(panels):
            ax = axes[i, j]
            e = emb[(emb.dataset == d) & (emb.rep == rep) & (emb.rep_params == params)]
            n = e[e.y == 0]
            ax.scatter(n.z1, n.z2, s=7, c=NORMAL_COLOR, lw=0, label="normalne")
            if d == "synthetic":
                for t, color in TYPE_COLORS.items():
                    a = e[e.type == t]
                    ax.scatter(a.z1, a.z2, s=16, c=color, ec="white", lw=0.4,
                               label=f"anomalie {TYPE_LABELS[t]}")
            else:
                a = e[e.y == 1]
                ax.scatter(a.z1, a.z2, s=16, c=ANOMALY_COLOR, ec="white", lw=0.4,
                           label="anomalie")
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(DISPLAY_NAMES[d])
        if d in ("ionosphere", "synthetic"):
            axes[i, 2].legend(fontsize=6.5, loc="best")
    save(fig, FIG / "embeddings_2d")


def fig_separation(runs, structure):
    """Visual separation (silhouette) versus detection quality."""
    main = runs[runs.main & (runs.detector != "Reconstruction")]
    best = select(main, ["dataset", "rep", "rep_params"])
    st = structure.groupby(["dataset", "rep", "rep_params"]).silhouette.mean().reset_index()
    df = best.merge(st, on=["dataset", "rep", "rep_params"])
    markers = {"Original": "*", "PCA": "o", "UMAP": "s", "AE": "^"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.9), sharey=True)
    for ax, d in zip(axes, HIGHDIM_DATASETS):
        sub = df[df.dataset == d]
        for rep in REPS:
            s = sub[sub.rep == rep]
            ax.scatter(s.silhouette, s.test_pr_auc, marker=markers[rep], s=40 if rep == "Original" else 18,
                       color=BLUES(0.9) if rep == "Original" else BLUES(0.35 + 0.2 * REPS.index(rep)),
                       label=REP_LABEL[rep], edgecolor="white", lw=0.4)
        r = np.corrcoef(sub.silhouette, sub.test_pr_auc)[0, 1]
        ax.set_title(f"{DISPLAY_NAMES[d]} ($r$ = {num(r, 2)})")
        ax.set_xlabel("silhouette (normalne/anomalie)")
    axes[0].set_ylabel("PR-AUC najlepszego detektora")
    axes[0].legend(fontsize=7)
    save(fig, FIG / "separation_vs_detection")
    df.to_csv(OUT / "separation_vs_detection.csv", index=False)


def fig_time(best):
    markers = {"Original": "*", "PCA": "o", "UMAP": "s", "AE": "^"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.9), sharey=True)
    for ax, d in zip(axes, HIGHDIM_DATASETS):
        sub = best[best.dataset == d]
        for _, r in sub.iterrows():
            ax.scatter(1000 * r.total_time, r.test_pr_auc, marker=markers[r.rep],
                       s=45 if r.rep == "Original" else 26, color=DET_COLORS[r.detector],
                       edgecolor="white", lw=0.4)
        ax.set_xscale("log")
        ax.set_xlabel("czas repr. + detekcji [ms]")
        ax.set_title(DISPLAY_NAMES[d])
    handles = [plt.Line2D([], [], ls="", marker=m, color="#52514e", label=REP_LABEL[k])
               for k, m in markers.items()]
    handles += [plt.Line2D([], [], ls="", marker="o", color=DET_COLORS[k], label=DET_LABEL[k])
                for k in DETECTORS]
    fig.legend(handles=handles, ncol=8, loc="upper center", bbox_to_anchor=(0.5, 1.08),
               fontsize=7.5)
    axes[0].set_ylabel("PR-AUC (test)")
    save(fig, FIG / "time_vs_quality")


def main():
    setup()
    TAB.mkdir(parents=True, exist_ok=True)
    runs, structure = load()
    best = select(runs[runs.main], ["dataset", "rep", "detector"])
    best.to_csv(OUT / "selected_configs.csv", index=False)

    table_configs(best)
    comparison = table_task1_comparison(best)
    comparison.to_csv(OUT / "task1_comparison.csv", index=False)
    table_ae_architecture(runs, structure)
    table_types(best)

    fig_matrix(best)
    fig_pca(runs[runs.main], structure)
    fig_umap(runs)
    fig_umap_structure(structure)
    fig_ae(runs)
    fig_embeddings()
    fig_separation(runs, structure)
    fig_time(best)
    print(best.pivot_table(index=["dataset", "rep"], columns="detector",
                           values="test_pr_auc").round(3))


if __name__ == "__main__":
    main()
