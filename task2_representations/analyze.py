"""Task 2: tables and figures for the report.

Selection rule (validation only): for every dataset, representation family
and detector, the pair (representation parameters, detector parameters) with
the highest mean validation PR-AUC over the seeds is chosen. Sweeps over one
representation parameter (e.g. the number of PCA components) choose the
remaining parameters, including the detector, on validation separately for
every value of the swept parameter. The number of seeds differs between
datasets (common/protocol.py).
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT
from common.datasets import DISPLAY_NAMES, SPECS, TASK2_DATASETS, load_dataset
from common.latex import num, pm, tabular
from common.style import (GLOBAL_COLOR, LOCAL_COLOR, NORMAL, WIDTH, line, save, setup,
                          styles_for)

OUT = ROOT / "results" / "task2"
FIG, TAB = OUT / "figures", OUT / "tables"
REPS = ["Original", "PCA", "UMAP", "AE"]
DETECTORS = ["IForest", "LOF", "OCSVM", "Reconstruction"]
DET_LABEL = {"IForest": "IForest", "LOF": "LOF", "OCSVM": "OCSVM",
             "Reconstruction": "błąd rekonstrukcji"}
REP_LABEL = {"Original": "dane oryginalne", "PCA": "PCA", "UMAP": "UMAP",
             "AE": "autoenkoder"}
DET_STYLE = {
    "IForest": {"color": "#0072B2", "marker": "o", "ls": "-", "hatch": ""},
    "LOF": {"color": "#D55E00", "marker": "s", "ls": "--", "hatch": "//"},
    "OCSVM": {"color": "#E69F00", "marker": "D", "ls": ":", "hatch": ".."},
    "Reconstruction": {"color": "#009E73", "marker": "^", "ls": "-.", "hatch": "\\\\"},
}
REP_STYLE = {
    "Original": {"color": "#000000", "marker": "*", "ls": "-", "hatch": ""},
    "PCA": {"color": "#0072B2", "marker": "o", "ls": "--", "hatch": "//"},
    "UMAP": {"color": "#CC79A7", "marker": "s", "ls": "-.", "hatch": "xx"},
    "AE": {"color": "#009E73", "marker": "^", "ls": ":", "hatch": "\\\\"},
}
DATASET_STYLE = styles_for(TASK2_DATASETS)
NAMES = [DISPLAY_NAMES[d] for d in TASK2_DATASETS]
MAIN_AE = '"hidden": [64, 32]'
SOURCE_NAMES = {"adbench": "ADBench", "odds": "ODDS", "dami": "DAMI"}
CONTENT = {
    "cardio": "zapisy KTG płodu; anomalia = zapis patologiczny",
    "arrhythmia": "zapisy EKG; anomalia = rzadki typ arytmii",
    "wdbc": "pomiary jąder komórkowych guzów; anomalia = guz złośliwy",
    "campaign": "kontakty banku z klientami; anomalia = klient założył lokatę",
}
SHOWN_FEATURES = 5


def load():
    runs = pd.read_csv(OUT / "runs.csv.gz")
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


def best_per_rep(best):
    """Per dataset and representation: the detector with the best validation PR-AUC."""
    idx = best.groupby(["dataset", "rep"]).val_pr_auc.idxmax()
    return best.loc[idx].reset_index(drop=True)


def with_param(df, key):
    out = df.copy()
    out[key] = out.rep_params.map(lambda p: json.loads(p)[key])
    return out


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


def signed(v, digits=3):
    return ("+" if v >= 0 else "") + num(v, digits)


# ------------------------------------------------------------------ tables

def table_datasets():
    rows = []
    for d in TASK2_DATASETS:
        s = SPECS[d]
        rows.append({
            "Zbiór danych": DISPLAY_NAMES[d],
            "Źródło": SOURCE_NAMES[s.source],
            "Co zawiera": CONTENT[d],
            "Obserwacje": f"{s.n:,}".replace(",", "\\,"),
            "Cechy": str(s.d),
            "Anomalie": f"{s.anomalies:,}".replace(",", "\\,")
                        + f" ({num(100 * s.anomalies / s.n, 1)}\\,\\%)",
        })
    tabular(pd.DataFrame(rows), TAB / "datasets.tex", align="llL{5.2cm}rrr")


def table_samples(d):
    """First normal and first anomalous record, the first few features."""
    data = load_dataset(d)
    rows = [np.flatnonzero(data.y == 0)[0], np.flatnonzero(data.y == 1)[0]]
    names = [f"cecha {i + 1}" for i in range(SHOWN_FEATURES)]

    def digits(j):
        return 0 if np.all(data.X[:, j] == np.round(data.X[:, j])) else 2

    table = pd.DataFrame({
        name: [num(data.X[r, j], digits(j)) for r in rows] for j, name in enumerate(names)})
    table["\\dots"] = "\\dots"
    table["Anomalia"] = ["nie" if data.y[r] == 0 else "tak" for r in rows]
    tabular(table, TAB / f"sample_{d}.tex", align="r" * (len(names) + 1) + "c")


def table_configs(best):
    lines = ["\\begin{longtable}{lllL{2.7cm}rrrrrrr}", "\\toprule",
             "Zbiór & Repr. & Detektor & Ustawienie & Prec. & Rec. & F1 & ROC-AUC & "
             "PR-AUC & \\makecell[r]{$t_{r}$\\\\{}[ms]} & \\makecell[r]{$t_{d}$\\\\{}[ms]} \\\\",
             "\\midrule", "\\endhead"]
    for d in TASK2_DATASETS:
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
                short_rep = {"Original": "oryg.", "AE": "AE"}.get(rep, rep)
                short_det = {"Reconstruction": "rekonstr."}.get(det, det)
                cells = [DISPLAY_NAMES[d] if first else "", short_rep, short_det,
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
    """PR-AUC per representation; the gain is measured against the original
    space with the same (task 2) detector grid. Task 1 results exist only for
    the datasets shared with task 1 (Cardio)."""
    t1 = pd.read_csv(ROOT / "results" / "task1" / "best_metrics.csv")
    t1 = t1.groupby(["dataset", "method"]).pr_auc.mean()
    rows = []
    for d in TASK2_DATASETS:
        for det in ("IForest", "LOF", "OCSVM"):
            r = {"Zbiór danych": DISPLAY_NAMES[d] if det == "IForest" else "",
                 "Detektor": det, "Zad. 1": t1.get((d, det), np.nan)}
            for rep in REPS:
                r[REP_LABEL[rep]] = best[(best.dataset == d) & (best.rep == rep)
                                         & (best.detector == det)].test_pr_auc.iloc[0]
            best_rep = max(REPS, key=lambda x: r[REP_LABEL[x]])
            r["Najlepsza"] = REP_LABEL[best_rep]
            r["Zysk"] = r[REP_LABEL[best_rep]] - r[REP_LABEL["Original"]]
            rows.append(r)
    df = pd.DataFrame(rows)
    out = df.copy()
    for c in ["Zad. 1"] + [REP_LABEL[x] for x in REPS]:
        out[c] = [num(v) for v in df[c]]
    out["Zysk"] = [signed(v) for v in df["Zysk"]]
    out = out.rename(columns={"dane oryginalne": "\\makecell[r]{dane\\\\oryginalne}"})
    tabular(out, TAB / "task1_comparison.tex", align="llrrrrrlr")
    return df


def table_ae_architecture(runs, structure):
    ae = runs[(runs.rep == "AE") & runs.rep_params.str.contains('"latent_dim": 8', regex=False)]
    st = structure[(structure.rep == "AE")
                   & structure.rep_params.str.contains('"latent_dim": 8', regex=False)]
    rows = []
    for d in TASK2_DATASETS:
        for hidden, label in (("[32]", "$d$-32-8"), ("[64, 32]", "$d$-64-32-8"),
                              ("[128, 64]", "$d$-128-64-8")):
            key = f'"hidden": {hidden}'
            sub = ae[(ae.dataset == d) & ae.rep_params.str.contains(key, regex=False)]
            rec = sub[sub.detector == "Reconstruction"]
            per_det = select(sub[sub.detector != "Reconstruction"], ["dataset", "detector"])
            det = per_det.loc[per_det.val_pr_auc.idxmax()]
            s = st[(st.dataset == d) & st.rep_params.str.contains(key, regex=False)]
            rows.append({
                "Zbiór danych": DISPLAY_NAMES[d] if hidden == "[32]" else "",
                "Architektura": label,
                "\\makecell[r]{PR-AUC błędu\\\\rekonstrukcji}": num(rec.test_pr_auc.mean()),
                "\\makecell[r]{PR-AUC detektora\\\\w przestrzeni $z$}":
                    f"{num(det.test_pr_auc)} ({det.detector})",
                "\\makecell[r]{Epoki\\\\uczenia}": num(s.ae_epochs.mean(), 0),
                "\\makecell[r]{Czas\\\\uczenia [s]}": num(s.rep_fit_time.mean(), 2)})
    tabular(pd.DataFrame(rows), TAB / "ae_architecture.tex", align="llrrrr")


def table_types(best):
    """Recall of global and local anomalies of the best detector (validation
    PR-AUC) in every representation."""
    chosen = best_per_rep(best)
    rows = []
    for d in TASK2_DATASETS:
        for i, rep in enumerate(REPS):
            r = chosen[(chosen.dataset == d) & (chosen.rep == rep)].iloc[0]
            rows.append({"Zbiór danych": DISPLAY_NAMES[d] if i == 0 else "",
                         "Reprezentacja": REP_LABEL[rep], "Detektor": DET_LABEL[r.detector],
                         "\\makecell[r]{Recall\\\\globalne}": r.test_recall_global,
                         "\\makecell[r]{Recall\\\\lokalne}": r.test_recall_local,
                         "PR-AUC": r.test_pr_auc})
    tabular(pd.DataFrame(rows), TAB / "recall_by_type.tex", align="lllrrr")


def table_pca_types(runs, structure):
    """LOF and the PCA reconstruction error for several numbers of components:
    PR-AUC and recall of global / local anomalies, every dataset."""
    pca = with_param(runs[runs.rep == "PCA"], "n_components")
    st = with_param(structure[structure.rep == "PCA"], "n_components")
    rows = []
    for d in TASK2_DATASETS:
        ev = st[st.dataset == d].groupby("n_components").explained_variance.mean()
        orig = select(runs[(runs.dataset == d) & (runs.rep == "Original")
                           & (runs.detector == "LOF")], ["dataset"]).iloc[0]
        lof = select(pca[(pca.dataset == d) & (pca.detector == "LOF")], ["n_components"])
        rec = select(pca[(pca.dataset == d) & (pca.detector == "Reconstruction")],
                     ["n_components"])
        lof, rec = lof.set_index("n_components"), rec.set_index("n_components")
        rows.append([DISPLAY_NAMES[d], "wszystkie", "1,00", num(orig.test_pr_auc),
                     num(orig.test_recall_global), num(orig.test_recall_local),
                     "--", "--"])
        for k in (1, 2, 5, 20):
            rows.append(["", str(k), num(ev[k], 2), num(lof.test_pr_auc[k]),
                         num(lof.test_recall_global[k]), num(lof.test_recall_local[k]),
                         num(rec.test_pr_auc[k]), num(rec.test_recall_global[k])])
    lines = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
             " & & & \\multicolumn{3}{c}{LOF} & \\multicolumn{2}{c}{błąd rekonstrukcji} \\\\",
             "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}",
             "Zbiór danych & $k$ & \\makecell[r]{zachowana\\\\wariancja} & PR-AUC & "
             "\\makecell[r]{Recall\\\\globalne} & \\makecell[r]{Recall\\\\lokalne} & PR-AUC & "
             "\\makecell[r]{Recall\\\\globalne} \\\\", "\\midrule"]
    for i, r in enumerate(rows):
        if i and r[0]:
            lines.append("\\midrule")
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "pca_types.tex").write_text("\n".join(lines) + "\n")


def param_spreads(runs):
    """Per dataset: how much the test PR-AUC of the validation-chosen setting
    changes when one representation parameter is varied (max - min over its
    values; everything else, including the detector, chosen on validation)."""
    sweeps = [("PCA", "n_components"), ("UMAP", "n_neighbors"), ("UMAP", "n_components"),
              ("UMAP", "min_dist"), ("AE", "latent_dim")]
    rows = []
    for d in TASK2_DATASETS:
        r = {"dataset": d}
        for rep, key in sweeps:
            sub = with_param(runs[(runs.dataset == d) & (runs.rep == rep)], key)
            best = select(sub, [key])
            r[f"{rep}:{key}"] = (best.test_pr_auc.max() - best.test_pr_auc.min()
                                 if len(best) > 1 else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def table_param_spread(spread):
    out = pd.DataFrame({
        "Zbiór danych": [DISPLAY_NAMES[d] for d in spread.dataset],
        "\\makecell[r]{PCA:\\\\liczba składowych $k$}": [num(v) for v in spread["PCA:n_components"]],
        "\\makecell[r]{UMAP:\\\\liczba sąsiadów nn}": [num(v) for v in spread["UMAP:n_neighbors"]],
        "\\makecell[r]{UMAP:\\\\wymiar $m$}": [num(v) for v in spread["UMAP:n_components"]],
        "\\makecell[r]{UMAP:\\\\odstęp md}": [num(v) for v in spread["UMAP:min_dist"]],
        "\\makecell[r]{Autoenkoder:\\\\wymiar $z$}": [num(v) for v in spread["AE:latent_dim"]],
    })
    tabular(out, TAB / "param_spread.tex", align="lrrrrr")


def table_structure(structure):
    """Trustworthiness of UMAP and class separation (silhouette) per representation."""
    rows = []
    for d in TASK2_DATASETS:
        st = structure[(structure.dataset == d) & structure.main]
        g = st.groupby(["rep", "rep_params"])[["trustworthiness", "silhouette"]].mean()

        def span(rep, col):
            v = g.loc[rep][col]
            return num(v.iloc[0], 2) if len(v) == 1 else f"{num(v.min(), 2)} -- {num(v.max(), 2)}"

        rows.append({"Zbiór danych": DISPLAY_NAMES[d],
                     "\\makecell[r]{Zaufanie\\\\sąsiedztw UMAP}": span("UMAP", "trustworthiness"),
                     "\\makecell[r]{Separacja:\\\\dane oryginalne}": span("Original", "silhouette"),
                     "\\makecell[r]{Separacja:\\\\PCA}": span("PCA", "silhouette"),
                     "\\makecell[r]{Separacja:\\\\UMAP}": span("UMAP", "silhouette"),
                     "\\makecell[r]{Separacja:\\\\autoenkoder}": span("AE", "silhouette")})
    tabular(pd.DataFrame(rows), TAB / "structure.tex", align="lrrrrr")


def time_summary(best):
    """Per dataset and representation: the validation-chosen detector, its test
    PR-AUC and the times of the representation and of the detector [ms]."""
    chosen = best_per_rep(best).copy()
    chosen["t_rep"] = 1000 * (chosen.rep_fit_time + chosen.rep_transform_time)
    chosen["t_det"] = 1000 * (chosen.det_fit_time + chosen.det_score_time)
    chosen["t"] = chosen.t_rep + chosen.t_det
    chosen["rank"] = chosen.groupby("dataset").test_pr_auc.rank(ascending=False)
    return chosen


def table_time(tq):
    rows = []
    for rep in REPS:
        s = tq[tq.rep == rep]
        rows.append({
            "Reprezentacja": REP_LABEL[rep],
            "\\makecell[r]{PR-AUC\\\\(średnio)}": num(s.test_pr_auc.mean()),
            "\\makecell[r]{Miejsce\\\\(średnio)}": num(s["rank"].mean(), 1),
            "\\makecell[r]{Czas reprezentacji:\\\\średnio [ms]}":
                "--" if rep == "Original" else num(s.t_rep.mean(), 0),
            "\\makecell[r]{Czas reprezentacji:\\\\od -- do [ms]}":
                "--" if rep == "Original" else f"{num(s.t_rep.min(), 0)} -- {num(s.t_rep.max(), 0)}",
            "\\makecell[r]{Czas detektora:\\\\średnio [ms]}": num(s.t_det.mean(), 0),
        })
    tabular(pd.DataFrame(rows), TAB / "time_quality.tex", align="lrrrrr")


# ------------------------------------------------------------------ figures

def grid_axes(height=4.8, sharey=True):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, height), sharey=sharey)
    return fig, axes.ravel()


def legend_above(fig, ax, ncol=4, y=1.0):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y))


def fig_matrix(best):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 6.0), constrained_layout=True)
    for ax, d in zip(axes.ravel(), TASK2_DATASETS):
        M = np.full((4, 4), np.nan)
        for i, rep in enumerate(REPS):
            for j, det in enumerate(DETECTORS):
                r = best[(best.dataset == d) & (best.rep == rep) & (best.detector == det)]
                if not r.empty:
                    M[i, j] = r.test_pr_auc.iloc[0]
        im = ax.imshow(np.ma.masked_invalid(M), cmap="Greys", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(M):
            text = "--" if np.isnan(v) else num(v, 2)
            ax.text(j, i, text, ha="center", va="center", fontsize=11,
                    color="white" if v > 0.55 else "black")
        ax.set_xticks(range(4), ["IForest", "LOF", "OCSVM", "rekonstr."], rotation=30,
                      ha="right")
        ax.set_yticks(range(4), ["oryginalne", "PCA", "UMAP", "autoenkoder"])
        ax.set_title(DISPLAY_NAMES[d])
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.6)
    cbar.set_label("PR-AUC (test)")
    save(fig, FIG / "pr_auc_matrix")


def fig_pca(runs, structure):
    pca = with_param(runs[runs.rep == "PCA"], "n_components")
    st = with_param(structure[structure.rep == "PCA"], "n_components")
    fig, axes = grid_axes()
    for ax, d in zip(axes, TASK2_DATASETS):
        for det in DETECTORS:
            best = select(pca[(pca.dataset == d) & (pca.detector == det)],
                          ["n_components"]).sort_values("n_components")
            line(ax, best.n_components, best.test_pr_auc, DET_STYLE[det], DET_LABEL[det])
        ev = st[st.dataset == d].groupby("n_components").explained_variance.mean()
        ax.plot(ev.index, ev.values, color="#777777", lw=1.6, ls=(0, (2, 2)),
                label="zachowana wariancja")
        ax.set_title(DISPLAY_NAMES[d])
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 5, 10, 15, 20])
    for ax in axes[2:]:
        ax.set_xlabel("liczba składowych $k$")
    for ax in axes[::2]:
        ax.set_ylabel("PR-AUC (test)")
    legend_above(fig, axes[0], ncol=3)
    fig.subplots_adjust(hspace=0.35)
    save(fig, FIG / "pca_components")


def fig_umap(runs):
    um = with_param(runs[runs.rep == "UMAP"], "n_neighbors")
    fig, axes = grid_axes()
    for ax, d in zip(axes, TASK2_DATASETS):
        for det in ("IForest", "LOF", "OCSVM"):
            best = select(um[(um.dataset == d) & (um.detector == det)],
                          ["n_neighbors"]).sort_values("n_neighbors")
            line(ax, best.n_neighbors, best.test_pr_auc, DET_STYLE[det], DET_LABEL[det])
        ax.set_xscale("log")
        ax.set_xticks([5, 15, 50], ["5", "15", "50"])
        ax.set_xlim(4, 60)
        ax.set_title(DISPLAY_NAMES[d])
        ax.set_ylim(0, 1.02)
    for ax in axes[2:]:
        ax.set_xlabel("liczba sąsiadów nn")
    for ax in axes[::2]:
        ax.set_ylabel("PR-AUC (test)")
    legend_above(fig, axes[0], ncol=3)
    fig.subplots_adjust(hspace=0.35)
    save(fig, FIG / "umap_params")


def fig_ae(runs):
    ae = with_param(runs[(runs.rep == "AE") & runs.main], "latent_dim")
    fig, axes = grid_axes()
    for ax, d in zip(axes, TASK2_DATASETS):
        for det in DETECTORS:
            best = select(ae[(ae.dataset == d) & (ae.detector == det)],
                          ["latent_dim"]).sort_values("latent_dim")
            line(ax, best.latent_dim, best.test_pr_auc, DET_STYLE[det], DET_LABEL[det])
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16], ["2", "4", "8", "16"])
        ax.set_title(DISPLAY_NAMES[d])
        ax.set_ylim(0, 1.02)
    for ax in axes[2:]:
        ax.set_xlabel("wymiar przestrzeni ukrytej $z$")
    for ax in axes[::2]:
        ax.set_ylabel("PR-AUC (test)")
    legend_above(fig, axes[0], ncol=4)
    fig.subplots_adjust(hspace=0.35)
    save(fig, FIG / "ae_latent")


def fig_embeddings():
    emb = pd.read_csv(OUT / "embeddings_2d.csv.gz")
    emb = emb[emb.part == "test"]
    panels = [("PCA", '{"n_components": 2}', "PCA ($k$=2)"),
              ("UMAP", '{"min_dist": 0.0, "n_components": 2, "n_neighbors": 15}',
               "UMAP ($m$=2, nn=15)"),
              ("AE", '{"hidden": [64, 32], "latent_dim": 2}', "autoenkoder ($z$=2)")]
    fig, axes = plt.subplots(4, 3, figsize=(WIDTH, 8.6))
    for i, d in enumerate(TASK2_DATASETS):
        for j, (rep, params, title) in enumerate(panels):
            ax = axes[i, j]
            e = emb[(emb.dataset == d) & (emb.rep == rep) & (emb.rep_params == params)]
            big = len(e) > 2000
            n = e[e.y == 0]
            ax.scatter(n.z1, n.z2, s=4 if big else 9, c=NORMAL, lw=0, label="normalne")
            g, loc = e[e.type == "global"], e[e.type == "local"]
            ax.scatter(g.z1, g.z2, s=10 if big else 26, c=GLOBAL_COLOR, marker="D",
                       ec="black", lw=0.3, label="anomalie globalne")
            ax.scatter(loc.z1, loc.z2, s=10 if big else 24, c=LOCAL_COLOR, marker="o",
                       ec="black", lw=0.3, label="anomalie lokalne")
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(DISPLAY_NAMES[d], fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.03),
               markerscale=1.6)
    fig.subplots_adjust(hspace=0.08, wspace=0.08)
    save(fig, FIG / "embeddings_2d")


def fig_separation(runs, structure):
    """Visual separation (silhouette) versus detection quality."""
    main = runs[runs.main & (runs.detector != "Reconstruction")]
    best = select(main, ["dataset", "rep", "rep_params"])
    st = structure.groupby(["dataset", "rep", "rep_params"]).silhouette.mean().reset_index()
    df = best.merge(st, on=["dataset", "rep", "rep_params"])
    fig, axes = grid_axes(sharey=False)
    for ax, d in zip(axes, TASK2_DATASETS):
        sub = df[df.dataset == d]
        for rep in REPS:
            s, st_ = sub[sub.rep == rep], REP_STYLE[rep]
            ax.scatter(s.silhouette, s.test_pr_auc, marker=st_["marker"],
                       s=110 if rep == "Original" else 40, color=st_["color"],
                       edgecolor="white", lw=0.4, label=REP_LABEL[rep])
        r = np.corrcoef(sub.silhouette, sub.test_pr_auc)[0, 1]
        ax.set_title(f"{DISPLAY_NAMES[d]} ($r$ = {num(r, 2)})")
    for ax in axes[2:]:
        ax.set_xlabel("separacja klas (silhouette)")
    for ax in axes[::2]:
        ax.set_ylabel("PR-AUC (test)")
    legend_above(fig, axes[0], ncol=4)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    save(fig, FIG / "separation_vs_detection")
    df.to_csv(OUT / "separation_vs_detection.csv", index=False)


def fig_time(tq):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 3.4))
    n_train = {d: SPECS[d].n // 2 for d in TASK2_DATASETS}
    order = sorted(TASK2_DATASETS, key=n_train.get)
    for rep in REPS:
        s = tq[tq.rep == rep].set_index("dataset")
        st = REP_STYLE[rep]
        a1.plot([s.t.min(), s.t.max()], [s.test_pr_auc.mean()] * 2, color=st["color"],
                lw=1.2, alpha=0.6)
        a1.scatter(s.t.mean(), s.test_pr_auc.mean(), s=110 if rep == "Original" else 70,
                   color=st["color"], marker=st["marker"], edgecolor="black", lw=0.5,
                   zorder=3)
        below = rep == "Original"
        a1.annotate(REP_LABEL[rep], (s.t.mean(), s.test_pr_auc.mean()),
                    xytext=(0, -16 if below else 8), textcoords="offset points",
                    ha="center", fontsize=10)
        if rep != "Original":  # no representation to compute
            line(a2, [n_train[d] for d in order], s.t_rep.loc[order].values, st,
                 REP_LABEL[rep])
    a1.set_xscale("log")
    a1.set_xlabel("czas reprezentacji i detekcji [ms]\n(punkt: średnio, kreska: od--do)")
    a1.set_ylabel("PR-AUC (test), średnio")
    a1.set_ylim(0.3, 0.75)
    a1.set_title("(a) jakość a czas, 4 zbiory danych")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("liczba punktów uczących")
    a2.set_ylabel("czas reprezentacji [ms]")
    a2.set_title("(b) czas a wielkość zbioru danych")
    a2.legend(fontsize=9.5, loc="upper left")
    fig.subplots_adjust(wspace=0.35)
    save(fig, FIG / "time_vs_quality")


def main():
    setup()
    TAB.mkdir(parents=True, exist_ok=True)
    runs, structure = load()
    best = select(runs[runs.main], ["dataset", "rep", "detector"])
    best.to_csv(OUT / "selected_configs.csv", index=False)

    table_datasets()
    for d in TASK2_DATASETS:
        table_samples(d)
    table_configs(best)
    comparison = table_task1_comparison(best)
    comparison.to_csv(OUT / "task1_comparison.csv", index=False)
    table_ae_architecture(runs, structure)
    table_types(best)
    table_pca_types(runs[runs.main], structure)
    spread = param_spreads(runs[runs.main])
    spread.to_csv(OUT / "param_spread.csv", index=False)
    table_param_spread(spread)
    table_structure(structure)
    tq = time_summary(best)
    tq.to_csv(OUT / "time_quality.csv", index=False)
    table_time(tq)

    fig_matrix(best)
    fig_pca(runs[runs.main], structure)
    fig_umap(runs)
    fig_ae(runs)
    fig_embeddings()
    fig_separation(runs, structure)
    fig_time(tq)
    print(best.pivot_table(index=["dataset", "rep"], columns="detector",
                           values="test_pr_auc").round(3))
    print(spread.round(3))
    print(tq[["dataset", "rep", "detector", "test_pr_auc", "t_rep", "t_det"]].round(3))


if __name__ == "__main__":
    main()
