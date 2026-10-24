"""Task 3: single detectors, majority voting and weighted score fusion.

Every variant is evaluated on the test part of each of the 10 splits; all
choices (per-detector thresholds, weights, detector subsets and fused-score
thresholds) are made on the training/validation parts.

Variants
--------
single         each base detector, threshold maximising validation F1
best-single    the base detector with the highest validation PR-AUC
MV m/4         anomaly if at least m detectors flag it (per-detector
               thresholds from validation F1 or from contamination 0.1)
fusion         S(x) = sum_k w_k * norm_k(s_k(x)) for every combination of
               normalisation, weighting scheme, detector subset and
               threshold rule
max            maximum of rank-normalised scores (classical alternative)

Normalisers are fitted either on the in-sample training scores or on the
(unlabelled) validation scores; the named variants use the latter.

Two own variants are evaluated:
DSF  (diversity-weighted skill fusion) - rank normalisation fitted on
     validation scores, weights proportional to validation skill times
     (1 - redundancy), greedy forward selection, validation-F1 threshold;
CBF  (competence-based fusion) - designed after analysing why DSF fails:
     the detectors that a bootstrap of the validation part cannot tell apart
     from the best one are averaged with equal weights (robust z-score
     fitted on validation scores), clearly worse ones are dropped.

Outputs (results/task3/): variants.csv, test_decisions.csv.gz,
complementarity.csv
"""

import argparse
import json
import time
from itertools import product

import numpy as np
import pandas as pd

from common.agreement import jaccard_matrix
from common.metrics import best_f1_threshold, contamination_threshold, evaluate
from task3_ensemble import ensemble as E
from task3_ensemble.run_base import BASE, output_dir

NORMS = ("minmax", "zscore", "robust", "rank")
SCHEMES = ("uniform", "ap", "skill", "diverse")
KEY_VARIANTS = ("best-single", "MV 2/4", "MV 3/4", "Fusion mean", "DSF", "CBF")
NAMED = {("rank", "val", "uniform", "all", "f1"): "Fusion mean",
         ("rank", "val", "ap", "all", "f1"): "Fusion AP",
         ("rank", "val", "skill", "all", "f1"): "Fusion skill",
         ("rank", "val", "diverse", "greedy", "f1"): "DSF"}


def arrays(df, part):
    sub = df[df.part == part]
    wide = sub.pivot(index="row", columns="base", values="score")[BASE]
    meta = sub[sub.base == BASE[0]].set_index("row").loc[wide.index]
    return wide.values.T, meta.y.values, meta.kind.values, wide.index.values


def per_kind_recall(pred, y, kinds):
    out = {}
    for kind in ("global", "local", "subspace"):
        m = kinds == kind
        out[f"recall_{kind}"] = pred[m].mean() if m.any() else np.nan
    return out


def run_split(df, base_time):
    S_tr, _, _, _ = arrays(df, "train")
    S_va, y_va, _, _ = arrays(df, "val")
    S_te, y_te, k_te, rows_te = arrays(df, "test")
    results, decisions = [], {}

    def record(name, family, score, pred, members, extra=None, fuse_time=0.0):
        r = {"variant": name, "family": family, **evaluate(y_te, score, pred),
             **per_kind_recall(pred, y_te, k_te),
             "members": "+".join(np.array(BASE)[members]), "n_members": int(members.sum()),
             "time": float(base_time[members].sum() + fuse_time), **(extra or {})}
        results.append(r)
        if name in KEY_VARIANTS:
            decisions[name] = pred

    all_members = np.ones(len(BASE), bool)
    # single detectors
    thr_f1 = np.array([best_f1_threshold(y_va, s) for s in S_va])
    thr_c = np.array([contamination_threshold(s, 0.1) for s in S_va])
    dec_f1 = (S_te >= thr_f1[:, None]).astype(int)
    dec_c = (S_te > thr_c[:, None]).astype(int)
    val_ap = E.skill(y_va, S_va)[1]
    for k, name in enumerate(BASE):
        m = np.zeros(len(BASE), bool)
        m[k] = True
        record(name, "single", S_te[k], dec_f1[k], m, {"val_pr_auc": val_ap[k]})
    k_best = int(np.argmax(val_ap))
    m = np.zeros(len(BASE), bool)
    m[k_best] = True
    record("best-single", "single", S_te[k_best], dec_f1[k_best], m,
           {"chosen": BASE[k_best]})

    # majority voting
    for label, dec in (("f1", dec_f1), ("contamination", dec_c)):
        v = E.votes(dec)
        for need in range(1, len(BASE) + 1):
            name = f"MV {need}/4" if label == "f1" else f"MV {need}/4 (c)"
            record(name, "voting", v, (v >= need).astype(int), all_members,
                   {"votes_needed": need, "base_threshold": label})

    # weighted score fusion
    for norm, fit_on in product(NORMS, ("train", "val")):
        N_va, N_te = E.normalise(S_tr if fit_on == "train" else S_va, S_va, S_te, method=norm)
        for scheme, select, rule in product(SCHEMES, ("all", "greedy"), ("f1", "contamination")):
            t0 = time.perf_counter()
            members = (E.greedy_selection(y_va, N_va, scheme) if select == "greedy"
                       else all_members.copy())
            w = np.zeros(len(BASE))
            w[members] = E.weights(y_va, N_va[members], scheme)
            f_va, f_te = E.fuse(N_va, w), E.fuse(N_te, w)
            t = E.threshold(rule, y_va, f_va)
            fuse_time = time.perf_counter() - t0
            config = (norm, fit_on, scheme, select, rule)
            name = NAMED.get(config, "fusion " + "/".join(config))
            record(name, "fusion", f_te, (f_te >= t).astype(int), members,
                   {"norm": norm, "fit_on": fit_on, "weights": scheme, "selection": select,
                    "threshold": rule, "w": json.dumps(np.round(w, 4).tolist())}, fuse_time)
        f_va, f_te = N_va.max(axis=0), N_te.max(axis=0)
        t = best_f1_threshold(y_va, f_va)
        record(f"max {norm}/{fit_on}", "fusion", f_te, (f_te >= t).astype(int), all_members,
               {"norm": norm, "fit_on": fit_on, "weights": "max", "selection": "all",
                "threshold": "f1"})

    # competence-based fusion (the named variant uses the robust z-score:
    # unlike the ECDF it does not saturate above the range of the
    # validation scores, and median/MAD ignore the anomalies in that sample)
    members = E.competent_set(y_va, S_va, seed=int(df.seed.iloc[0]))
    w = members / members.sum()
    for norm in NORMS:
        t0 = time.perf_counter()
        N_va, N_te = E.normalise(S_va, S_va, S_te, method=norm)
        f_va, f_te = E.fuse(N_va, w), E.fuse(N_te, w)
        t = best_f1_threshold(y_va, f_va)
        record("CBF" if norm == "robust" else f"CBF {norm}", "fusion", f_te,
               (f_te >= t).astype(int), members,
               {"norm": norm, "fit_on": "val", "weights": "competence",
                "selection": "bootstrap", "threshold": "f1",
                "w": json.dumps(np.round(w, 4).tolist())}, time.perf_counter() - t0)

    # complementarity of the base detectors (analysis on the test part)
    J = jaccard_matrix(dec_f1)
    rho = np.corrcoef(np.argsort(np.argsort(S_te, axis=1), axis=1))
    hits = dec_f1[:, y_te == 1]
    comp = {"mean_jaccard": np.nanmean(J[np.triu_indices(len(BASE), 1)]),
            "mean_spearman": rho[np.triu_indices(len(BASE), 1)].mean(),
            "coverage": hits.max(axis=0).mean() if hits.size else np.nan,
            "unique_hits": (hits.sum(axis=0) == 1).mean() if hits.size else np.nan,
            "false_alarms_single": ((dec_f1[:, y_te == 0].sum(axis=0) == 1).sum()
                                    / max((dec_f1[:, y_te == 0].sum(axis=0) >= 1).sum(), 1))}
    dec = pd.DataFrame({"row": rows_te, "y": y_te, "kind": k_te,
                        **{b: dec_f1[i] for i, b in enumerate(BASE)}, **decisions})
    return results, dec, comp, J


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true", help="use fresh seeds 10-19")
    out = output_dir(parser.parse_args().confirm)
    scores = pd.read_csv(out / "base_scores.csv.gz")
    times = pd.read_csv(out / "base_times.csv")
    all_rows, all_dec, comps, jacc = [], [], [], []
    for (d, seed), df in scores.groupby(["dataset", "seed"]):
        t = times[(times.dataset == d) & (times.seed == seed)].set_index("base").loc[BASE, "time"]
        rows, dec, comp, J = run_split(df, t.values)
        all_rows += [{"dataset": d, "seed": seed, **r} for r in rows]
        all_dec.append(dec.assign(dataset=d, seed=seed))
        comps.append({"dataset": d, "seed": seed, **comp})
        jacc += [{"dataset": d, "seed": seed, "a": a, "b": b, "jaccard": J[i, j]}
                 for i, a in enumerate(BASE) for j, b in enumerate(BASE)]
    pd.DataFrame(all_rows).to_csv(out / "variants.csv", index=False)
    pd.concat(all_dec).to_csv(out / "test_decisions.csv.gz", index=False)
    pd.DataFrame(comps).to_csv(out / "complementarity.csv", index=False)
    pd.DataFrame(jacc).to_csv(out / "jaccard.csv", index=False)


if __name__ == "__main__":
    main()
