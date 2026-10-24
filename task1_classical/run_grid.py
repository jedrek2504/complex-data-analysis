"""Task 1: hyperparameter grid for five classical detectors.

For every dataset, seed and configuration the detector is fitted on the
training part and evaluated on the validation and test parts with
(a) its own decision rule (contamination / nu / eps) and
(b) the threshold maximising F1 on the validation part.

Output: results/task1/grid.csv
"""

import json
import os
import time

import pandas as pd
from joblib import Parallel, delayed

from common import ROOT
from common.datasets import TASK1_DATASETS
from common.detectors import build
from common.metrics import best_f1_threshold, evaluate
from common.protocol import make_experiment, seeds_for
from task1_classical.config import CONTAMINATIONS, score_grid, uses_contamination

OUT = ROOT / "results" / "task1"
N_JOBS = int(os.environ.get("N_JOBS", 3))


def run_config(exp, method, params):
    t0 = time.perf_counter()
    det = build(method, **params).fit(exp.X_train)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    s_test = det.score(exp.X_test)
    score_time = time.perf_counter() - t0
    s_val = det.score(exp.X_val)
    t_val = best_f1_threshold(exp.y_val, s_val)
    tuned = evaluate(exp.y_test, s_test, (s_test >= t_val).astype(int))

    variants = [{}]
    if uses_contamination(method):
        variants = [{"contamination": c} for c in CONTAMINATIONS]
    rows = []
    for extra in variants:
        for key, value in extra.items():
            setattr(det, key, value)
        row = {"dataset": exp.dataset, "seed": exp.seed, "method": method,
               "params": json.dumps(params, sort_keys=True), **params, **extra,
               "fit_time": fit_time, "score_time": score_time}
        if method == "DBSCAN":
            row.update(eps=det.eps, n_clusters=det.n_clusters_)
        for part, X, y, s in (("val", exp.X_val, exp.y_val, s_val),
                              ("test", exp.X_test, exp.y_test, s_test)):
            for k, v in evaluate(y, s, det.predict(X, s)).items():
                row[f"{part}_{k}"] = v
        row.update({f"tuned_{k}": v for k, v in tuned.items()})
        rows.append(row)
    return rows


def run_job(dataset, seed):
    exp = make_experiment(dataset, seed)
    rows = []
    for method, params in score_grid():
        rows.extend(run_config(exp, method, params))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(d, s) for d in TASK1_DATASETS for s in seeds_for(d)]
    results = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_job)(d, s) for d, s in jobs)
    pd.DataFrame([r for rows in results for r in rows]).to_csv(OUT / "grid.csv", index=False)


if __name__ == "__main__":
    main()
