"""Task 3: scores of the four base detectors.

Base detectors use four different principles: isolation (IForest), local
density (LOF), a kernel boundary (OCSVM) and the distance to a linear
subspace (PCA reconstruction error). For every dataset each detector keeps
the representation and hyperparameters selected on validation data in task 2
(results/task2/selected_configs.csv), so no test information is used.

The script refits the detectors for every seed and stores raw scores of the
training, validation and test parts, which are then combined in many ways
by task3_ensemble/run_ensembles.py without refitting.

Outputs (results/task3/): base_configs.csv, base_scores.csv.gz, base_times.csv

With ``--confirm`` the same detectors are run on fresh splits (the next
block of seeds, see common/protocol.py) that were not used while designing
the ensembles; the outputs then go to results/task3/confirm/.
"""

import argparse
import json
import os
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import pandas as pd  # noqa: E402

from common import ROOT  # noqa: E402
from common.anomaly_types import anomaly_kind  # noqa: E402
from common.datasets import TASK3_DATASETS  # noqa: E402
from common.detectors import build as build_detector  # noqa: E402
from common.protocol import make_experiment, seeds_for  # noqa: E402
from task2_representations.representations import build as build_rep  # noqa: E402
from task2_representations.run import _warm_up_umap  # noqa: E402

OUT = ROOT / "results" / "task3"
BASE = ["IForest", "LOF", "OCSVM", "PCA"]


def output_dir(confirm):
    return OUT / "confirm" if confirm else OUT


def choose_base_configs():
    """Per dataset: best representation + parameters of each base detector."""
    sel = pd.read_csv(ROOT / "results" / "task2" / "selected_configs.csv")
    rows = []
    for d in TASK3_DATASETS:
        for name in BASE:
            if name == "PCA":
                cand = sel[(sel.dataset == d) & (sel.rep == "PCA")
                           & (sel.detector == "Reconstruction")]
            else:
                cand = sel[(sel.dataset == d) & (sel.detector == name)]
            r = cand.loc[cand.val_pr_auc.idxmax()]
            rows.append({"dataset": d, "base": name, "rep": r.rep, "rep_params": r.rep_params,
                         "detector": r.detector, "det_params": r.det_params,
                         "val_pr_auc_task2": r.val_pr_auc, "test_pr_auc_task2": r.test_pr_auc})
    return pd.DataFrame(rows)


def base_scores(cfg, exp):
    """Raw train/val/test scores and the total time (representation + detector)."""
    rep_params = json.loads(cfg.rep_params)
    if cfg.rep in ("UMAP", "AE"):
        rep_params["seed"] = exp.seed
    t0 = time.perf_counter()
    rep = build_rep(cfg.rep, **rep_params)
    Z_tr, Z_va, Z_te = rep.fit_transform_all(exp.X_train, exp.X_val, exp.X_test)
    if cfg.detector == "Reconstruction":
        train, val, test = (rep.anomaly_score(X) for X in (exp.X_train, exp.X_val, exp.X_test))
    else:
        det = build_detector(cfg.detector, **json.loads(cfg.det_params)).fit(Z_tr)
        train, val, test = det.train_scores_, det.score(Z_va), det.score(Z_te)
    return train, val, test, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true", help="use the fresh seeds")
    confirm = parser.parse_args().confirm
    out = output_dir(confirm)
    out.mkdir(parents=True, exist_ok=True)
    configs = choose_base_configs()
    configs.to_csv(out / "base_configs.csv", index=False)
    _warm_up_umap()  # keep numba compilation out of the timings
    scores, times = [], []
    for d in TASK3_DATASETS:
        kinds = anomaly_kind(d)
        for seed in seeds_for(d, confirm):
            exp = make_experiment(d, seed)
            for _, cfg in configs[configs.dataset == d].iterrows():
                tr, va, te, elapsed = base_scores(cfg, exp)
                times.append({"dataset": d, "seed": seed, "base": cfg.base, "time": elapsed})
                for part, s, idx, y in (("train", tr, exp.idx_train, exp.y_train),
                                        ("val", va, exp.idx_val, exp.y_val),
                                        ("test", te, exp.idx_test, exp.y_test)):
                    scores.append(pd.DataFrame({"dataset": d, "seed": seed, "base": cfg.base,
                                                "part": part, "row": idx, "y": y,
                                                "kind": kinds[idx], "score": s}))
            print(d, seed, flush=True)
    pd.concat(scores).to_csv(out / "base_scores.csv.gz", index=False)
    pd.DataFrame(times).to_csv(out / "base_times.csv", index=False)


if __name__ == "__main__":
    main()
