"""Task 1: evaluation of the best configuration of every method.

For each dataset and method the configuration with the highest mean
validation PR-AUC (over all seeds) is selected from results/task1/grid.csv.
It is refitted for every seed; the decision threshold is the one maximising
F1 on the validation part. Per-sample test scores and decisions are stored
for the agreement and local/global analyses (see common/anomaly_types.py).

Outputs (results/task1/): best_configs.csv, best_metrics.csv,
test_predictions.csv.gz
"""

import json
import time

import pandas as pd

from common.anomaly_types import anomaly_kind
from common.datasets import TASK1_DATASETS
from common.detectors import build
from common.metrics import best_f1_threshold, evaluate
from common.protocol import SEEDS, make_experiment
from task1_classical.config import METHODS
from task1_classical.run_grid import OUT


def select_configs(grid):
    """Best score-defining parameters per dataset and method (validation PR-AUC)."""
    score_level = grid.drop_duplicates(["dataset", "seed", "method", "params"])
    mean = (score_level.groupby(["dataset", "method", "params"])["val_pr_auc"]
            .mean().reset_index())
    best = mean.loc[mean.groupby(["dataset", "method"])["val_pr_auc"].idxmax()]
    return best.reset_index(drop=True)


def main():
    grid = pd.read_csv(OUT / "grid.csv")
    best = select_configs(grid)
    best.to_csv(OUT / "best_configs.csv", index=False)

    metrics, predictions = [], []
    for dataset in TASK1_DATASETS:
        for seed in SEEDS:
            exp = make_experiment(dataset, seed)
            kinds = anomaly_kind(dataset, seed)[exp.idx_test]
            for method in METHODS:
                row = best[(best.dataset == dataset) & (best.method == method)].iloc[0]
                params = json.loads(row.params)
                t0 = time.perf_counter()
                det = build(method, **params).fit(exp.X_train)
                fit_time = time.perf_counter() - t0
                t0 = time.perf_counter()
                s_test = det.score(exp.X_test)
                score_time = time.perf_counter() - t0
                threshold = best_f1_threshold(exp.y_val, det.score(exp.X_val))
                pred = (s_test >= threshold).astype(int)
                metrics.append({"dataset": dataset, "seed": seed, "method": method,
                                "params": row.params, "fit_time": fit_time,
                                "score_time": score_time,
                                **evaluate(exp.y_test, s_test, pred)})
                predictions.append(pd.DataFrame({
                    "dataset": dataset, "seed": seed, "method": method,
                    "row": exp.idx_test, "y": exp.y_test, "kind": kinds,
                    "score": s_test, "pred": pred}))
    pd.DataFrame(metrics).to_csv(OUT / "best_metrics.csv", index=False)
    pd.concat(predictions).to_csv(OUT / "test_predictions.csv.gz", index=False)
    print(pd.DataFrame(metrics).groupby(["dataset", "method"])[["pr_auc", "f1"]]
          .mean().round(3).unstack())


if __name__ == "__main__":
    main()
