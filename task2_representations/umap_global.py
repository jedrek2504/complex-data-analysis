"""Task 2: why UMAP loses global anomalies.

For every split the original data and the UMAP variant chosen on validation
(results/task2/selected_configs.csv) are fitted on the training part. For the
test part three threshold-free numbers compare global anomalies
(common/anomaly_types.py) with normal observations; each is the chance that a
random global anomaly gets a higher value than a random normal observation
(ROC-AUC, 0.5 = no difference):
  - the score of the detector chosen on validation for the representation,
  - the mean distance to the 10 nearest training points (far from neighbours),
  - the distance to the mean of the training part (at the edge of the data).

Outputs: results/task2/umap_global.csv, tables/umap_global.tex
"""

import json
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from common.anomaly_types import anomaly_kind  # noqa: E402
from common.datasets import DISPLAY_NAMES, TASK2_DATASETS  # noqa: E402
from common.detectors import build as build_detector  # noqa: E402
from common.latex import num, tabular  # noqa: E402
from common.protocol import make_experiment, seeds_for  # noqa: E402
from task2_representations.analyze import DET_LABEL, OUT, TAB, best_per_rep  # noqa: E402
from task2_representations.representations import build as build_rep  # noqa: E402

K = 10
REPS = ["Original", "UMAP"]
MEASURES = {"det": "\\makecell[r]{ROC-AUC detektora\\\\dla anomalii globalnych}",
            "knn": "\\makecell[r]{Dalej od sąsiadów niż\\\\obserwacja normalna [\\%]}",
            "centre": "\\makecell[r]{Dalej od środka danych niż\\\\obserwacja normalna [\\%]}"}


def global_auc(Z_train, Z_test, row, is_global, is_normal):
    det = build_detector(row.detector, **json.loads(row.det_params)).fit(Z_train)
    knn = NearestNeighbors(n_neighbors=K).fit(Z_train).kneighbors(Z_test)[0].mean(axis=1)
    centre = np.linalg.norm(Z_test - Z_train.mean(axis=0), axis=1)
    keep = is_global | is_normal
    return {name: roc_auc_score(is_global[keep], values[keep])
            for name, values in (("det", det.score(Z_test)), ("knn", knn), ("centre", centre))}


def main():
    chosen = best_per_rep(pd.read_csv(OUT / "selected_configs.csv")).set_index(["dataset", "rep"])
    rows = []
    for d in TASK2_DATASETS:
        kind = anomaly_kind(d)
        for seed in seeds_for(d):
            exp = make_experiment(d, seed)
            is_global = kind[exp.idx_test] == "global"
            if not is_global.any():
                continue
            for rep in REPS:
                row = chosen.loc[(d, rep)]
                params = json.loads(row.rep_params)
                if rep == "UMAP":
                    params["seed"] = seed
                Z_train, Z_test = build_rep(rep, **params).fit_transform_all(exp.X_train,
                                                                            exp.X_test)
                rows.append({"dataset": d, "seed": seed, "rep": rep, "detector": row.detector,
                             **global_auc(Z_train, Z_test, row, is_global, exp.y_test == 0)})
            print(rows[-2:], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "umap_global.csv", index=False)
    mean = df.groupby(["dataset", "rep"]).mean(numeric_only=True)

    table = []
    for d in TASK2_DATASETS:
        o, u = mean.loc[(d, "Original")], mean.loc[(d, "UMAP")]
        dets = [DET_LABEL[chosen.loc[(d, r)].detector] for r in REPS]
        cells = {"Zbiór danych": DISPLAY_NAMES[d], "Detektor": " / ".join(dict.fromkeys(dets))}
        for m, label in MEASURES.items():
            scale, digits = (1, 2) if m == "det" else (100, 0)
            cells[label] = (f"{num(scale * o[m], digits)} $\\rightarrow$ "
                            f"{num(scale * u[m], digits)}")
        table.append(cells)
    tabular(pd.DataFrame(table), TAB / "umap_global.tex", align="llccc")
    print(mean.round(2))


if __name__ == "__main__":
    main()
