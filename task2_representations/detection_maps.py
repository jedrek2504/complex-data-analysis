"""Task 2: what the best detector of every representation found.

For every dataset and representation the detector (with its representation
and detector parameters) with the best mean validation PR-AUC is taken from
results/task2/selected_configs.csv. It is refitted on split 0, its threshold
maximises F1 on the validation part and the test part is only drawn. All
representations are shown in the same view - the plane of the first two
principal components of the standardised training part - so that the maps
can be compared point by point.

Outputs (results/task2/): figures/detected_<rep>.pdf, tables/detected_counts.tex
"""

import json
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import outcome_maps as maps  # noqa: E402
from common.datasets import DISPLAY_NAMES, TASK2_DATASETS  # noqa: E402
from common.detectors import build as build_detector  # noqa: E402
from common.latex import tabular  # noqa: E402
from common.metrics import best_f1_threshold  # noqa: E402
from common.protocol import make_experiment  # noqa: E402
from common.style import save, setup  # noqa: E402
from task2_representations.analyze import (DET_LABEL, OUT, REP_LABEL, REPS,  # noqa: E402
                                           best_per_rep)
from task2_representations.representations import build as build_rep  # noqa: E402

SEED = 0


def predict(exp, row):
    """Refit the selected representation and detector, threshold on validation."""
    params = json.loads(row.rep_params)
    if row.rep in ("UMAP", "AE"):
        params["seed"] = SEED
    rep = build_rep(row.rep, **params)
    Z_tr, Z_va, Z_te = rep.fit_transform_all(exp.X_train, exp.X_val, exp.X_test)
    if row.detector == "Reconstruction":
        s_va, s_te = rep.anomaly_score(exp.X_val), rep.anomaly_score(exp.X_test)
    else:
        det = build_detector(row.detector, **json.loads(row.det_params)).fit(Z_tr)
        s_va, s_te = det.score(Z_va), det.score(Z_te)
    return (s_te >= best_f1_threshold(exp.y_val, s_va)).astype(int)


def main():
    setup()
    chosen = best_per_rep(pd.read_csv(OUT / "selected_configs.csv"))
    data = {}
    for d in TASK2_DATASETS:
        exp = make_experiment(d, SEED)
        data[d] = (exp, maps.projection(exp).transform(exp.X_test))
    found = {}
    for rep in REPS:
        fig, axes, cax = maps.panels(len(TASK2_DATASETS))
        for ax, d in zip(axes, TASK2_DATASETS):
            exp, Z = data[d]
            row = chosen[(chosen.dataset == d) & (chosen.rep == rep)].iloc[0]
            pred = predict(exp, row)
            c = maps.counts(exp.y_test, pred)
            maps.outcome_points(ax, Z, exp.y_test, pred)
            maps.outcome_title(ax, d, c, f" ({DET_LABEL[row.detector]})")
            found[(d, rep)] = (row.detector, c)
            print(d, rep, row.detector, c, flush=True)
        maps.add_legend(fig, cax, axes[0])
        fig.subplots_adjust(hspace=0.3)
        save(fig, OUT / "figures" / f"detected_{rep.lower()}")

    rows = []
    for d in TASK2_DATASETS:
        cells = {"Zbiór danych": DISPLAY_NAMES[d],
                 "\\makecell[r]{Anomalie\\\\w teście}": str(int(data[d][0].y_test.sum()))}
        for rep in REPS:
            det, c = found[(d, rep)]
            cells[REP_LABEL[rep]] = (f"\\makecell{{{c['TP']} / {c['FN']} / {c['FP']}\\\\"
                                     f"\\footnotesize({DET_LABEL[det]})}}")
        rows.append(cells)
    tabular(pd.DataFrame(rows), OUT / "tables" / "detected_counts.tex",
            align="lr" + "c" * len(REPS))


if __name__ == "__main__":
    main()
