"""Task 3: what the best single detector and the ensembles found.

Decisions on the test part of split 0 are taken from run_ensembles
(results/task3/test_decisions.csv.gz); every threshold, weight and subset of
detectors was chosen on the training/validation parts. All variants are
shown in the same view - the plane of the first two principal components of
the standardised training part - so that the maps can be compared point by
point. The panel title names the detectors used in split 0.

Outputs (results/task3/): figures/detected_<variant>.pdf, tables/detected_counts.tex
"""

import pandas as pd

from common import outcome_maps as maps
from common.datasets import DISPLAY_NAMES, TASK3_DATASETS
from common.latex import tabular
from common.protocol import make_experiment
from common.style import save, setup
from task3_ensemble.analyze import ROW_LABEL
from task3_ensemble.run_base import OUT

SEED = 0
VARIANTS = {"best-single": "best_single", "MV 2/4": "mv2", "Fusion mean": "fusion_mean",
            "DSF": "dsf", "CBF": "cbf"}


def main():
    setup()
    dec = pd.read_csv(OUT / "test_decisions.csv.gz")
    dec = dec[dec.seed == SEED]
    variants = pd.read_csv(OUT / "variants.csv")
    members = variants[variants.seed == SEED].set_index(["dataset", "variant"]).members
    data = {}
    for d in TASK3_DATASETS:
        exp = make_experiment(d, SEED)
        p = dec[dec.dataset == d].set_index("row").loc[exp.idx_test]
        assert (p.y.values == exp.y_test).all()
        data[d] = (exp, maps.projection(exp).transform(exp.X_test), p)
    found = {}
    for v, name in VARIANTS.items():
        fig, axes, cax = maps.panels(len(TASK3_DATASETS))
        for ax, d in zip(axes, TASK3_DATASETS):
            exp, Z, p = data[d]
            pred = p[v].values
            c = maps.counts(exp.y_test, pred)
            maps.outcome_points(ax, Z, exp.y_test, pred)
            used = "" if v.startswith("MV") or v == "Fusion mean" else members[(d, v)]
            maps.outcome_title(ax, d, c, f" ({used.replace('+', ', ')})" if used else "")
            found[(d, v)] = c
        maps.add_legend(fig, cax, axes[0])
        fig.subplots_adjust(hspace=0.3)
        save(fig, OUT / "figures" / f"detected_{name}")

    rows = []
    for d in TASK3_DATASETS:
        cells = {"Zbiór danych": DISPLAY_NAMES[d],
                 "\\makecell[r]{Anomalie\\\\w teście}": str(int(data[d][0].y_test.sum()))}
        for v in VARIANTS:
            c = found[(d, v)]
            cells[ROW_LABEL.get(v, v)] = f"{c['TP']} / {c['FN']} / {c['FP']}"
        rows.append(cells)
    tabular(pd.DataFrame(rows), OUT / "tables" / "detected_counts.tex",
            align="lr" + "c" * len(VARIANTS))
    print(pd.DataFrame(found))


if __name__ == "__main__":
    main()
