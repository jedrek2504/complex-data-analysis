"""Figure style of the task 1 report: fixed style per method and dataset.

The general settings (fonts, palette, outcome colours) are in common/style.py.
"""

from common.datasets import DISPLAY_NAMES, TASK1_DATASETS
from common.style import (FN_COLOR, FP_COLOR, NORMAL, TP_COLOR, WIDTH,  # noqa: F401
                          line, save, setup)

METHOD_STYLE = {
    "IForest": {"color": "#0072B2", "marker": "o", "ls": "-", "hatch": ""},
    "LOF": {"color": "#D55E00", "marker": "s", "ls": "--", "hatch": "//"},
    "COF": {"color": "#009E73", "marker": "^", "ls": "-.", "hatch": "\\\\"},
    "OCSVM": {"color": "#E69F00", "marker": "D", "ls": ":", "hatch": ".."},
    "DBSCAN": {"color": "#CC79A7", "marker": "v", "ls": (0, (5, 1, 1, 1)), "hatch": "xx"},
}

DATASET_STYLE = {
    "breastw": {"color": "#000000", "marker": "o", "ls": "-"},
    "mammography": {"color": "#56B4E9", "marker": "s", "ls": "--"},
    "wine": {"color": "#E69F00", "marker": "^", "ls": "-."},
    "heart": {"color": "#CC79A7", "marker": "D", "ls": ":"},
    "cardio": {"color": "#009E73", "marker": "v", "ls": (0, (5, 1, 1, 1))},
}

NAMES = [DISPLAY_NAMES[d] for d in TASK1_DATASETS]
