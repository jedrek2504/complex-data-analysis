"""Figure style of the task 1 report: large fonts, colour-blind safe colours.

Every category gets a colour from the Okabe-Ito palette together with its
own marker and line style, so that lines stay distinguishable in grey-scale
print. Figures are drawn at 7.5 in width (close to the text width), so the
fonts keep roughly their nominal size in the PDF.
"""

import matplotlib.pyplot as plt

from common.datasets import DISPLAY_NAMES, TASK1_DATASETS
from common.plotting import save  # noqa: F401  (re-exported for the scripts)

WIDTH = 7.5

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

# outcome of a decision on the test part
NORMAL = "#cfcfcf"
TP_COLOR = "#0072B2"
FN_COLOR = "#D55E00"
FP_COLOR = "#E69F00"

NAMES = [DISPLAY_NAMES[d] for d in TASK1_DATASETS]


def setup():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
        "axes.edgecolor": "#8a8a8a",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e3e3e3",
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 7,
    })


def line(ax, x, y, style, label):
    ax.plot(x, y, color=style["color"], marker=style["marker"], ls=style["ls"],
            label=label)
