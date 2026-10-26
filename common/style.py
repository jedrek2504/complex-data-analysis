"""Figure style of the reports: large fonts, colour-blind safe colours.

Every category gets a colour from the Okabe-Ito palette together with its
own marker and line style, so that lines stay distinguishable in grey-scale
print. Figures are drawn at 7.5 in width (close to the text width), so the
fonts keep roughly their nominal size in the PDF.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

WIDTH = 7.5

# Okabe-Ito colours with a marker, line style and hatch, assigned in this order
STYLES = [
    {"color": "#0072B2", "marker": "o", "ls": "-", "hatch": ""},
    {"color": "#D55E00", "marker": "s", "ls": "--", "hatch": "//"},
    {"color": "#009E73", "marker": "^", "ls": "-.", "hatch": "\\\\"},
    {"color": "#E69F00", "marker": "D", "ls": ":", "hatch": ".."},
    {"color": "#CC79A7", "marker": "v", "ls": (0, (5, 1, 1, 1)), "hatch": "xx"},
    {"color": "#56B4E9", "marker": "P", "ls": (0, (3, 1, 1, 1, 1, 1)), "hatch": "--"},
    {"color": "#000000", "marker": "X", "ls": (0, (1, 1)), "hatch": "++"},
]

# outcome of a decision on the test part
NORMAL = "#cfcfcf"
TP_COLOR = "#0072B2"
FN_COLOR = "#D55E00"
FP_COLOR = "#E69F00"

# anomaly types (common/anomaly_types.py)
GLOBAL_COLOR = "#0072B2"
LOCAL_COLOR = "#D55E00"

# gain / loss around zero: orange - grey - blue
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#c2410c", "#eb6834", "#f6c5ac", "#f2f1ee", "#b7d3f6", "#3987e5", "#184f95"])


def styles_for(keys):
    """Fixed style per category, in the order of ``keys``."""
    return {k: STYLES[i % len(STYLES)] for i, k in enumerate(keys)}


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


def bar(ax, x, height, width, style, label, **kw):
    ax.bar(x, height, width, color=style["color"], hatch=style["hatch"],
           edgecolor="white", lw=0.5, label=label, **kw)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # no creation date: re-running a script does not change unchanged figures
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None})
    plt.close(fig)
