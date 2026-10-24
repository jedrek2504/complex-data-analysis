"""Shared figure style: fixed colours per method, recessive axes, PDF output."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Colour follows the entity: every method keeps its colour in all figures.
METHOD_COLORS = {
    "IForest": "#2a78d6",
    "LOF": "#eb6834",
    "COF": "#1baf7a",
    "OCSVM": "#eda100",
    "DBSCAN": "#e87ba4",
    "PCA": "#008300",
    "AE": "#6250d6",
    "Majority": "#6250d6",
    "Fusion": "#e34948",
}
TYPE_COLORS = {"global": "#2a78d6", "local": "#eb6834", "subspace": "#1baf7a"}
TYPE_LABELS = {"global": "globalne", "local": "lokalne", "subspace": "podprzestrzenne"}
NORMAL_COLOR = "#b4b2aa"
ANOMALY_COLOR = "#e34948"
TEXT = "#0b0b0b"
MUTED = "#52514e"

BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#c2410c", "#eb6834", "#f6c5ac", "#f2f1ee", "#b7d3f6", "#3987e5", "#184f95"])


def setup():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "lines.markersize": 5,
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # no creation date: re-running a script does not change unchanged figures
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None})
    plt.close(fig)
