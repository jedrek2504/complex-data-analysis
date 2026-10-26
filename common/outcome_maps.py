"""Maps of what a detector found on the test part, in a 2-D PCA view.

The view is the plane of the first two principal components of the
(standardised) training part; the test part is projected onto it. Every
dataset gets one panel: normal points that were not flagged are light grey,
detected anomalies (TP), missed anomalies (FN) and false alarms (FP) have
their own colour and marker.
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from common.datasets import DISPLAY_NAMES
from common.style import FN_COLOR, FP_COLOR, NORMAL, TP_COLOR, WIDTH

LARGE_TEST = 2000  # test parts with more points are drawn with smaller markers
VERY_LARGE_TEST = 5000  # ... and with smaller anomaly markers


def projection(exp):
    return PCA(2, random_state=0).fit(exp.X_train)


def counts(y, pred):
    return {"TP": int(((pred == 1) & (y == 1)).sum()), "FN": int(((pred == 0) & (y == 1)).sum()),
            "FP": int(((pred == 1) & (y == 0)).sum())}


def panels(n, height=None):
    """One panel per dataset. Five datasets: 2 x 3 grid, the sixth cell holds
    the legend (returned as ``cax``); four: 2 x 2 grid, ``cax`` is None and the
    legend goes below the figure."""
    cols = 3 if n == 5 else 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(WIDTH, height or 2.35 * rows + 0.3))
    axes = axes.ravel()
    for ax in axes[:n]:
        ax.set_xticks([]), ax.set_yticks([])
    cax = None
    if len(axes) > n:
        cax = axes[n]
        cax.axis("off")
    return fig, axes[:n], cax


def outcome_points(ax, Z, y, pred):
    s = 0.6 if len(y) > LARGE_TEST else 1.0
    a = 0.5 if len(y) > VERY_LARGE_TEST else 1.0
    tn = (pred == 0) & (y == 0)
    ax.scatter(*Z[tn].T, s=10 * s, c=NORMAL, lw=0, label="normalne, nieoznaczone")
    ax.scatter(*Z[(pred == 1) & (y == 0)].T, s=30 * s, c=FP_COLOR, marker="^", ec="black",
               lw=0.4, label="fałszywy alarm (FP)")
    ax.scatter(*Z[(pred == 1) & (y == 1)].T, s=34 * a, c=TP_COLOR, marker="o", ec="black",
               lw=0.4, label="wykryta anomalia (TP)")
    ax.scatter(*Z[(pred == 0) & (y == 1)].T, s=55 * a, c=FN_COLOR, marker="X", ec="black",
               lw=0.4, label="przeoczona anomalia (FN)")


def outcome_title(ax, dataset, c, extra=""):
    ax.set_title(f"{DISPLAY_NAMES[dataset]}{extra}\nTP {c['TP']}, FN {c['FN']}, FP {c['FP']}",
                 fontsize=10.5)


def add_legend(fig, cax, ax, extra=()):
    """Legend in the free grid cell, or below the figure when there is none."""
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = list(extra) + handles, [h.get_label() for h in extra] + labels
    if cax is not None:
        cax.legend(handles, labels, loc="center", markerscale=1.3, fontsize=10.5)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=2, markerscale=1.3,
                   fontsize=10.5, bbox_to_anchor=(0.5, -0.06))
