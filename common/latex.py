"""Helpers writing result tables as LaTeX (booktabs) fragments for the reports."""

import numpy as np


def num(x, digits=3):
    """Format a number with a decimal comma (Polish convention)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{digits}f}".replace(".", ",")


def pm(mean, std, digits=3):
    return f"{num(mean, digits)} $\\pm$ {num(std, digits)}"


def tabular(df, path, align=None, bold_max=None, digits=3):
    """Write ``df`` as a tabular environment.

    ``bold_max`` lists numeric columns in which the best value per table is
    set in bold. Cells that are strings are written unchanged.
    """
    cols = list(df.columns)
    align = align or "l" + "r" * (len(cols) - 1)
    best = {c: df[c].max() for c in (bold_max or []) if c in df}
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(cols) + " \\\\", "\\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cell = num(v, digits)
                if c in best and np.isclose(v, best[c]):
                    cell = f"\\textbf{{{cell}}}"
            else:
                cell = str(v)
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
