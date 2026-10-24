"""Hyperparameter grids for task 1.

Parameters are split into those that change the anomaly score (a new model has
to be fitted) and ``contamination``, which only moves the decision threshold
over the training scores, so all its values reuse one fitted model.
"""

from itertools import product

CONTAMINATIONS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.35)  # 0.35: Breastw
METHODS = ("IForest", "LOF", "COF", "OCSVM", "DBSCAN")


def score_grid():
    """Yield (method, params) pairs that require fitting a model."""
    for n in (50, 100, 200, 500):
        yield "IForest", {"n_estimators": n}
    for k in (5, 10, 20, 35, 50):
        yield "LOF", {"n_neighbors": k}
        yield "COF", {"n_neighbors": k}
    for kernel, nu in product(("rbf", "poly", "sigmoid", "linear"),
                              (0.01, 0.05, 0.10, 0.20, 0.50)):
        gammas = (1.0,) if kernel == "linear" else (0.1, 1.0, 10.0)
        for g in gammas:
            yield "OCSVM", {"kernel": kernel, "nu": nu, "gamma_factor": g}
    for m, q in product((3, 5, 10, 20), (0.50, 0.75, 0.90, 0.95, 0.99)):
        yield "DBSCAN", {"min_samples": m, "eps_quantile": q}


def uses_contamination(method):
    return method in ("IForest", "LOF", "COF")
