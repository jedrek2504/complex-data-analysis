"""Agreement between sets of observations flagged by different detectors."""

import numpy as np


def jaccard(a, b):
    """Jaccard similarity of two binary indicator vectors (nan if both empty)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = np.sum(a | b)
    return np.sum(a & b) / union if union else np.nan


def jaccard_matrix(preds):
    """Pairwise Jaccard similarities; preds has shape (n_detectors, n_samples)."""
    n = len(preds)
    M = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = jaccard(preds[i], preds[j])
    return M
