"""Global / local labelling of anomalies (used only for analysis).

The synthetic dataset carries its true anomaly types. For real datasets an
anomaly is labelled *global* when its robust Mahalanobis distance to the
normal class exceeds that of 99 % of the normal observations, and *local*
otherwise (it lies inside the global envelope of the normal data and can
only be told apart by its neighbourhood).
"""

import numpy as np
from sklearn.covariance import MinCovDet

from .datasets import load_dataset


def anomaly_kind(name, seed):
    """'normal' / 'global' / 'local' (/ 'subspace') label for every row."""
    data = load_dataset(name, seed)
    if data.anomaly_type is not None:
        return data.anomaly_type
    X = (data.X - data.X.mean(0)) / data.X.std(0)
    normal = X[data.y == 0]
    mcd = MinCovDet(random_state=0, support_fraction=0.9).fit(normal)
    dist = mcd.mahalanobis(X)
    cutoff = np.quantile(dist[data.y == 0], 0.99)
    kind = np.where(dist > cutoff, "global", "local")
    return np.where(data.y == 1, kind, "normal")
