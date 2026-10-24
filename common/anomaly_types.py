"""Global / local labelling of benchmark anomalies (used only for analysis).

An anomaly is labelled *global* when its robust Mahalanobis distance to the
normal class exceeds that of 99 % of the normal observations, and *local*
otherwise (it lies inside the global envelope of the normal data and can
only be told apart by its neighbourhood). The labels of the benchmark are
not changed; this is only a description of where its anomalies lie.

The covariance of the normal class is estimated with MCD when there are at
least five normal points per feature and the covariance is well conditioned.
Otherwise MCD is not defined (Arrhythmia: 386 normal points, 274 features)
or numerically unstable (Cardio and Campaign contain linearly dependent
columns, e.g. one-hot codes) and the Ledoit-Wolf shrinkage estimate is used.
"""

from functools import lru_cache

import numpy as np
from sklearn.covariance import LedoitWolf, MinCovDet

from .datasets import load_dataset

MAX_CONDITION = 1e8


@lru_cache(maxsize=None)
def anomaly_kind(name):
    """'normal' / 'global' / 'local' label for every row of the dataset."""
    data = load_dataset(name)
    sd = data.X.std(0)
    X = (data.X - data.X.mean(0)) / np.where(sd > 0, sd, 1.0)
    normal = X[data.y == 0]
    if (len(normal) >= 5 * X.shape[1]
            and np.linalg.cond(np.cov(normal, rowvar=False)) < MAX_CONDITION):
        cov = MinCovDet(random_state=0, support_fraction=0.9).fit(normal)
    else:
        cov = LedoitWolf().fit(normal)
    dist = cov.mahalanobis(X)
    cutoff = np.quantile(dist[data.y == 0], 0.99)
    kind = np.where(dist > cutoff, "global", "local")
    return np.where(data.y == 1, kind, "normal")
