"""Anomaly detectors with a common interface.

Every detector is fitted on (unlabelled) training data and exposes:

* ``score(X)`` – anomaly score, higher = more anomalous,
* ``train_scores_`` – scores of the training points (in-sample),
* ``predict(X)`` – the detector's own binary decision (1 = anomaly), i.e. the
  decision implied by its parameters (``contamination``, ``nu`` or ``eps``).
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

from .cof import COF
from .metrics import contamination_threshold


class Detector:
    contamination = 0.1

    def fit(self, X):
        raise NotImplementedError

    def score(self, X):
        raise NotImplementedError

    def threshold(self):
        return contamination_threshold(self.train_scores_, self.contamination)

    def predict(self, X, scores=None):
        scores = self.score(X) if scores is None else scores
        return (scores > self.threshold()).astype(int)


class IForestDetector(Detector):
    def __init__(self, n_estimators=100, contamination=0.1, max_samples="auto",
                 random_state=0):
        self.contamination = contamination
        self.model = IsolationForest(n_estimators=n_estimators,
                                     max_samples=max_samples,
                                     random_state=random_state)

    def fit(self, X):
        self.model.fit(X)
        self.train_scores_ = self.score(X)
        return self

    def score(self, X):
        return -self.model.score_samples(X)


class LOFDetector(Detector):
    def __init__(self, n_neighbors=20, contamination=0.1):
        self.contamination = contamination
        self.n_neighbors = n_neighbors

    def fit(self, X):
        k = min(self.n_neighbors, len(X) - 1)
        self.model = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(X)
        self.train_scores_ = -self.model.negative_outlier_factor_
        return self

    def score(self, X):
        return -self.model.score_samples(X)


class COFDetector(Detector):
    def __init__(self, n_neighbors=20, contamination=0.1):
        self.contamination = contamination
        self.model = COF(n_neighbors=n_neighbors)

    def fit(self, X):
        self.model.fit(X)
        self.train_scores_ = self.model.train_scores_
        return self

    def score(self, X):
        return self.model.score_samples(X)


class OCSVMDetector(Detector):
    """One-Class SVM; gamma is given relative to 1 / n_features."""

    def __init__(self, kernel="rbf", nu=0.1, gamma_factor=1.0, degree=3):
        self.kernel, self.nu = kernel, nu
        self.gamma_factor, self.degree = gamma_factor, degree

    def fit(self, X):
        gamma = self.gamma_factor / X.shape[1]
        self.model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=gamma,
                                 degree=self.degree, coef0=0.0).fit(X)
        self.train_scores_ = self.score(X)
        return self

    def score(self, X):
        return -self.model.decision_function(X)

    def threshold(self):
        return 0.0  # the SVM boundary itself, controlled by nu


class DBSCANDetector(Detector):
    """DBSCAN turned into a detector that can score new data.

    ``eps`` is set as a quantile of the distance to the ``min_samples``-th
    neighbour in the training data (the k-distance plot heuristic), so one
    grid works for differently scaled datasets. The score of a point is its
    distance to the nearest core point; a point is an anomaly (noise) when
    this distance exceeds ``eps``.
    """

    def __init__(self, min_samples=5, eps_quantile=0.9, eps=None):
        self.min_samples, self.eps_quantile, self.eps = min_samples, eps_quantile, eps

    def fit(self, X):
        if self.eps is None:
            nn = NearestNeighbors(n_neighbors=self.min_samples).fit(X)
            kdist = nn.kneighbors(X)[0][:, -1]
            self.eps = float(np.quantile(kdist, self.eps_quantile))
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        cores = X[self.model.core_sample_indices_]
        if len(cores) == 0:  # everything is noise: fall back to all points
            cores = X
        self.core_nn_ = NearestNeighbors(n_neighbors=1).fit(cores)
        self.n_clusters_ = len(set(self.model.labels_) - {-1})
        self.train_scores_ = self.score(X)
        return self

    def score(self, X):
        return self.core_nn_.kneighbors(X)[0][:, 0]

    def threshold(self):
        return self.eps


class PCADetector(Detector):
    """Reconstruction error after projecting onto the leading components."""

    def __init__(self, n_components=0.9, contamination=0.1):
        self.n_components, self.contamination = n_components, contamination

    def fit(self, X):
        self.model = PCA(n_components=self.n_components, random_state=0).fit(X)
        self.train_scores_ = self.score(X)
        return self

    def score(self, X):
        recon = self.model.inverse_transform(self.model.transform(X))
        return np.sum((X - recon) ** 2, axis=1)


DETECTORS = {
    "IForest": IForestDetector,
    "LOF": LOFDetector,
    "COF": COFDetector,
    "OCSVM": OCSVMDetector,
    "DBSCAN": DBSCANDetector,
    "PCA": PCADetector,
}


def build(method, **params):
    return DETECTORS[method](**params)
