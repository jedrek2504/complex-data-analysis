"""Connectivity-based Outlier Factor (Tang et al., 2002).

Implemented from scratch so that the model fitted on training data can score
new observations (novelty setting), like LOF with ``novelty=True``.

For a point p with k nearest neighbours N_k(p), the set-based nearest path
adds neighbours one by one, always choosing the point closest to the set built
so far (Prim's order); e_1..e_k are the lengths of the added edges. The
average chaining distance weights early edges more heavily:

    ac_dist(p) = sum_i 2 (k + 1 - i) / (k (k + 1)) * e_i

and COF(p) = ac_dist(p) / mean_{o in N_k(p)} ac_dist(o). Values clearly
above 1 mean that p is connected to its neighbourhood more loosely than the
neighbours are connected to theirs.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _chaining_distance(points):
    """Average chaining distance of points[:, 0] w.r.t. points[:, 1:].

    points: array (n, k + 1, d); returns array (n,).
    """
    n, m, _ = points.shape
    k = m - 1
    dist = np.linalg.norm(points[:, :, None, :] - points[:, None, :, :], axis=-1)
    in_set = np.zeros((n, m), bool)
    in_set[:, 0] = True
    reach = dist[:, 0].copy()  # distance of each candidate to the current set
    weights = 2 * (k + 1 - np.arange(1, k + 1)) / (k * (k + 1))
    rows = np.arange(n)
    ac = np.zeros(n)
    for i in range(k):
        candidate = np.where(in_set, np.inf, reach)
        j = candidate.argmin(axis=1)
        ac += weights[i] * candidate[rows, j]
        in_set[rows, j] = True
        reach = np.minimum(reach, dist[rows, j])
    return ac


class COF:
    def __init__(self, n_neighbors=20, batch_size=512):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size

    def fit(self, X):
        self.X_ = np.asarray(X, float)
        k = min(self.n_neighbors, len(self.X_) - 1)
        self.k_ = k
        self.nn_ = NearestNeighbors(n_neighbors=k + 1).fit(self.X_)
        _, idx = self.nn_.kneighbors(self.X_)
        idx = idx[:, 1:]  # drop the point itself
        self.ac_ = self._ac_dist(self.X_, idx)
        self.train_scores_ = self.ac_ / np.maximum(self.ac_[idx].mean(axis=1), 1e-12)
        return self

    def _ac_dist(self, X, idx):
        out = np.empty(len(X))
        for start in range(0, len(X), self.batch_size):
            sl = slice(start, start + self.batch_size)
            pts = np.concatenate([X[sl, None, :], self.X_[idx[sl]]], axis=1)
            out[sl] = _chaining_distance(pts)
        return out

    def score_samples(self, X):
        """COF of new points with respect to the training data."""
        X = np.asarray(X, float)
        _, idx = self.nn_.kneighbors(X, n_neighbors=self.k_)
        ac = self._ac_dist(X, idx)
        return ac / np.maximum(self.ac_[idx].mean(axis=1), 1e-12)
