"""Score normalisation, voting and weighted score fusion.

Notation: ``S`` is an array (n_detectors, n_samples) of raw anomaly scores
(higher = more anomalous). Normalisers are fitted on the scores of the
training part only, weights and thresholds on the validation part only.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from common.metrics import best_f1_threshold, contamination_threshold

# ------------------------------------------------------------ normalisation


class MinMax:
    def fit(self, s):
        self.lo, self.hi = np.min(s), np.max(s)
        return self

    def transform(self, s):
        return (s - self.lo) / max(self.hi - self.lo, 1e-12)


class ZScore:
    def fit(self, s):
        self.mu, self.sd = np.mean(s), np.std(s)
        return self

    def transform(self, s):
        return (s - self.mu) / max(self.sd, 1e-12)


class RobustZ:
    """(s - median) / (1.4826 MAD): insensitive to the anomalies themselves."""

    def fit(self, s):
        self.med = np.median(s)
        self.mad = 1.4826 * np.median(np.abs(s - self.med))
        return self

    def transform(self, s):
        return (s - self.med) / max(self.mad, 1e-12)


class Rank:
    """Empirical CDF of training scores: share of training points scored lower."""

    def fit(self, s):
        self.ref = np.sort(s)
        return self

    def transform(self, s):
        return np.searchsorted(self.ref, s, side="right") / len(self.ref)


NORMALISERS = {"minmax": MinMax, "zscore": ZScore, "robust": RobustZ, "rank": Rank}


def normalise(train, *others, method="rank"):
    """Fit one normaliser per detector on its training scores."""
    out = [np.empty_like(train, float)] + [np.empty_like(o, float) for o in others]
    for k in range(len(train)):
        norm = NORMALISERS[method]().fit(train[k])
        for arr, src in zip(out, (train, *others)):
            arr[k] = norm.transform(src[k])
    return out


# ------------------------------------------------------------------- voting


def votes(decisions):
    """Number of detectors flagging each sample; decisions (n_det, n)."""
    return np.asarray(decisions).sum(axis=0)


# ------------------------------------------------------------------ weights


def skill(y_val, S_val):
    """Validation PR-AUC above the chance level (= share of anomalies)."""
    base = np.mean(y_val)
    ap = np.array([average_precision_score(y_val, s) for s in S_val])
    return np.clip(ap - base, 0, None), ap


def redundancy(S_val):
    """Mean Spearman correlation of each detector with the other detectors."""
    n = len(S_val)
    if n == 1:
        return np.zeros(1)
    rho = spearmanr(np.asarray(S_val).T).statistic
    rho = np.atleast_2d(rho)
    return (rho.sum(axis=1) - 1) / (n - 1)


def weights(y_val, S_val, scheme):
    """Weights summing to one.

    uniform  - plain mean of normalised scores,
    ap       - proportional to validation PR-AUC,
    skill    - proportional to PR-AUC above chance (useless detectors get 0),
    diverse  - skill multiplied by (1 - mean correlation with the others),
               i.e. detectors that are good *and* bring information the
               others do not have get more weight.
    """
    n = len(S_val)
    if scheme == "uniform":
        w = np.ones(n)
    elif scheme == "ap":
        w = skill(y_val, S_val)[1]
    elif scheme == "skill":
        w = skill(y_val, S_val)[0]
    elif scheme == "diverse":
        w = skill(y_val, S_val)[0] * np.clip(1 - redundancy(S_val), 0.05, None)
    else:
        raise ValueError(scheme)
    if w.sum() <= 0:
        w = np.ones(n)
    return w / w.sum()


def fuse(S, w):
    return np.asarray(w) @ np.asarray(S)


def greedy_selection(y_val, S_val, scheme="diverse", tol=1e-3):
    """Forward selection of detectors on validation PR-AUC.

    Start from the detector with the best validation PR-AUC and keep adding
    the detector that improves the fused validation PR-AUC most; stop when
    no detector improves it by more than ``tol``. Returns a boolean mask.
    """
    ap = skill(y_val, S_val)[1]
    chosen = [int(np.argmax(ap))]
    best = ap[chosen[0]]
    while len(chosen) < len(S_val):
        gains = {}
        for k in set(range(len(S_val))) - set(chosen):
            idx = chosen + [k]
            w = weights(y_val, S_val[idx], scheme)
            gains[k] = average_precision_score(y_val, fuse(S_val[idx], w))
        k, value = max(gains.items(), key=lambda kv: kv[1])
        if value <= best + tol:
            break
        chosen.append(k)
        best = value
    mask = np.zeros(len(S_val), bool)
    mask[chosen] = True
    return mask


# --------------------------------------------------------------- threshold


def threshold(rule, y_val, fused_val, fused_train, contamination=0.1):
    """Decision threshold of the fused score.

    f1  - maximises F1 on the validation part (uses validation labels),
    contamination - flags the expected share of training points (no labels).
    """
    if rule == "f1":
        return best_f1_threshold(y_val, fused_val)
    return contamination_threshold(fused_train, contamination)
