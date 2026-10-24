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


def normalise(fit, *parts, method="rank"):
    """Fit one normaliser per detector on ``fit`` and apply it to ``parts``.

    ``fit`` holds unlabelled scores, (n_detectors, n). In-sample training
    scores are often optimistic (a kernel or neighbourhood model scores its
    own training points as very normal), so fitting on out-of-sample scores
    of the validation part makes the normalised scales comparable with the
    test scores. Labels are never used here.
    """
    out = [np.empty_like(p, float) for p in parts]
    for k in range(len(fit)):
        norm = NORMALISERS[method]().fit(fit[k])
        for arr, src in zip(out, parts):
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


def competent_set(y_val, S_val, n_boot=500, alpha=0.1, seed=0):
    """Detectors not significantly worse than the best one on validation data.

    The validation PR-AUC of every detector is re-estimated on bootstrap
    resamples of the validation part (each resample keeps at least one
    anomaly). Detector k is kept when it matches or beats the detector with
    the best validation PR-AUC in at least ``alpha`` of the resamples, i.e.
    when the data do not allow us to say that it is worse. Returns a mask.
    """
    rng = np.random.default_rng(seed)
    ap = skill(y_val, S_val)[1]
    best = int(np.argmax(ap))
    anomalies, normals = np.flatnonzero(y_val == 1), np.flatnonzero(y_val == 0)
    as_good = np.zeros(len(S_val))
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(anomalies, len(anomalies)),
                              rng.choice(normals, len(normals))])
        boot = np.array([average_precision_score(y_val[idx], s[idx]) for s in S_val])
        as_good += boot >= boot[best]
    return as_good / n_boot >= alpha


# --------------------------------------------------------------- threshold


def threshold(rule, y_val, fused_val, contamination=0.1):
    """Decision threshold of the fused score.

    f1  - maximises F1 on the validation part (uses validation labels),
    contamination - flags the expected share of validation points
                    (out-of-sample scores, no labels).
    """
    if rule == "f1":
        return best_f1_threshold(y_val, fused_val)
    return contamination_threshold(fused_val, contamination)
