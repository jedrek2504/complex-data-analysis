"""Experimental protocol shared by all tasks.

For every dataset and seed the data are split (stratified) into
train / validation / test = 50 / 25 / 25 %. Detectors are fitted on the
training part without labels (unsupervised setting, the training part
contains anomalies). Labels of the validation part are used to choose
hyperparameters, decision thresholds and ensemble weights; the test part is
used only for the final evaluation. The same seeds give identical splits in
all three tasks.

Number of splits per dataset
----------------------------
* 10 seeds (0-9) by default;
* 30 seeds for datasets with at most 10 anomalies (WDBC, Wine, Statlog Heart):
  a validation or test part then holds only 1-3 anomalies, so one split says
  little and the mean over more splits is needed;
* 5 seeds for Campaign (41 188 rows): one task-2 job on this dataset takes
  about as long as all splits of the other datasets together.

The stratified split guarantees at least one anomaly in every part; this is
also checked explicitly.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .datasets import SPECS, load_dataset

SPLIT = (0.50, 0.25, 0.25)
N_SEEDS_DEFAULT = 10
N_SEEDS_SMALL = 30  # datasets with few anomalies
N_SEEDS_LARGE = 5  # campaign
SMALL_ANOMALY_COUNT = 10


def n_seeds(name):
    if SPECS[name].anomalies <= SMALL_ANOMALY_COUNT:
        return N_SEEDS_SMALL
    if name == "campaign":
        return N_SEEDS_LARGE
    return N_SEEDS_DEFAULT


def seeds_for(name, confirm=False):
    """Seeds used for design; with ``confirm`` the next, unused block."""
    n = n_seeds(name)
    return tuple(range(n, 2 * n)) if confirm else tuple(range(n))


@dataclass
class Experiment:
    dataset: str
    seed: int
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray  # row indices into the benchmark file
    idx_val: np.ndarray
    idx_test: np.ndarray


def split_indices(y, seed):
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, train_size=SPLIT[0], stratify=y,
                                   random_state=seed)
    val_share = SPLIT[1] / (SPLIT[1] + SPLIT[2])
    val, test = train_test_split(rest, train_size=val_share, stratify=y[rest],
                                 random_state=seed)
    for part in (train, val, test):
        assert y[part].sum() >= 1, "every part must contain an anomaly"
    return np.sort(train), np.sort(val), np.sort(test)


def make_experiment(name, seed, scale=True):
    """Load, split and standardise (scaler fitted on the training part only)."""
    data = load_dataset(name)
    tr, va, te = split_indices(data.y, seed)
    X_tr, X_va, X_te = data.X[tr], data.X[va], data.X[te]
    if scale:
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_va, X_te = (scaler.transform(a) for a in (X_tr, X_va, X_te))
    return Experiment(name, seed, X_tr, X_va, X_te, data.y[tr], data.y[va],
                      data.y[te], tr, va, te)
