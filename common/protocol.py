"""Experimental protocol shared by all tasks.

For every dataset and seed the data are split (stratified) into
train / validation / test = 50 / 25 / 25 %. Detectors are fitted on the
training part without labels (unsupervised setting, the training part
contains anomalies). Labels of the validation part are used to choose
hyperparameters, decision thresholds and ensemble weights; the test part is
used only for the final evaluation. The same seeds give identical splits in
all three tasks.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .datasets import load_dataset

SEEDS = tuple(range(10))
SPLIT = (0.50, 0.25, 0.25)


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
    idx_train: np.ndarray  # row indices into the (subsampled) dataset
    idx_val: np.ndarray
    idx_test: np.ndarray
    types_test: np.ndarray = None  # anomaly type of test rows (synthetic only)


def split_indices(y, seed):
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, train_size=SPLIT[0], stratify=y,
                                   random_state=seed)
    val_share = SPLIT[1] / (SPLIT[1] + SPLIT[2])
    val, test = train_test_split(rest, train_size=val_share, stratify=y[rest],
                                 random_state=seed)
    return np.sort(train), np.sort(val), np.sort(test)


def make_experiment(name, seed, scale=True):
    """Load, split and standardise (scaler fitted on the training part only)."""
    data = load_dataset(name, seed)
    tr, va, te = split_indices(data.y, seed)
    X_tr, X_va, X_te = data.X[tr], data.X[va], data.X[te]
    if scale:
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_va, X_te = (scaler.transform(a) for a in (X_tr, X_va, X_te))
    types = data.anomaly_type[te] if data.anomaly_type is not None else None
    return Experiment(name, seed, X_tr, X_va, X_te, data.y[tr], data.y[va],
                      data.y[te], tr, va, te, types)
