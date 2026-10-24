"""Dataset loading and framing as anomaly-detection problems.

Real datasets are classification problems; the majority class is treated as
normal and the minority class as anomalous. When the minority class exceeds
``TARGET_CONTAMINATION`` of the data it is randomly subsampled (with the run
seed), so each seed sees a different subset of anomalies.

The synthetic dataset gives full control over the anomaly type:
``global`` (far from all data), ``local`` (close to a dense cluster but
outside its scale) and ``subspace`` (off the low-dimensional manifold on
which normal data lives).
"""

from dataclasses import dataclass, field
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from . import DATA_DIR

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
TARGET_CONTAMINATION = 0.10
SYNTHETIC_SEED = 2026

# Datasets used in the tasks (Glass has fewer than 20 features: task 1 only).
TASK1_DATASETS = ("breast_cancer", "ionosphere", "parkinsons", "glass", "synthetic")
HIGHDIM_DATASETS = ("breast_cancer", "ionosphere", "parkinsons", "synthetic")

DISPLAY_NAMES = {
    "breast_cancer": "Breast Cancer",
    "ionosphere": "Ionosphere",
    "parkinsons": "Parkinsons",
    "glass": "Glass",
    "synthetic": "Synthetic",
}


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray  # 1 = anomaly, 0 = normal
    feature_names: list
    anomaly_type: np.ndarray = field(default=None)  # only for synthetic data
    description: str = ""

    @property
    def contamination(self):
        return float(self.y.mean())


def _fetch(relative_path):
    """Download a UCI file once and cache it under data/raw."""
    target = DATA_DIR / "raw" / relative_path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(UCI_URL + relative_path, target)
    return target


def _raw_breast_cancer():
    bunch = load_breast_cancer()
    # target: 0 = malignant (212), 1 = benign (357)
    y = (bunch.target == 0).astype(int)
    return bunch.data, y, list(bunch.feature_names), "malignant tumours"


def _raw_ionosphere():
    df = pd.read_csv(_fetch("ionosphere/ionosphere.data"), header=None)
    X = df.iloc[:, :34]
    X = X.loc[:, X.std() > 0]  # attribute 2 is constant
    y = (df[34] == "b").astype(int).to_numpy()
    names = [f"a{i + 1:02d}" for i in X.columns]
    return X.to_numpy(float), y, names, "'bad' radar returns"


def _raw_parkinsons():
    df = pd.read_csv(_fetch("parkinsons/parkinsons.data"))
    y = (df["status"] == 0).astype(int).to_numpy()  # healthy is the minority
    X = df.drop(columns=["name", "status"])
    return X.to_numpy(float), y, list(X.columns), "recordings of healthy subjects"


def _raw_glass():
    cols = ["id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "type"]
    df = pd.read_csv(_fetch("glass/glass.data"), header=None, names=cols)
    y = (df["type"] == 6).astype(int).to_numpy()  # tableware glass, 9 samples
    X = df[cols[1:-1]]
    return X.to_numpy(float), y, list(X.columns), "tableware glass (type 6)"


def make_synthetic(n_normal=900, n_global=40, n_local=30, n_subspace=30,
                   dim=30, latent_dim=4, noise=0.15, seed=SYNTHETIC_SEED):
    """Gaussian clusters on a random 4-D linear manifold embedded in 30-D."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    W, W_perp = basis[:, :latent_dim], basis[:, latent_dim:]

    centers = np.array([[4.0, 0, 0, 0], [-3.0, 3, 0, 0], [-3.0, -3, 2, 0]])
    scales = np.array([0.35, 1.2, 1.2])  # one dense and two wide clusters
    sizes = [n_normal // 3] * 3
    z_normal = np.vstack([c + s * rng.normal(size=(n, latent_dim))
                          for c, s, n in zip(centers, scales, sizes)])

    def random_directions(n, k):
        v = rng.normal(size=(n, k))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    # global: far away in the latent space plus strong isotropic noise
    z_global = random_directions(n_global, latent_dim) * rng.uniform(8, 10, (n_global, 1))
    # local: a shell of 4-6 sigma around the dense cluster, still much closer
    # to it than the wide clusters are to their own centres
    z_local = centers[0] + random_directions(n_local, latent_dim) * \
        rng.uniform(4, 6, (n_local, 1)) * scales[0]
    # subspace: ordinary latent position, but shifted off the manifold
    pick = rng.integers(1, 3, n_subspace)
    z_sub = centers[pick] + scales[pick, None] * rng.normal(size=(n_subspace, latent_dim))
    offset = W_perp @ random_directions(n_subspace, dim - latent_dim).T * 2.5

    X = np.vstack([z_normal, z_global, z_local, z_sub]) @ W.T
    X += noise * rng.normal(size=X.shape)
    X[n_normal:n_normal + n_global] += rng.normal(size=(n_global, dim))
    X[-n_subspace:] += offset.T

    types = np.array(["normal"] * n_normal + ["global"] * n_global
                     + ["local"] * n_local + ["subspace"] * n_subspace)
    y = (types != "normal").astype(int)
    names = [f"x{i + 1:02d}" for i in range(dim)]
    return X, y, names, types


def make_synthetic_2d(seed=SYNTHETIC_SEED):
    """Small 2-D analogue used only to visualise decision regions."""
    rng = np.random.default_rng(seed)
    dense = rng.normal([3.0, 3.0], 0.3, size=(150, 2))
    wide = rng.normal([-2.0, -1.0], 1.3, size=(250, 2))
    angle = rng.uniform(0, 2 * np.pi, 15)
    local = np.c_[np.cos(angle), np.sin(angle)] * rng.uniform(1.3, 1.8, (15, 1)) + [3.0, 3.0]
    glob = rng.uniform(-8, 8, size=(60, 2))
    glob = glob[np.linalg.norm(glob - [-2.0, -1.0], axis=1) > 6][:15]
    X = np.vstack([dense, wide, local, glob])
    types = np.array(["normal"] * 400 + ["local"] * 15 + ["global"] * len(glob))
    return X, (types != "normal").astype(int), types


_RAW_LOADERS = {
    "breast_cancer": _raw_breast_cancer,
    "ionosphere": _raw_ionosphere,
    "parkinsons": _raw_parkinsons,
    "glass": _raw_glass,
}


def load_dataset(name, seed=0, contamination=TARGET_CONTAMINATION):
    """Return the dataset framed for anomaly detection for a given run seed."""
    if name == "synthetic":
        X, y, names, types = make_synthetic()
        return Dataset(name, X, y, names, types, "synthetic anomalies of three types")

    X, y, names, description = _RAW_LOADERS[name]()
    n_normal = int((y == 0).sum())
    n_keep = int(round(contamination * n_normal / (1 - contamination)))
    anomalies = np.flatnonzero(y == 1)
    if len(anomalies) > n_keep:
        rng = np.random.default_rng(seed)
        anomalies = np.sort(rng.choice(anomalies, n_keep, replace=False))
    keep = np.concatenate([np.flatnonzero(y == 0), anomalies])
    keep.sort()
    return Dataset(name, X[keep], y[keep], names, None, description)

