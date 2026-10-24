"""Benchmark datasets for anomaly detection.

Every dataset is read exactly as published in an anomaly-detection benchmark:
the matrix X and the labels y (1 = anomaly) come from the file, nothing is
generated, removed or subsampled here. The only later operations are the
stratified split and the scaling fitted on the training part
(common/protocol.py).

Sources
-------
ADBench  Han et al., NeurIPS 2022 - repository Minqi824/ADBench, pinned commit,
         directory adbench/datasets/Classical (files *.npz with keys X, y)
ODDS     Rayana, 2016 - arrhythmia.mat (variables X, y); the ODDS server does
         not allow automated downloads, the file is placed in data/raw/odds/
         by hand
DAMI     Campos et al., DMKD 30(4), 2016 - HeartDisease (Statlog Heart),
         variant without duplicates, not normalised, 5 % anomalies, v02

Every file is checked against its SHA256 in data/CHECKSUMS.txt and against
the shape and number of anomalies documented by the benchmark.
"""

import hashlib
import tarfile
from dataclasses import dataclass
from urllib.request import urlretrieve

import numpy as np
from scipy.io import arff, loadmat

from . import DATA_DIR

RAW = DATA_DIR / "raw"
CHECKSUMS = DATA_DIR / "CHECKSUMS.txt"

ADBENCH_COMMIT = "3dac8221081e190f157d78e93bfa8867f90d0965"
ADBENCH_URL = ("https://raw.githubusercontent.com/Minqi824/ADBench/"
               f"{ADBENCH_COMMIT}/adbench/datasets/Classical/")
DAMI_URL = "https://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/HeartDisease.tar.gz"
ODDS_PAGE = "https://odds.cs.stonybrook.edu/?p=312"
HEART_FILE = "HeartDisease/HeartDisease_withoutdupl_05_v02.arff"

TASK1_DATASETS = ("breastw", "mammography", "wine", "heart", "cardio")
TASK2_DATASETS = ("cardio", "arrhythmia", "wdbc", "campaign")
TASK3_DATASETS = ("cardio", "arrhythmia", "wdbc", "campaign")

DISPLAY_NAMES = {
    "breastw": "Breastw",
    "wdbc": "WDBC",
    "mammography": "Mammography",
    "wine": "Wine",
    "cardio": "Cardio",
    "campaign": "Campaign",
    "arrhythmia": "Arrhythmia",
    "heart": "Statlog Heart",
}


@dataclass(frozen=True)
class Spec:
    source: str  # "adbench", "odds" or "dami"
    file: str  # path relative to data/raw
    n: int
    d: int
    anomalies: int


SPECS = {
    "breastw": Spec("adbench", "adbench/4_breastw.npz", 683, 9, 239),
    "wdbc": Spec("adbench", "adbench/43_WDBC.npz", 367, 30, 10),
    "mammography": Spec("adbench", "adbench/23_mammography.npz", 11183, 6, 260),
    "wine": Spec("adbench", "adbench/45_wine.npz", 129, 13, 10),
    "cardio": Spec("adbench", "adbench/6_cardio.npz", 1831, 21, 176),
    "campaign": Spec("adbench", "adbench/5_campaign.npz", 41188, 62, 4640),
    "arrhythmia": Spec("odds", "odds/arrhythmia.mat", 452, 274, 66),
    "heart": Spec("dami", f"dami/{HEART_FILE}", 157, 13, 7),
}


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray  # 1 = anomaly, 0 = normal
    feature_names: list

    @property
    def contamination(self):
        return float(self.y.mean())


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_checksums():
    out = {}
    for line in CHECKSUMS.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, path = line.split()
            out[path] = digest
    return out


def _download(spec):
    target = RAW / spec.file
    target.parent.mkdir(parents=True, exist_ok=True)
    if spec.source == "adbench":
        urlretrieve(ADBENCH_URL + target.name, target)
    elif spec.source == "dami":
        archive = RAW / "dami" / "HeartDisease.tar.gz"
        urlretrieve(DAMI_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extract(HEART_FILE, RAW / "dami", filter="data")
    else:
        raise FileNotFoundError(
            f"{target} is missing: download arrhythmia.mat from {ODDS_PAGE} "
            "and place it there")


def _path(spec):
    """Path of the raw file, downloaded if needed and verified by SHA256."""
    path = RAW / spec.file
    if not path.exists():
        _download(spec)
    expected = _expected_checksums().get(spec.file)
    if expected is not None and _sha256(path) != expected:
        raise ValueError(f"{path}: SHA256 differs from data/CHECKSUMS.txt")
    return path


def _read(spec):
    path = _path(spec)
    if spec.source == "adbench":
        data = np.load(path, allow_pickle=True)
        return data["X"], data["y"], None
    if spec.source == "odds":
        data = loadmat(path)
        return data["X"], data["y"].ravel(), None
    records, meta = arff.loadarff(path)
    names = [n for n in meta.names() if n not in ("id", "outlier")]
    X = np.column_stack([records[n] for n in names])
    y = (records["outlier"] == b"yes").astype(int)
    return X, y, names


def load_dataset(name):
    """Return X and y exactly as stored in the benchmark file."""
    spec = SPECS[name]
    X, y, names = _read(spec)
    X, y = np.asarray(X, float), np.asarray(y).astype(int)
    assert X.shape == (spec.n, spec.d), f"{name}: shape {X.shape}"
    assert set(np.unique(y)) == {0, 1}, f"{name}: labels {np.unique(y)}"
    assert int(y.sum()) == spec.anomalies, f"{name}: {int(y.sum())} anomalies"
    assert not np.isnan(X).any(), f"{name}: missing values"
    names = names or [f"x{i + 1:02d}" for i in range(X.shape[1])]
    return Dataset(name, X, y, names)
