"""Task 2: detection in different data representations.

For every dataset (>= 22 features) and seed, each representation
configuration is fitted on the training part; IForest, LOF and OCSVM are then
fitted in the representation space with a small hyperparameter grid. PCA and
the autoencoder also provide their own score (reconstruction error).
Representation and detector settings are chosen later on validation data.

Outputs (results/task2/): runs.csv, structure.csv, embeddings_2d.csv.gz,
ae_history.csv
"""

import json
import os
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.manifold import trustworthiness  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

from common import ROOT  # noqa: E402
from common.anomaly_types import anomaly_kind  # noqa: E402
from common.datasets import HIGHDIM_DATASETS, load_dataset  # noqa: E402
from common.detectors import build as build_detector  # noqa: E402
from common.metrics import best_f1_threshold, evaluate, ranking_metrics  # noqa: E402
from common.protocol import SEEDS, make_experiment  # noqa: E402
from task2_representations.representations import build as build_rep  # noqa: E402

OUT = ROOT / "results" / "task2"
N_JOBS = int(os.environ.get("N_JOBS", 6))


def representation_grid(dim):
    yield "Original", {}
    for k in (1, 2, 3, 5, 8, 10, 15, 20):
        if k < dim:
            yield "PCA", {"n_components": k}
    for c in (2, 5, 10):
        for n in (5, 15, 50):
            for md in (0.0, 0.5):
                yield "UMAP", {"n_components": c, "n_neighbors": n, "min_dist": md}
    for latent in (2, 4, 8, 16):
        yield "AE", {"latent_dim": latent, "hidden": [64, 32]}
    for hidden in ([32], [128, 64]):  # architecture ablation at latent 8
        yield "AE", {"latent_dim": 8, "hidden": hidden}


def detector_grid():
    yield "IForest", {"n_estimators": 200}
    for k in (5, 10, 20, 35, 50):
        yield "LOF", {"n_neighbors": k}
    for nu in (0.05, 0.1, 0.5):
        for g in (0.1, 1.0, 10.0):
            yield "OCSVM", {"kernel": "rbf", "nu": nu, "gamma_factor": g}


def _metrics_row(exp, kinds, s_val, s_test):
    t = best_f1_threshold(exp.y_val, s_val)
    val = ranking_metrics(exp.y_val, s_val)
    pred = (s_test >= t).astype(int)
    test = evaluate(exp.y_test, s_test, pred)
    row = {"val_roc_auc": val["roc_auc"], "val_pr_auc": val["pr_auc"],
           **{f"test_{k}": v for k, v in test.items()}}
    for kind in ("global", "local", "subspace"):  # recall per anomaly type
        mask = kinds == kind
        row[f"test_recall_{kind}"] = pred[mask].mean() if mask.any() else np.nan
    return row


def _warm_up_umap():
    """Compile UMAP's numba code once per process so it is not timed."""
    rng = np.random.default_rng(0)
    build_rep("UMAP", n_components=2).fit(rng.normal(size=(60, 5)))


def run_job(dataset, seed):
    torch.set_num_threads(1)
    _warm_up_umap()
    exp = make_experiment(dataset, seed)
    types = load_dataset(dataset, seed).anomaly_type
    kinds = anomaly_kind(dataset, seed)[exp.idx_test]
    rows, structure, embeddings, history = [], [], [], []
    for rep_kind, rep_params in representation_grid(exp.X_train.shape[1]):
        params = dict(rep_params)
        if rep_kind in ("UMAP", "AE"):
            params["seed"] = seed
        rep = build_rep(rep_kind, **params)
        Z_tr, Z_va, Z_te = rep.fit_transform_all(exp.X_train, exp.X_val, exp.X_test)
        base = {"dataset": dataset, "seed": seed, "rep": rep_kind,
                "rep_params": json.dumps(rep_params, sort_keys=True),
                "rep_dim": Z_tr.shape[1], "rep_fit_time": rep.fit_time_,
                "rep_transform_time": rep.transform_time_}

        info = {**base,
                "explained_variance": getattr(rep, "explained_variance_", np.nan),
                "ae_epochs": getattr(rep, "epochs_trained_", np.nan),
                "trustworthiness": (1.0 if rep_kind == "Original" else
                                    trustworthiness(exp.X_test, Z_te, n_neighbors=10)),
                "silhouette": silhouette_score(Z_te, exp.y_test)}
        structure.append(info)

        if rep.native_score:
            t0 = time.perf_counter()
            s_te = rep.anomaly_score(exp.X_test)
            score_time = time.perf_counter() - t0
            s_va = rep.anomaly_score(exp.X_val)
            rows.append({**base, "detector": "Reconstruction", "det_params": "{}",
                         "det_fit_time": 0.0, "det_score_time": score_time,
                         **_metrics_row(exp, kinds, s_va, s_te)})

        for det_kind, det_params in detector_grid():
            t0 = time.perf_counter()
            det = build_detector(det_kind, **det_params).fit(Z_tr)
            fit_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            s_te = det.score(Z_te)
            score_time = time.perf_counter() - t0
            rows.append({**base, "detector": det_kind,
                         "det_params": json.dumps(det_params, sort_keys=True),
                         "det_fit_time": fit_time, "det_score_time": score_time,
                         **_metrics_row(exp, kinds, det.score(Z_va), s_te)})

        if seed == 0 and Z_tr.shape[1] == 2:
            for part, Z, idx, y in (("train", Z_tr, exp.idx_train, exp.y_train),
                                    ("val", Z_va, exp.idx_val, exp.y_val),
                                    ("test", Z_te, exp.idx_test, exp.y_test)):
                embeddings.append(pd.DataFrame({
                    "dataset": dataset, "rep": rep_kind, "rep_params": base["rep_params"],
                    "part": part, "y": y,
                    "type": types[idx] if types is not None else np.where(y == 1, "anomaly", "normal"),
                    "z1": Z[:, 0], "z2": Z[:, 1]}))
        if seed == 0 and rep_kind == "AE":
            for epoch, (tr, ho) in enumerate(rep.history_):
                history.append({"dataset": dataset, "rep_params": base["rep_params"],
                                "epoch": epoch + 1, "train_loss": tr, "holdout_loss": ho})
    return rows, structure, embeddings, history


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(d, s) for d in HIGHDIM_DATASETS for s in SEEDS]
    results = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_job)(d, s) for d, s in jobs)
    pd.DataFrame([r for res in results for r in res[0]]).to_csv(OUT / "runs.csv", index=False)
    pd.DataFrame([r for res in results for r in res[1]]).to_csv(OUT / "structure.csv", index=False)
    pd.concat([e for res in results for e in res[2]]).to_csv(
        OUT / "embeddings_2d.csv.gz", index=False)
    pd.DataFrame([h for res in results for h in res[3]]).to_csv(
        OUT / "ae_history.csv", index=False)


if __name__ == "__main__":
    main()
