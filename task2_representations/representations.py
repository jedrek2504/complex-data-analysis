"""Data representations compared in task 2.

All representations are fitted on the (unlabelled) training part only and
then applied to the validation and test parts. After the transformation the
representation is rescaled by a single factor (the mean standard deviation
of its training coordinates), which keeps its geometry intact but puts every
representation on a comparable scale, so that detector hyperparameters such
as gamma = 1 / d mean the same in every space.
"""

import os
import time
import warnings

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

warnings.filterwarnings("ignore", message=".*n_jobs value.*")


class Representation:
    native_score = False  # True if the representation yields its own anomaly score

    def fit(self, X):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform_all(self, X_train, *others):
        """Fit on X_train, transform all parts and apply isotropic rescaling."""
        t0 = time.perf_counter()
        self.fit(X_train)
        Z_train = self.transform(X_train)
        self.fit_time_ = time.perf_counter() - t0
        t0 = time.perf_counter()
        Z_others = [self.transform(X) for X in others]
        self.transform_time_ = time.perf_counter() - t0
        center = Z_train.mean(axis=0)
        scale = np.sqrt(np.mean(Z_train.var(axis=0))) or 1.0
        return [(Z - center) / scale for Z in (Z_train, *Z_others)]


class Original(Representation):
    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X, float)


class PCARep(Representation):
    native_score = True

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        k = min(self.n_components, X.shape[1], len(X) - 1)
        self.model = PCA(n_components=k, random_state=0).fit(X)
        self.explained_variance_ = float(self.model.explained_variance_ratio_.sum())
        return self

    def transform(self, X):
        return self.model.transform(X)

    def anomaly_score(self, X):
        """Squared reconstruction error (distance to the PCA subspace)."""
        recon = self.model.inverse_transform(self.model.transform(X))
        return np.sum((X - recon) ** 2, axis=1)


class UMAPRep(Representation):
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, seed=0):
        self.params = dict(n_components=n_components, n_neighbors=n_neighbors,
                           min_dist=min_dist, random_state=seed)

    def fit(self, X):
        import umap  # imported lazily: compiling numba code is slow

        params = dict(self.params, n_neighbors=min(self.params["n_neighbors"], len(X) - 1))
        self.model = umap.UMAP(**params).fit(X)
        return self

    def transform(self, X):
        return self.model.transform(X)


class _AENet(nn.Module):
    def __init__(self, dim, latent, hidden=(64, 32), dropout=0.1):
        super().__init__()
        enc, prev = [], dim
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        enc.append(nn.Linear(prev, latent))
        dec, prev = [], latent
        for h in reversed(hidden):
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec.append(nn.Linear(prev, dim))
        self.encoder, self.decoder = nn.Sequential(*enc), nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class AutoencoderRep(Representation):
    """Under-complete MLP autoencoder d -> 64 -> 32 -> latent -> 32 -> 64 -> d.

    Trained on the training part with MSE loss (Adam, weight decay); 15 % of
    the training rows are held out for early stopping, so no labels and no
    validation/test rows are used during training. The mini-batch has 32 rows,
    or more for large datasets so that an epoch has at most 64 updates
    (Campaign: 274 rows), which keeps the training time manageable.
    """

    native_score = True

    def __init__(self, latent_dim=8, hidden=(64, 32), epochs=400, batch_size=None,
                 lr=1e-3, weight_decay=1e-5, patience=30, seed=0, device=None):
        self.latent_dim, self.hidden = latent_dim, tuple(hidden)
        self.epochs, self.batch_size = epochs, batch_size
        self.lr, self.weight_decay, self.patience, self.seed = lr, weight_decay, patience, seed
        self.device = device or os.environ.get("AE_DEVICE", "cpu")

    def fit(self, X):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = torch.tensor(np.asarray(X, np.float32))
        idx = rng.permutation(len(X))
        n_hold = max(int(0.15 * len(X)), 1)
        X_hold, X_fit = X[idx[:n_hold]].to(self.device), X[idx[n_hold:]].to(self.device)
        batch_size = self.batch_size or max(32, -(-len(X_fit) // 64))
        self.net = _AENet(X.shape[1], self.latent_dim, self.hidden).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        best, best_state, wait = np.inf, None, 0
        self.history_ = []
        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(X_fit), device=self.device)
            train_loss = 0.0
            for start in range(0, len(X_fit), batch_size):
                batch = X_fit[perm[start:start + batch_size]]
                recon, _ = self.net(batch)
                loss = nn.functional.mse_loss(recon, batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item() * len(batch)
            self.net.eval()
            with torch.no_grad():
                hold_loss = nn.functional.mse_loss(self.net(X_hold)[0], X_hold).item()
            self.history_.append((train_loss / len(X_fit), hold_loss))
            if hold_loss < best - 1e-5:
                best, wait = hold_loss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        self.epochs_trained_ = len(self.history_)
        return self

    def _forward(self, X):
        with torch.no_grad():
            x = torch.tensor(np.asarray(X, np.float32), device=self.device)
            recon, z = self.net(x)
        return x.cpu().numpy(), recon.cpu().numpy(), z.cpu().numpy()

    def transform(self, X):
        return self._forward(X)[2]

    def anomaly_score(self, X):
        x, recon, _ = self._forward(X)
        return np.mean((x - recon) ** 2, axis=1)


def build(kind, **params):
    return {"Original": Original, "PCA": PCARep, "UMAP": UMAPRep,
            "AE": AutoencoderRep}[kind](**params)
