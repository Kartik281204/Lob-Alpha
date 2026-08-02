"""
models.py
==========
Deep learning architectures for predicting short-term price direction /
volatility bursts from LOB microstructure features.

Two implementations are provided:

1. PyTorch models (LSTMAlpha, TransformerAlpha) — the real production
   architectures. Requires `torch` (pip install torch). Use these when
   running with GPU/CPU torch available.

2. `SklearnMLPAlpha` — a pure scikit-learn MLP fallback with hand-built
   lag/window features, used automatically when torch is not installed
   (e.g. this sandbox has no internet access to install torch). It follows
   the same fit/predict interface so `pipeline.py` and `backtest.py` don't
   need to know which one is active.

Swap between them by setting `USE_TORCH = True/False` at the top of
pipeline.py, or just let `get_model()` auto-detect.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------
# 1. PyTorch models — production architectures
# --------------------------------------------------------------------------
if TORCH_AVAILABLE:

    class LSTMAlpha(nn.Module):
        """
        LSTM sequence model over a rolling window of LOB feature vectors.

        n_outputs=1 for regression (predict continuous forward return --
        recommended, see features.make_return_labels) or n_outputs=3 for
        {-1,0,1} directional classification.
        """

        def __init__(self, n_features: int, hidden_size: int = 64, n_layers: int = 2, n_outputs: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                dropout=0.2 if n_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, n_outputs),
            )

        def forward(self, x):  # x: (batch, seq_len, n_features)
            out, (h_n, _) = self.lstm(x)
            last_hidden = h_n[-1]  # (batch, hidden_size)
            return self.head(last_hidden)

    class TransformerAlpha(nn.Module):
        """Transformer encoder over a rolling window of LOB feature vectors."""

        def __init__(
            self,
            n_features: int,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            n_outputs: int = 1,
            seq_len: int = 50,
        ):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, n_outputs)
            )

        def forward(self, x):  # x: (batch, seq_len, n_features)
            h = self.input_proj(x) + self.pos_embedding[:, : x.size(1), :]
            h = self.encoder(h)
            pooled = h.mean(dim=1)  # mean-pool over sequence
            return self.head(pooled)

    def train_torch_model(model, X_train, y_train, X_val, y_val, task="regression", epochs=15, lr=1e-3, batch_size=256):
        """
        Standard supervised training loop for the torch models above.

        task="regression": y are continuous forward returns, MSE loss (recommended).
        task="classification": y are {0,1,2} class indices, cross-entropy loss.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32 if task == "regression" else torch.long)
        n = len(X_train_t)

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n)
            total_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                xb, yb = X_train_t[idx].to(device), y_train_t[idx].to(device)
                opt.zero_grad()
                out = model(xb)
                if task == "regression":
                    out = out.squeeze(-1)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(idx)

            model.eval()
            with torch.no_grad():
                val_out = model(torch.tensor(X_val, dtype=torch.float32).to(device))
                if task == "regression":
                    val_pred = val_out.squeeze(-1).cpu().numpy()
                    val_metric = np.corrcoef(val_pred, y_val)[0, 1]
                    metric_name = "val_corr"
                else:
                    val_pred = val_out.argmax(dim=1).cpu().numpy()
                    val_metric = (val_pred == y_val).mean()
                    metric_name = "val_acc"
            print(f"epoch {epoch+1:02d}  train_loss={total_loss/n:.6f}  {metric_name}={val_metric:.4f}")
        return model


# --------------------------------------------------------------------------
# 2. scikit-learn fallback — runs anywhere, no torch dependency
# --------------------------------------------------------------------------
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler


class LinearAlpha:
    """
    Linear baseline (Ridge regression / logistic regression) on the same
    window-summary features as SklearnMLPAlpha.

    Always benchmark a deep model against this. Microstructure alpha
    signals are frequently close to linear in the engineered features
    (e.g. order-imbalance -> short-term drift), and with limited training
    data a regularized linear model can match or beat a nonlinear net that
    has more capacity than the signal-to-noise ratio supports. Reporting
    this comparison honestly is standard practice in quant research, not
    a weakness of the pipeline.
    """

    def __init__(self, task: str = "regression", alpha: float = 1.0):
        self.task = task
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha) if task == "regression" else LogisticRegression(
            max_iter=1000, class_weight="balanced"
        )

    @staticmethod
    def _window_to_flat(X_seq: np.ndarray) -> np.ndarray:
        last = X_seq[:, -1, :]
        mean = X_seq.mean(axis=1)
        std = X_seq.std(axis=1)
        return np.concatenate([last, mean, std], axis=1)

    def fit(self, X_train_seq: np.ndarray, y_train: np.ndarray):
        X_flat = self.scaler.fit_transform(self._window_to_flat(X_train_seq))
        self.model.fit(X_flat, y_train)
        return self

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        X_flat = self.scaler.transform(self._window_to_flat(X_seq))
        return self.model.predict(X_flat)


class SklearnMLPAlpha:
    """
    Offline-friendly stand-in for the sequence models above.

    Instead of feeding a raw (seq_len, n_features) tensor to an LSTM, we
    flatten a rolling window into a single feature vector (lag features),
    which an MLP can consume. This captures short-range temporal structure
    without requiring torch, at the cost of the longer-range memory an
    LSTM/Transformer would learn.

    task="regression" (default, recommended): predicts continuous forward
    return -- pairs with features.make_return_labels() and quantile-based
    signal extraction in pipeline.py, which is far more effective on the
    weak-but-real signal typical of microstructure alpha than hard
    classification.
    task="classification": predicts {-1,0,1} directional class.
    """

    def __init__(self, seq_len: int = 20, task: str = "regression",
                 hidden_layer_sizes=(32,), max_iter: int = 500, alpha: float = 0.1):
        self.seq_len = seq_len
        self.task = task
        self.scaler = StandardScaler()
        model_cls = MLPRegressor if task == "regression" else MLPClassifier
        self.model = model_cls(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            alpha=alpha,
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=42,
        )

    @staticmethod
    def _window_to_flat(X_seq: np.ndarray) -> np.ndarray:
        """
        (n_samples, seq_len, n_features) -> (n_samples, 3 * n_features).

        Rather than a raw flatten (seq_len * n_features dims, which badly
        dilutes signal-to-noise for a plain MLP on a weak microstructure
        signal), summarize each window with its last (most recent) reading
        plus its mean and std -- a standard lag/rolling-stat feature
        engineering approach that an MLP can actually learn from. The real
        LSTM/Transformer models don't need this trick; they learn the
        useful summary directly from the raw sequence.
        """
        last = X_seq[:, -1, :]
        mean = X_seq.mean(axis=1)
        std = X_seq.std(axis=1)
        return np.concatenate([last, mean, std], axis=1)

    def fit(self, X_train_seq: np.ndarray, y_train: np.ndarray):
        X_flat = self._window_to_flat(X_train_seq)
        X_flat = self.scaler.fit_transform(X_flat)
        self.model.fit(X_flat, y_train)
        return self

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        X_flat = self._window_to_flat(X_seq)
        X_flat = self.scaler.transform(X_flat)
        return self.model.predict(X_flat)

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only available for task='classification'")
        X_flat = self._window_to_flat(X_seq)
        X_flat = self.scaler.transform(X_flat)
        return self.model.predict_proba(X_flat)


def get_model(n_features: int, seq_len: int, architecture: str = "lstm", task: str = "regression"):
    """
    Factory: returns a torch model if torch is available, else the sklearn
    fallback. `architecture` in {"lstm", "transformer"} only matters when
    torch is available. `task` in {"regression", "classification"}.
    """
    n_outputs = 1 if task == "regression" else 3
    if TORCH_AVAILABLE:
        if architecture == "transformer":
            return TransformerAlpha(n_features=n_features, seq_len=seq_len, n_outputs=n_outputs)
        return LSTMAlpha(n_features=n_features, n_outputs=n_outputs)
    print("[models.py] torch not found — using SklearnMLPAlpha fallback. "
          "Install torch for the real LSTM/Transformer architectures.")
    return SklearnMLPAlpha(seq_len=seq_len, task=task)
