import numpy as np
import pytest

from src.models import SklearnMLPAlpha, LinearAlpha, get_model, TORCH_AVAILABLE


@pytest.fixture
def toy_sequences():
    rng = np.random.default_rng(0)
    n_samples, seq_len, n_features = 200, 10, 6
    X = rng.normal(size=(n_samples, seq_len, n_features)).astype(np.float32)
    # y correlated with the last timestep's first feature, plus noise
    y = X[:, -1, 0] * 0.5 + rng.normal(scale=0.1, size=n_samples)
    return X.astype(np.float32), y.astype(np.float32)


def test_sklearn_mlp_alpha_fit_predict_shape(toy_sequences):
    X, y = toy_sequences
    model = SklearnMLPAlpha(seq_len=X.shape[1], task="regression")
    model.fit(X[:150], y[:150])
    preds = model.predict(X[150:])
    assert preds.shape == (50,)


def test_sklearn_mlp_alpha_learns_signal(toy_sequences):
    """Sanity check: predictions should correlate with the true target better than chance."""
    X, y = toy_sequences
    model = SklearnMLPAlpha(seq_len=X.shape[1], task="regression")
    model.fit(X[:150], y[:150])
    preds = model.predict(X[150:])
    corr = np.corrcoef(preds, y[150:])[0, 1]
    assert corr > 0.2


def test_linear_alpha_fit_predict_shape(toy_sequences):
    X, y = toy_sequences
    model = LinearAlpha(task="regression")
    model.fit(X[:150], y[:150])
    preds = model.predict(X[150:])
    assert preds.shape == (50,)


def test_linear_alpha_window_to_flat_dims(toy_sequences):
    X, _ = toy_sequences
    flat = LinearAlpha._window_to_flat(X)
    # last + mean + std, each of size n_features
    assert flat.shape == (X.shape[0], X.shape[2] * 3)


def test_get_model_factory_returns_usable_model(toy_sequences):
    X, y = toy_sequences
    model = get_model(n_features=X.shape[2], seq_len=X.shape[1], task="regression")
    if not TORCH_AVAILABLE:
        assert isinstance(model, SklearnMLPAlpha)
        model.fit(X[:150], y[:150])
        preds = model.predict(X[150:])
        assert preds.shape == (50,)
    else:
        import torch
        assert isinstance(model, torch.nn.Module)


def test_classification_mode_predict_proba_shape(toy_sequences):
    X, y = toy_sequences
    y_cls = (y > np.median(y)).astype(int)  # binary-ish class labels for a smoke test
    model = SklearnMLPAlpha(seq_len=X.shape[1], task="classification")
    model.fit(X[:150], y_cls[:150])
    proba = model.predict_proba(X[150:])
    assert proba.shape[0] == 50


def test_predict_proba_raises_for_regression_task(toy_sequences):
    X, y = toy_sequences
    model = SklearnMLPAlpha(seq_len=X.shape[1], task="regression")
    model.fit(X[:150], y[:150])
    with pytest.raises(ValueError):
        model.predict_proba(X[150:])
