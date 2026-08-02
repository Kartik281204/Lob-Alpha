import numpy as np
import pandas as pd
import pytest

from src.pipeline import make_sequences, quantile_signal, information_coefficient


def test_make_sequences_shapes():
    features = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 4)))
    labels = pd.Series(np.random.default_rng(1).normal(size=100))
    X_seq, y_seq, idx = make_sequences(features, labels, seq_len=10)
    assert X_seq.shape == (90, 10, 4)
    assert y_seq.shape == (90,)
    assert len(idx) == 90


def test_make_sequences_alignment():
    """y_seq[i] should equal labels at position i + seq_len."""
    features = pd.DataFrame(np.arange(20).reshape(20, 1).astype(float))
    labels = pd.Series(np.arange(20).astype(float) * 10)
    X_seq, y_seq, idx = make_sequences(features, labels, seq_len=5)
    assert y_seq[0] == labels.iloc[5]
    assert y_seq[-1] == labels.iloc[19]


def test_quantile_signal_extreme_values_only():
    pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    sig = quantile_signal(pred, q=0.2)
    assert (sig[:2] == -1).all()  # bottom 20%
    assert (sig[-2:] == 1).all()  # top 20%
    assert (sig[2:-2] == 0).all()  # rest flat


def test_quantile_signal_output_shape_matches_input():
    pred = np.random.default_rng(0).normal(size=500)
    sig = quantile_signal(pred, q=0.1)
    assert sig.shape == pred.shape
    assert set(np.unique(sig)).issubset({-1, 0, 1})


def test_information_coefficient_perfect_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(information_coefficient(x, x), 1.0)


def test_information_coefficient_no_correlation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=10_000)
    y = rng.normal(size=10_000)
    assert abs(information_coefficient(x, y)) < 0.05


@pytest.mark.slow
def test_full_pipeline_smoke_run(monkeypatch):
    """
    End-to-end smoke test with a tiny simulation so it runs in CI in a few
    seconds. Only checks the pipeline completes and returns well-formed
    output -- not that the strategy is profitable (see README for why
    that's a genuine, expected research finding, not a bug).
    """
    import src.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "N_STEPS", 1200)
    monkeypatch.setattr(pipeline_module, "SEQ_LEN", 10)
    monkeypatch.setattr(pipeline_module, "HORIZON", 20)

    result = pipeline_module.run_pipeline()

    assert "ic" in result
    assert np.isfinite(result["ic"])
    for key in ("backtest_aggressive", "backtest_passive"):
        assert "stats" in result[key]
        assert "equity_curve" in result[key]
