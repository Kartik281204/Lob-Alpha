"""
pipeline.py
============
End-to-end run: simulate LOB data -> engineer features -> label (continuous
forward return) -> build rolling sequences -> train model -> convert
predictions to a trading signal via top/bottom quantile selection ->
backtest with transaction costs.

Why quantile-based signal extraction instead of hard classification:
real microstructure alpha signals are weak per-prediction (information
coefficient / correlation with forward returns is typically 0.02-0.15)
but produce a systematic, tradeable edge when you only act on the
highest-conviction subset of predictions and let the law of large numbers
work across many trades. This is standard practice in HFT alpha research
(see qlib's "IC" evaluation framework, FinRL's ranking-based execution).

Usage:
    python -m src.pipeline
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulate_lob import simulate_lob
from .features import build_feature_matrix, make_return_labels
from .models import get_model, LinearAlpha, TORCH_AVAILABLE
from .backtest import run_backtest

SEQ_LEN = 20
HORIZON = 50
N_STEPS = 20_000
SIGNAL_QUANTILE = 0.10  # trade only the top/bottom 10% most confident predictions


def make_sequences(features: pd.DataFrame, labels: pd.Series, seq_len: int):
    """Build rolling (seq_len, n_features) windows -> label at window end."""
    X_arr = features.values
    y_arr = labels.values
    n = len(features) - seq_len
    X_seq = np.zeros((n, seq_len, X_arr.shape[1]), dtype=np.float32)
    y_seq = np.zeros(n, dtype=np.float32)
    for i in range(n):
        X_seq[i] = X_arr[i : i + seq_len]
        y_seq[i] = y_arr[i + seq_len]
    return X_seq, y_seq, features.index[seq_len:]


def quantile_signal(pred: np.ndarray, q: float = SIGNAL_QUANTILE) -> np.ndarray:
    """Convert continuous predictions to {-1, 0, 1}: trade only top/bottom quantile."""
    lo, hi = np.quantile(pred, [q, 1 - q])
    signal = np.zeros_like(pred, dtype=np.int64)
    signal[pred >= hi] = 1
    signal[pred <= lo] = -1
    return signal


def information_coefficient(pred: np.ndarray, actual: np.ndarray) -> float:
    """Rank correlation between predicted and realized forward return."""
    return float(np.corrcoef(pred, actual)[0, 1])


def run_pipeline():
    print(f"1) Simulating {N_STEPS:,} LOB events...")
    lob = simulate_lob(n_steps=N_STEPS, seed=7)

    print("2) Engineering microstructure features...")
    feats = build_feature_matrix(lob)
    labels_raw = make_return_labels(lob, horizon=HORIZON)

    print("3) Building rolling sequences...")
    X_seq, y_seq, seq_index = make_sequences(feats, labels_raw, SEQ_LEN)

    # the last `HORIZON` rows have no forward return (nothing to look ahead
    # to) -- drop them rather than let NaNs poison training/evaluation
    valid = ~np.isnan(y_seq)
    X_seq, y_seq, seq_index = X_seq[valid], y_seq[valid], seq_index[valid]

    n = len(X_seq)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]
    test_index = seq_index[val_end:]
    test_lob = lob.loc[test_index]

    print(f"   train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    print("4) Training linear baseline (Ridge) for comparison...")
    baseline = LinearAlpha(task="regression").fit(X_train, y_train)
    baseline_val_pred = baseline.predict(X_val)
    baseline_val_ic = information_coefficient(baseline_val_pred, y_val)
    print(f"   baseline val IC: {baseline_val_ic:.4f}")

    print(f"5) Training deep model (torch available: {TORCH_AVAILABLE})...")
    model = get_model(n_features=X_seq.shape[2], seq_len=SEQ_LEN, architecture="lstm", task="regression")

    if TORCH_AVAILABLE:
        from .models import train_torch_model
        import torch
        model = train_torch_model(model, X_train, y_train, X_val, y_val, task="regression", epochs=15)
        model.eval()
        with torch.no_grad():
            deep_val_pred = model(torch.tensor(X_val, dtype=torch.float32)).squeeze(-1).numpy()
    else:
        model.fit(X_train, y_train)
        deep_val_pred = model.predict(X_val)
    deep_val_ic = information_coefficient(deep_val_pred, y_val)
    print(f"   deep model val IC: {deep_val_ic:.4f}")

    print("6) Selecting best model by validation IC (standard quant-research practice:")
    print("   always benchmark deep models against a linear baseline)...")
    if deep_val_ic >= baseline_val_ic:
        print("   -> deep model selected")
        if TORCH_AVAILABLE:
            with torch.no_grad():
                test_pred = model(torch.tensor(X_test, dtype=torch.float32)).squeeze(-1).numpy()
        else:
            test_pred = model.predict(X_test)
    else:
        print("   -> linear baseline selected (outperformed deep model on validation)")
        test_pred = baseline.predict(X_test)

    ic = information_coefficient(test_pred, y_test)
    print(f"   out-of-sample Information Coefficient (corr with realized return): {ic:.4f}")

    print(f"7) Converting predictions to trading signal (top/bottom {SIGNAL_QUANTILE:.0%} quantile)...")
    signal_arr = quantile_signal(test_pred, SIGNAL_QUANTILE)
    signal = pd.Series(signal_arr, index=test_index)
    print(f"   signal counts: long={int((signal_arr==1).sum())}  short={int((signal_arr==-1).sum())}  flat={int((signal_arr==0).sum())}")

    print("8) Backtesting with transaction costs, both execution styles...")
    result_aggr = run_backtest(test_lob, signal, horizon=HORIZON, order_size=100,
                                fee_bps=1.0, impact_coef=0.15, execution="aggressive")
    result_pass = run_backtest(test_lob, signal, horizon=HORIZON, order_size=100,
                                fee_bps=1.0, impact_coef=0.15, execution="passive")

    print("\n--- Aggressive (taker: crosses spread each fill) ---")
    for k, v in result_aggr["stats"].items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    print("\n--- Passive (maker: posts at touch, no spread cost) ---")
    for k, v in result_pass["stats"].items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    return {
        "lob": lob, "features": feats, "model": model,
        "test_pred": test_pred, "ic": ic,
        "backtest_aggressive": result_aggr,
        "backtest_passive": result_pass,
    }


if __name__ == "__main__":
    run_pipeline()
