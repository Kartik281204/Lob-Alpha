"""
features.py
============
Microstructure feature engineering for limit order book (LOB) data.

Implements the standard feature families used in HFT alpha research
(see Cont, Kukanov & Stoikov 2014; Kercheval & Zhang 2015; Sirignano 2019):

1. Order Imbalance     — volume imbalance across book levels
2. Spread Dynamics      — bid-ask spread level & rate of change
3. Depth Ratios         — relative liquidity resting on each side
4. Trade Flow Metrics   — signed volume, trade intensity, VWAP deviation

All features are computed causally (only past/current data), which is
required for any signal that will be backtested or traded live.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _level_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith(prefix)],
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )


def order_imbalance_features(df: pd.DataFrame, n_levels: int = 5) -> pd.DataFrame:
    """Volume imbalance at level 1 and cumulative across top-N levels."""
    out = pd.DataFrame(index=df.index)
    bid_sz_cols = _level_cols(df, "bid_sz_")[:n_levels]
    ask_sz_cols = _level_cols(df, "ask_sz_")[:n_levels]

    bid_vol = df[bid_sz_cols[0]]
    ask_vol = df[ask_sz_cols[0]]
    out["imbalance_l1"] = (bid_vol - ask_vol) / (bid_vol + ask_vol)

    cum_bid = df[bid_sz_cols].sum(axis=1)
    cum_ask = df[ask_sz_cols].sum(axis=1)
    out[f"imbalance_top{n_levels}"] = (cum_bid - cum_ask) / (cum_bid + cum_ask)

    # weighted imbalance: closer levels weighted more heavily
    weights = np.array([1 / (i + 1) for i in range(n_levels)])
    w_bid = (df[bid_sz_cols].values * weights).sum(axis=1)
    w_ask = (df[ask_sz_cols].values * weights).sum(axis=1)
    out["imbalance_weighted"] = (w_bid - w_ask) / (w_bid + w_ask)

    return out


def spread_dynamics_features(df: pd.DataFrame, windows=(10, 50, 200)) -> pd.DataFrame:
    """Bid-ask spread level, relative spread, and its rolling rate of change."""
    out = pd.DataFrame(index=df.index)
    best_bid = df["bid_px_1"]
    best_ask = df["ask_px_1"]
    mid = df["mid_price"]

    spread = best_ask - best_bid
    out["spread"] = spread
    out["relative_spread"] = spread / mid

    for w in windows:
        out[f"spread_ma_{w}"] = spread.rolling(w, min_periods=1).mean()
        out[f"spread_chg_{w}"] = spread.diff(w)

    out["microprice"] = (
        best_bid * df["ask_sz_1"] + best_ask * df["bid_sz_1"]
    ) / (df["bid_sz_1"] + df["ask_sz_1"])
    out["microprice_dev"] = out["microprice"] - mid

    return out


def depth_ratio_features(df: pd.DataFrame, n_levels: int = 5) -> pd.DataFrame:
    """Relative liquidity resting on bid vs ask side, per level and cumulative."""
    out = pd.DataFrame(index=df.index)
    bid_sz_cols = _level_cols(df, "bid_sz_")[:n_levels]
    ask_sz_cols = _level_cols(df, "ask_sz_")[:n_levels]

    for i, (b, a) in enumerate(zip(bid_sz_cols, ask_sz_cols), start=1):
        out[f"depth_ratio_l{i}"] = df[b] / (df[a] + 1e-9)

    out["total_bid_depth"] = df[bid_sz_cols].sum(axis=1)
    out["total_ask_depth"] = df[ask_sz_cols].sum(axis=1)
    out["depth_ratio_total"] = out["total_bid_depth"] / (out["total_ask_depth"] + 1e-9)

    return out


def trade_flow_features(df: pd.DataFrame, windows=(20, 100, 500)) -> pd.DataFrame:
    """Signed volume / trade intensity / VWAP deviation from executed trades."""
    out = pd.DataFrame(index=df.index)
    signed_vol = (df["trade_size"].fillna(0) * df["trade_side"].fillna(0))
    traded = df["trade_size"].notna().astype(int)

    for w in windows:
        out[f"signed_vol_sum_{w}"] = signed_vol.rolling(w, min_periods=1).sum()
        out[f"trade_intensity_{w}"] = traded.rolling(w, min_periods=1).sum()

    vwap = (
        (df["trade_price"].ffill() * df["trade_size"].fillna(0))
        .rolling(200, min_periods=1)
        .sum()
        / df["trade_size"].fillna(0).rolling(200, min_periods=1).sum().replace(0, np.nan)
    )
    out["vwap_dev"] = df["mid_price"] - vwap

    return out


def build_feature_matrix(df: pd.DataFrame, n_levels: int = 5) -> pd.DataFrame:
    """Assemble the full feature matrix from all four feature families."""
    feats = pd.concat(
        [
            order_imbalance_features(df, n_levels),
            spread_dynamics_features(df),
            depth_ratio_features(df, n_levels),
            trade_flow_features(df),
        ],
        axis=1,
    )
    feats = feats.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return feats


def make_return_labels(df: pd.DataFrame, horizon: int = 50) -> pd.Series:
    """
    Continuous forward log-return label, `horizon` steps ahead.

    Preferred over `make_labels`' hard {-1,0,1} classes for weak HFT
    signals: regression preserves magnitude/confidence information that a
    thresholded classifier throws away, which matters when you plan to
    trade only the highest-conviction quantile of predictions (the
    standard way real microstructure alpha signals are deployed, since
    per-prediction accuracy is inherently low but a systematic edge
    survives in aggregate over many trades).
    """
    return np.log(df["mid_price"].shift(-horizon) / df["mid_price"])


def make_labels(
    df: pd.DataFrame, horizon: int = 50, threshold: float = 0.0005
) -> pd.Series:
    """
    Label short-term price movement `horizon` steps ahead.

    Returns a categorical target: -1 (down), 0 (flat), 1 (up), based on
    forward log-return exceeding `threshold`. This is the standard
    "directional classification" framing used in HFT alpha research
    (as opposed to raw regression on noisy tick returns).
    """
    fwd_ret = np.log(df["mid_price"].shift(-horizon) / df["mid_price"])
    labels = pd.Series(0, index=df.index)
    labels[fwd_ret > threshold] = 1
    labels[fwd_ret < -threshold] = -1
    return labels
