import numpy as np
import pandas as pd
import pytest

from src.simulate_lob import simulate_lob
from src.features import (
    order_imbalance_features,
    spread_dynamics_features,
    depth_ratio_features,
    trade_flow_features,
    build_feature_matrix,
    make_labels,
    make_return_labels,
)


@pytest.fixture(scope="module")
def lob():
    return simulate_lob(n_steps=1000, n_levels=5, seed=11)


def test_order_imbalance_bounded(lob):
    feats = order_imbalance_features(lob)
    # imbalance ratios are normalized differences -> must be in [-1, 1]
    for col in feats.columns:
        assert feats[col].between(-1, 1).all(), col


def test_spread_dynamics_spread_nonnegative(lob):
    feats = spread_dynamics_features(lob)
    assert (feats["spread"] >= 0).all()
    assert (feats["relative_spread"] >= 0).all()


def test_depth_ratio_positive(lob):
    feats = depth_ratio_features(lob)
    ratio_cols = [c for c in feats.columns if c.startswith("depth_ratio")]
    for col in ratio_cols:
        assert (feats[col] > 0).all(), col


def test_trade_flow_no_nans_after_fill(lob):
    feats = trade_flow_features(lob)
    # vwap_dev may be NaN before the first trade only; everything else must be clean
    other_cols = [c for c in feats.columns if c != "vwap_dev"]
    assert not feats[other_cols].isna().any().any()


def test_build_feature_matrix_no_nan_or_inf(lob):
    feats = build_feature_matrix(lob)
    assert not feats.isna().any().any()
    assert np.isfinite(feats.values).all()


def test_build_feature_matrix_same_index(lob):
    feats = build_feature_matrix(lob)
    assert feats.index.equals(lob.index)


def test_make_labels_values_in_expected_set(lob):
    labels = make_labels(lob, horizon=20, threshold=0.0005)
    assert set(labels.unique()).issubset({-1, 0, 1})


def test_make_labels_tail_is_zero_default_fill(lob):
    # forward-looking labels near the end have NaN forward return -> default 0
    labels = make_labels(lob, horizon=20, threshold=0.0005)
    assert labels.iloc[-1] == 0


def test_make_return_labels_matches_manual_calc(lob):
    horizon = 15
    labels = make_return_labels(lob, horizon=horizon)
    manual = np.log(lob["mid_price"].shift(-horizon) / lob["mid_price"])
    pd.testing.assert_series_equal(labels, manual, check_names=False)


def test_make_return_labels_tail_is_nan(lob):
    horizon = 15
    labels = make_return_labels(lob, horizon=horizon)
    assert labels.iloc[-horizon:].isna().all()


def test_causality_features_do_not_use_future_trade_price(lob):
    """
    Regression guard: trade_flow_features must not silently backward-fill
    (which would leak future trade prices into past rows).
    """
    feats = trade_flow_features(lob)
    # vwap_dev before the very first trade must be NaN, not filled from a
    # later trade -- this would indicate accidental bfill/lookahead.
    first_trade_i = lob["trade_price"].first_valid_index()
    if first_trade_i is not None and first_trade_i > lob.index[0]:
        pre_trade = feats.loc[: first_trade_i - 1, "vwap_dev"] if first_trade_i > 0 else pd.Series(dtype=float)
        assert pre_trade.isna().all()
