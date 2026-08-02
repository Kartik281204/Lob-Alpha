import numpy as np
import pandas as pd
import pytest

from src.simulate_lob import simulate_lob


def test_output_shape_and_length():
    df = simulate_lob(n_steps=500, n_levels=5, seed=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500


def test_expected_columns_present():
    df = simulate_lob(n_steps=100, n_levels=3, seed=1)
    expected = {"mid_price", "trade_price", "trade_size", "trade_side"}
    for lvl in range(1, 4):
        expected |= {f"bid_px_{lvl}", f"bid_sz_{lvl}", f"ask_px_{lvl}", f"ask_sz_{lvl}"}
    assert expected.issubset(set(df.columns))


def test_reproducible_with_same_seed():
    df1 = simulate_lob(n_steps=200, seed=42)
    df2 = simulate_lob(n_steps=200, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_diverge():
    df1 = simulate_lob(n_steps=200, seed=1)
    df2 = simulate_lob(n_steps=200, seed=2)
    assert not df1["mid_price"].equals(df2["mid_price"])


def test_book_is_never_crossed():
    """Best bid should always be strictly below best ask (no crossed book)."""
    df = simulate_lob(n_steps=2000, seed=7)
    assert (df["bid_px_1"] < df["ask_px_1"]).all()


def test_prices_and_sizes_are_positive():
    df = simulate_lob(n_steps=1000, n_levels=5, seed=3)
    for lvl in range(1, 6):
        assert (df[f"bid_px_{lvl}"] > 0).all()
        assert (df[f"ask_px_{lvl}"] > 0).all()
        assert (df[f"bid_sz_{lvl}"] > 0).all()
        assert (df[f"ask_sz_{lvl}"] > 0).all()


def test_mid_price_stays_near_start_price_short_run():
    """Sanity bound: over a short run the walk shouldn't blow up or hit zero."""
    df = simulate_lob(n_steps=1000, start_price=100.0, seed=5)
    assert df["mid_price"].min() > 50.0
    assert df["mid_price"].max() < 200.0


def test_some_trades_occur():
    df = simulate_lob(n_steps=2000, seed=9)
    assert df["trade_price"].notna().sum() > 0


@pytest.mark.parametrize("n_levels", [1, 3, 5, 10])
def test_variable_n_levels(n_levels):
    df = simulate_lob(n_steps=50, n_levels=n_levels, seed=1)
    assert f"bid_px_{n_levels}" in df.columns
    assert f"bid_px_{n_levels + 1}" not in df.columns
