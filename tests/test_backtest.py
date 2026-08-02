import numpy as np
import pandas as pd
import pytest

from src.backtest import estimate_slippage, run_backtest, compute_stats


def test_estimate_slippage_increases_with_order_size():
    small = estimate_slippage(order_size=10, top_of_book_depth=1000, spread=0.02)
    large = estimate_slippage(order_size=500, top_of_book_depth=1000, spread=0.02)
    assert large > small


def test_estimate_slippage_decreases_with_depth():
    thin = estimate_slippage(order_size=100, top_of_book_depth=100, spread=0.02)
    deep = estimate_slippage(order_size=100, top_of_book_depth=100_000, spread=0.02)
    assert thin > deep


def test_estimate_slippage_passive_cheaper_than_aggressive():
    aggressive = estimate_slippage(order_size=100, top_of_book_depth=1000, spread=0.02, spread_cost_mult=1.0)
    passive = estimate_slippage(order_size=100, top_of_book_depth=1000, spread=0.02, spread_cost_mult=0.0)
    assert passive < aggressive
    # difference should be exactly the half-spread
    assert np.isclose(aggressive - passive, 0.01)


def _make_flat_lob(n=200, price=100.0, spread=0.02, depth=500.0):
    idx = pd.RangeIndex(n)
    return pd.DataFrame(
        {
            "mid_price": price,
            "bid_px_1": price - spread / 2,
            "ask_px_1": price + spread / 2,
            "bid_sz_1": depth,
            "ask_sz_1": depth,
        },
        index=idx,
    )


def test_run_backtest_no_signal_produces_no_trades():
    lob = _make_flat_lob()
    signal = pd.Series(0, index=lob.index)
    result = run_backtest(lob, signal, horizon=10)
    assert result["trades"].empty
    assert result["stats"] == {"n_trades": 0}


def test_run_backtest_flat_price_long_signal_loses_only_costs():
    """
    With a perfectly flat mid-price, a long trade should lose exactly the
    round-trip transaction cost (fees + slippage), never make money.
    """
    lob = _make_flat_lob(n=100)
    signal = pd.Series(0, index=lob.index)
    signal.iloc[5] = 1  # single long entry
    result = run_backtest(lob, signal, horizon=10, order_size=50, fee_bps=1.0, impact_coef=0.1)
    assert result["stats"]["n_trades"] == 1
    assert result["stats"]["total_pnl"] < 0  # costs only, price never moved


def test_run_backtest_passive_beats_aggressive_on_flat_market():
    lob = _make_flat_lob(n=100)
    signal = pd.Series(0, index=lob.index)
    signal.iloc[5] = 1
    aggr = run_backtest(lob, signal, horizon=10, execution="aggressive")
    passive = run_backtest(lob, signal, horizon=10, execution="passive")
    assert passive["stats"]["total_pnl"] > aggr["stats"]["total_pnl"]


def test_compute_stats_empty_trades_returns_zero_dict():
    stats = compute_stats(pd.DataFrame(), pd.Series(dtype=float))
    assert stats == {"n_trades": 0}


def test_compute_stats_win_rate_and_profit_factor():
    trades = pd.DataFrame({"pnl": [10.0, -5.0, 20.0, -2.0]})
    equity = pd.Series([10, 5, 25, 23])
    stats = compute_stats(trades, equity)
    assert stats["n_trades"] == 4
    assert stats["win_rate"] == 0.5
    assert np.isclose(stats["profit_factor"], 30 / 7)


def test_min_hold_prevents_overlapping_entries():
    lob = _make_flat_lob(n=100)
    signal = pd.Series(0, index=lob.index)
    signal.iloc[5] = 1
    signal.iloc[6] = 1  # would open a second position immediately after
    result = run_backtest(lob, signal, horizon=20, min_hold=10)
    # min_hold should suppress the second entry since position is still open
    assert result["stats"]["n_trades"] <= 1
