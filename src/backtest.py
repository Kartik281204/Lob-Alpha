"""
backtest.py
============
Event-driven backtest for a directional signal derived from LOB features,
with a realistic transaction cost model:

- Fees: flat per-share/contract commission (bps of notional).
- Slippage: modeled as a function of order size relative to top-of-book
  depth (larger orders walk the book further) plus a fixed half-spread
  cost for crossing the market on entry/exit.

Signal convention: prediction in {-1, 0, 1} (short / flat / long).
Position is held for `horizon` steps (matching the label horizon used in
features.make_labels) then closed, unless a new signal overrides it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_slippage(
    order_size: float, top_of_book_depth: float, spread: float,
    impact_coef: float = 0.1, spread_cost_mult: float = 1.0,
) -> float:
    """
    Simple square-root market impact model (Almgren-Chriss style):
    slippage = spread_cost_mult * half_spread + impact_coef * spread * sqrt(order_size / depth)

    spread_cost_mult=1.0 (aggressive/taker): pays the full half-spread to
    cross the market on top of impact.
    spread_cost_mult=0.0 (passive/maker): posts at the touch, pays only
    market impact (assumes the resting order gets filled -- a simplification
    that ignores fill-probability / adverse-selection risk of passive
    orders, which real market-making systems model explicitly).
    """
    half_spread = spread / 2
    size_ratio = order_size / max(top_of_book_depth, 1.0)
    impact = impact_coef * spread * np.sqrt(size_ratio)
    return spread_cost_mult * half_spread + impact


def run_backtest(
    df: pd.DataFrame,
    predictions: pd.Series,
    horizon: int = 50,
    order_size: float = 100.0,
    fee_bps: float = 1.0,
    impact_coef: float = 0.1,
    min_hold: int = 5,
    execution: str = "aggressive",
) -> dict:
    """
    Simulate trading the signal on the synthetic/real LOB series.

    Parameters
    ----------
    df : the LOB dataframe (must have mid_price, bid/ask px & sz level 1)
    predictions : Series aligned to df.index with values in {-1, 0, 1}
    horizon : holding period in steps for each position
    order_size : order size used for slippage/impact estimation
    fee_bps : commission in basis points of notional, charged per fill
    impact_coef : market impact coefficient (see estimate_slippage)
    min_hold : minimum steps between opening new positions (avoid churn)
    execution : "aggressive" (taker) crosses the spread on every fill --
        pays half-spread + market impact each way, fills immediately.
        "passive" (maker) posts a limit order at the near touch and pays
        only market impact (no spread cost), reflecting a market-maker /
        resting-order execution style. Many microstructure signals whose
        edge is on the order of one spread are unprofitable taker-side but
        profitable maker-side -- comparing both is standard practice.

    Returns
    -------
    dict with keys: trades (DataFrame), equity_curve (Series), stats (dict)
    """
    spread_cost_mult = 1.0 if execution == "aggressive" else 0.0
    idx = df.index
    mid = df["mid_price"]
    spread = df["ask_px_1"] - df["bid_px_1"]
    bid_depth = df["bid_sz_1"]
    ask_depth = df["ask_sz_1"]

    trades = []
    equity = 0.0
    equity_curve = np.zeros(len(idx))
    pos = 0
    entry_price = None
    entry_i = None
    last_trade_i = -min_hold

    for i, t in enumerate(idx):
        sig = predictions.iloc[i]

        # close existing position at horizon
        if pos != 0 and entry_i is not None and (i - entry_i) >= horizon:
            exit_depth = bid_depth.iloc[i] if pos > 0 else ask_depth.iloc[i]
            slip = estimate_slippage(order_size, exit_depth, spread.iloc[i], impact_coef, spread_cost_mult)
            exit_price = mid.iloc[i] - pos * slip  # adverse fill
            fee = fee_bps / 1e4 * exit_price * order_size
            pnl = pos * (exit_price - entry_price) * order_size - fee
            equity += pnl
            trades.append(
                dict(entry_t=idx[entry_i], exit_t=t, side=pos, entry_price=entry_price,
                     exit_price=exit_price, pnl=pnl)
            )
            pos = 0
            entry_price = None
            entry_i = None

        # open new position on a nonzero signal, respecting min_hold spacing
        if pos == 0 and sig != 0 and (i - last_trade_i) >= min_hold:
            entry_depth = ask_depth.iloc[i] if sig > 0 else bid_depth.iloc[i]
            slip = estimate_slippage(order_size, entry_depth, spread.iloc[i], impact_coef, spread_cost_mult)
            entry_price = mid.iloc[i] + sig * slip  # adverse fill (pay the spread/impact)
            fee = fee_bps / 1e4 * entry_price * order_size
            equity -= fee
            pos = sig
            entry_i = i
            last_trade_i = i

        equity_curve[i] = equity

    trades_df = pd.DataFrame(trades)
    equity_series = pd.Series(equity_curve, index=idx)

    stats = compute_stats(trades_df, equity_series)
    return {"trades": trades_df, "equity_curve": equity_series, "stats": stats}


def compute_stats(trades: pd.DataFrame, equity_curve: pd.Series) -> dict:
    if trades.empty:
        return {"n_trades": 0}

    returns = equity_curve.diff().fillna(equity_curve.iloc[0])
    sharpe = (
        (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(len(returns))
        if returns.std() > 0
        else 0.0
    )
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_dd = drawdown.min()

    win_rate = (trades["pnl"] > 0).mean()
    avg_win = trades.loc[trades["pnl"] > 0, "pnl"].mean() if (trades["pnl"] > 0).any() else 0.0
    avg_loss = trades.loc[trades["pnl"] < 0, "pnl"].mean() if (trades["pnl"] < 0).any() else 0.0
    profit_factor = (
        trades.loc[trades["pnl"] > 0, "pnl"].sum() / abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
        if (trades["pnl"] < 0).any()
        else np.inf
    )

    return {
        "n_trades": len(trades),
        "total_pnl": trades["pnl"].sum(),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe_like": sharpe,
        "max_drawdown": max_dd,
        "final_equity": equity_curve.iloc[-1],
    }
