"""
simulate_lob.py
================
Generates synthetic high-frequency limit order book (LOB) data.

Real tick-level LOB data (e.g. LOBSTER, NASDAQ ITCH) is either paywalled or
requires exchange membership, so this module produces a statistically
realistic synthetic order book using a standard microstructure simulation:

- Mid-price follows a jump-diffusion process (captures volatility bursts).
- Order arrivals (limit, market, cancel) follow independent Poisson processes
  at each of `n_levels` price levels on both sides of the book, per the
  Cont-Stoikov-Talreja / Avellaneda-Stoikov style queueing model.
- Trade flow is derived from market orders that cross the spread.

Swap this module out for a real data loader (LOBSTER CSVs, ITCH parser,
exchange WebSocket capture) without touching features.py / models.py —
they only require the standardized output schema below.

Output schema (one row per event / snapshot):
    timestamp, mid_price,
    bid_px_1..N, bid_sz_1..N, ask_px_1..N, ask_sz_1..N,
    trade_price, trade_size, trade_side  (NaN if no trade this step)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_lob(
    n_steps: int = 50_000,
    n_levels: int = 5,
    tick_size: float = 0.01,
    start_price: float = 100.00,
    seed: int = 42,
    base_intensity: float = 1.0,
    jump_prob: float = 0.002,
    jump_scale: float = 0.20,
) -> pd.DataFrame:
    """
    Simulate a synthetic multi-level limit order book.

    Parameters
    ----------
    n_steps : number of discrete time steps (events) to simulate
    n_levels : depth of book to simulate on each side
    tick_size : minimum price increment
    start_price : initial mid price
    seed : RNG seed for reproducibility
    base_intensity : baseline Poisson intensity for order arrivals
    jump_prob : probability per step of a volatility "burst" (regime shift)
    jump_scale : multiplier applied to intensity during a burst

    Returns
    -------
    pd.DataFrame indexed by timestamp with the schema described in the
    module docstring.
    """
    rng = np.random.default_rng(seed)

    mid = start_price
    intensity_state = base_intensity  # time-varying activity level (vol clustering)

    rows = []
    # book state: size resting at each of n_levels on bid/ask
    bid_sizes = rng.integers(50, 500, size=n_levels).astype(float)
    ask_sizes = rng.integers(50, 500, size=n_levels).astype(float)

    for t in range(n_steps):
        # --- volatility regime (bursts) ---
        if rng.random() < jump_prob:
            intensity_state = base_intensity * (1 + jump_scale * rng.uniform(2, 5))
        else:
            # mean-revert intensity back to baseline
            intensity_state += (base_intensity - intensity_state) * 0.05

        # --- order-flow-imbalance effect (the "alpha" a model can learn) ---
        # Real markets show short-term predictability: heavy resting bid size
        # relative to ask tends to precede small upward drift, and vice versa
        # (Cont, Kukanov & Stoikov 2014). We inject a small, noisy version of
        # this effect so the simulated data has genuine, learnable signal --
        # this is a "known-answer" sanity check for the pipeline, not a claim
        # about real markets.
        imb = (bid_sizes[0] - ask_sizes[0]) / (bid_sizes[0] + ask_sizes[0])
        imbalance_drift = 0.35 * tick_size * imb

        # --- mid price innovation: jump-diffusion + imbalance drift ---
        diffusion = rng.normal(0, tick_size * 0.5 * intensity_state)
        jump = 0.0
        if rng.random() < jump_prob:
            jump = rng.choice([-1, 1]) * jump_scale * rng.exponential(tick_size * 3)
        mid = max(tick_size, mid + diffusion + jump + imbalance_drift)
        mid = round(mid / tick_size) * tick_size

        # --- order book depth dynamics: Poisson arrivals/cancels per level ---
        lam = intensity_state
        bid_sizes += rng.poisson(lam * 8, size=n_levels) - rng.poisson(lam * 7, size=n_levels)
        ask_sizes += rng.poisson(lam * 8, size=n_levels) - rng.poisson(lam * 7, size=n_levels)
        bid_sizes = np.clip(bid_sizes, 10, None)
        ask_sizes = np.clip(ask_sizes, 10, None)

        half_spread = tick_size * (1 + rng.exponential(0.5))
        bid_px = mid - half_spread - np.arange(n_levels) * tick_size
        ask_px = mid + half_spread + np.arange(n_levels) * tick_size

        # --- trade flow: market orders crossing the spread ---
        trade_price = np.nan
        trade_size = np.nan
        trade_side = np.nan
        if rng.random() < 0.30 * intensity_state / base_intensity:
            side = rng.choice(["buy", "sell"])
            size = float(rng.integers(1, 200))
            if side == "buy":
                trade_price = ask_px[0]
                ask_sizes[0] = max(10.0, ask_sizes[0] - size)
            else:
                trade_price = bid_px[0]
                bid_sizes[0] = max(10.0, bid_sizes[0] - size)
            trade_size = size
            trade_side = 1.0 if side == "buy" else -1.0

        row = {"timestamp": t, "mid_price": mid}
        for lvl in range(n_levels):
            row[f"bid_px_{lvl+1}"] = bid_px[lvl]
            row[f"bid_sz_{lvl+1}"] = bid_sizes[lvl]
            row[f"ask_px_{lvl+1}"] = ask_px[lvl]
            row[f"ask_sz_{lvl+1}"] = ask_sizes[lvl]
        row["trade_price"] = trade_price
        row["trade_size"] = trade_size
        row["trade_side"] = trade_side
        rows.append(row)

    df = pd.DataFrame(rows).set_index("timestamp")
    return df


if __name__ == "__main__":
    df = simulate_lob(n_steps=5_000)
    print(df.head())
    print(f"\nGenerated {len(df):,} LOB snapshots")
