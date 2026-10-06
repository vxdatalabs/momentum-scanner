"""
Build the candidate universe from Webull's screener endpoints.

Rather than pulling bars for every listed US equity (thousands of symbols
against a per-second rate limit), we seed from the ranked lists Webull already
computes, then do the real work on the union. For swing horizons the useful
seeds are the 1-month and 3-month movers on both sides, symbols pressing 52-week
extremes, and the relative-volume leaders that flag where participation moved.

The union typically lands in the 300-700 symbol range, which is a few minutes of
batched bar requests and a very manageable scoring pool.
"""

from __future__ import annotations

import logging

import pandas as pd

from .wb import WebullData, EntitlementError

log = logging.getLogger(__name__)

# (method, kwargs, label, which side it feeds)
LONG_SEEDS = [
    ("gainers_losers", dict(rank_type="MONTH_1", direction="DESC"), "1m gainers"),
    ("gainers_losers", dict(rank_type="MONTH_3", direction="DESC"), "3m gainers"),
    ("gainers_losers", dict(rank_type="DAY_5", direction="DESC"), "5d gainers"),
    ("week52_hl", dict(rank_type="NEW_HIGH"), "52w new highs"),
    ("week52_hl", dict(rank_type="NEAR_HIGH"), "52w near highs"),
]

SHORT_SEEDS = [
    ("gainers_losers", dict(rank_type="MONTH_1", direction="ASC"), "1m losers"),
    ("gainers_losers", dict(rank_type="MONTH_3", direction="ASC"), "3m losers"),
    ("gainers_losers", dict(rank_type="DAY_5", direction="ASC"), "5d losers"),
    ("week52_hl", dict(rank_type="NEW_LOW"), "52w new lows"),
    ("week52_hl", dict(rank_type="NEAR_LOW"), "52w near lows"),
]

BOTH_SEEDS = [
    ("most_active", dict(rank_type="RELATIVE_VOLUME_10D"), "relative volume"),
    ("most_active", dict(rank_type="TURNOVER"), "turnover"),
]


def build(wb: WebullData, side: str, extra_symbols: list[str] | None = None) -> pd.DataFrame:
    """
    Returns a frame of unique symbols with the seed lists each came from.
    `side` is "long", "short", or "both".
    """
    seeds = list(BOTH_SEEDS)
    if side in ("long", "both"):
        seeds += LONG_SEEDS
    if side in ("short", "both"):
        seeds += SHORT_SEEDS

    frames = []
    for method, kwargs, label in seeds:
        try:
            df = getattr(wb, method)(**kwargs)
        except EntitlementError:
            raise
        except Exception as exc:
            log.warning("seed %-16s failed: %s", label, exc)
            continue
        if df.empty:
            log.warning("seed %-16s returned nothing", label)
            continue
        df = df.assign(seed=label)
        log.info("seed %-16s -> %4d symbols", label, len(df))
        frames.append(df)

    if not frames and not extra_symbols:
        return pd.DataFrame(columns=["symbol", "seeds", "seed_count"])

    if frames:
        allrows = pd.concat(frames, ignore_index=True)
    else:
        allrows = pd.DataFrame(columns=["symbol", "seed"])

    if extra_symbols:
        allrows = pd.concat([
            allrows,
            pd.DataFrame({"symbol": [s.upper() for s in extra_symbols], "seed": "manual"}),
        ], ignore_index=True)

    allrows["symbol"] = allrows["symbol"].str.upper().str.strip()
    allrows = allrows[allrows["symbol"].str.match(r"^[A-Z][A-Z.\-]{0,5}$", na=False)]

    agg = {
        "seeds": ("seed", lambda s: ", ".join(sorted(set(s)))),
        "seed_count": ("seed", "nunique"),
    }
    for col in ("wb_market_value", "wb_rvol_10d"):
        if col in allrows.columns:
            agg[col] = (col, "max")

    grouped = allrows.groupby("symbol").agg(**agg).reset_index()
    return grouped.sort_values("seed_count", ascending=False).reset_index(drop=True)
