"""
Offline demo mode.

Webull's OpenAPI application takes a day or two to approve, and the market-data
subscription is separate again. `--demo` runs the entire pipeline against
synthetic bars and synthetic option quotes so you can see the output shape, tune
weights and gates, and confirm the install works before any of that lands.

Nothing here touches the network. The numbers are fabricated.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

TICKERS = [
    # (symbol, daily drift, daily vol, start price)
    ("ARCX", 0.0085, 0.024, 45), ("BRTN", 0.0070, 0.021, 120),
    ("CLDQ", 0.0060, 0.030, 78),  ("DYNV", 0.0045, 0.019, 210),
    ("EMBR", 0.0030, 0.026, 33),  ("FLUX", 0.0015, 0.022, 95),
    ("GRVT", 0.0002, 0.020, 150), ("HALO", -0.0010, 0.023, 64),
    ("IONX", -0.0030, 0.027, 88), ("JUNO", -0.0045, 0.021, 175),
    ("KRAS", -0.0060, 0.025, 240),("LUMN", -0.0080, 0.029, 310),
    ("MTRX", 0.0055, 0.018, 58),  ("NOVA", 0.0040, 0.033, 142),
    ("ORBT", -0.0050, 0.020, 199),("PYLN", 0.0025, 0.028, 72),
]


def bars(n: int = 300, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    out = {}
    for sym, drift, vol, start in TICKERS:
        logret = drift + rng.normal(0, vol, n)
        close = start * np.exp(np.cumsum(logret))
        intraday = np.abs(rng.normal(0, vol * 0.45, n))
        out[sym] = pd.DataFrame({
            "date": dates,
            "open": np.concatenate([[start], close[:-1]]),
            "high": close * (1 + intraday),
            "low": close * (1 - intraday),
            "close": close,
            "volume": rng.integers(1_500_000, 12_000_000, n).astype(float),
        })
    return out


class DemoClient:
    """Stands in for WebullData. Only the methods the scan path uses."""

    def __init__(self, seed: int = 42):
        self._bars = bars(seed=seed)
        self.rng = np.random.default_rng(seed + 1)
        self.last_raw: dict = {}

    def daily_bars(self, symbols, count=300):
        return {s: df for s, df in self._bars.items() if s in set(symbols)}

    def option_contracts(self, underlying, start, end, strike_low, strike_high):
        expiry = pd.Timestamp(date.today() + timedelta(days=35))
        mid_strike = round((strike_low + strike_high) / 2)
        rows = []
        for offset in (-2, 0, 2):
            strike = max(round(mid_strike * (1 + offset * 0.025), 1), 1.0)
            for kind in ("CALL", "PUT"):
                tag = "C" if kind == "CALL" else "P"
                rows.append({
                    "option_symbol": f"{underlying}{expiry:%y%m%d}{tag}{int(strike*1000):08d}",
                    "underlying": underlying,
                    "strike": strike,
                    "expiry": expiry,
                    "option_type": kind,
                })
        return pd.DataFrame(rows)

    def option_snapshot(self, option_symbols):
        rows = []
        for sym in option_symbols:
            mid = float(self.rng.uniform(1.5, 12.0))
            spread = float(self.rng.uniform(0.01, 0.12)) * mid
            rows.append({
                "option_symbol": sym,
                "opt_bid": round(mid - spread / 2, 2),
                "opt_ask": round(mid + spread / 2, 2),
                "opt_last": round(mid, 2),
                "opt_volume": float(self.rng.integers(50, 6000)),
                "open_interest": float(self.rng.integers(40, 15000)),
                "iv": float(self.rng.uniform(0.28, 0.95)),
                "delta": float(self.rng.uniform(0.35, 0.60)),
            })
        return pd.DataFrame(rows)
