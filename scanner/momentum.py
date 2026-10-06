"""
Swing-momentum factor computation and cross-sectional ranking.

Design notes
------------
Factors are computed per symbol from daily bars, then z-scored *across the
candidate pool* before weighting. Cross-sectional normalization is what makes a
composite meaningful: a 12% 21-day return is strong in a flat tape and
unremarkable in a melt-up, and z-scoring against today's pool captures that.

The trend-quality factor is an exponential-regression slope times R-squared
(annualized). Raw return alone rewards a single gap; slope-times-R2 rewards
symbols that ground steadily higher, which is what you want when you're holding
an option through several sessions of theta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --- primitives --------------------------------------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def exp_slope_r2(closes: pd.Series, window: int = 90) -> tuple[float, float]:
    """
    Annualized exponential regression slope and its R-squared.

    Fits log(price) ~ a + b*t over the window, annualizes as exp(b)^252 - 1.
    """
    y = np.log(closes.tail(window).to_numpy(dtype=float))
    if len(y) < max(20, window // 3) or not np.all(np.isfinite(y)):
        return float("nan"), float("nan")
    x = np.arange(len(y), dtype=float)
    b, a = np.polyfit(x, y, 1)
    fit = a + b * x
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    annualized = float(np.exp(b * TRADING_DAYS) - 1.0)
    return annualized, float(max(r2, 0.0))


def pct_return(closes: pd.Series, n: int) -> float:
    if len(closes) <= n:
        return float("nan")
    a, b = float(closes.iloc[-1 - n]), float(closes.iloc[-1])
    return (b / a - 1.0) if a > 0 else float("nan")


def hist_vol(closes: pd.Series, n: int = 20) -> float:
    """Annualized close-to-close volatility."""
    rets = np.log(closes / closes.shift(1)).tail(n)
    if rets.notna().sum() < max(5, n // 2):
        return float("nan")
    return float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS))


# --- per-symbol factor block -------------------------------------------------

def factors(df: pd.DataFrame) -> dict:
    """Compute the raw factor block for one symbol's daily bars."""
    if df is None or len(df) < 60:
        return {}

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    last = float(close.iloc[-1])

    ema20 = ema(close, 20)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean() if len(close) >= 200 else pd.Series(
        [np.nan] * len(close), index=close.index
    )

    a = atr(df, 14)
    atr_last = float(a.iloc[-1])
    atr_pct = atr_last / last if last > 0 else np.nan

    slope_ann, r2 = exp_slope_r2(close, 90)
    slope_q = slope_ann * r2 if np.isfinite(slope_ann) and np.isfinite(r2) else np.nan

    lookback_52w = close.tail(252)
    high_52w = float(lookback_52w.max())
    low_52w = float(lookback_52w.min())

    avg_vol_20 = float(volume.tail(20).mean())
    avg_vol_60 = float(volume.tail(60).mean()) if len(volume) >= 60 else avg_vol_20
    rvol = avg_vol_20 / avg_vol_60 if avg_vol_60 > 0 else np.nan
    dollar_vol = avg_vol_20 * last

    # How stretched from the 20-EMA, in ATR units. Signed: + is above.
    ext = (last - float(ema20.iloc[-1])) / atr_last if atr_last > 0 else np.nan

    above = [
        last > float(ema20.iloc[-1]),
        last > float(sma50.iloc[-1]) if np.isfinite(sma50.iloc[-1]) else False,
        float(ema20.iloc[-1]) > float(sma50.iloc[-1]) if np.isfinite(sma50.iloc[-1]) else False,
        last > float(sma200.iloc[-1]) if np.isfinite(sma200.iloc[-1]) else False,
    ]
    trend_align = sum(1 for x in above if x) / len(above)

    # Fraction of the last 20 sessions closing up — steadiness of participation.
    up_days = float((close.diff().tail(20) > 0).mean())

    # Max single-day gap in the window; a high value means the "trend" is one print.
    gaps = (df["open"] / df["close"].shift(1) - 1.0).abs().tail(21)
    max_gap = float(gaps.max()) if gaps.notna().any() else np.nan

    return {
        "price": last,
        "ret_5d": pct_return(close, 5),
        "ret_21d": pct_return(close, 21),
        "ret_63d": pct_return(close, 63),
        "ret_126d": pct_return(close, 126),
        "slope_ann": slope_ann,
        "slope_r2": r2,
        "slope_q": slope_q,
        "trend_align": trend_align,
        "atr_pct": atr_pct,
        "hv20": hist_vol(close, 20),
        "hv60": hist_vol(close, 60),
        "rvol": rvol,
        "avg_dollar_vol": dollar_vol,
        "pct_from_52w_high": (last / high_52w - 1.0) if high_52w > 0 else np.nan,
        "pct_from_52w_low": (last / low_52w - 1.0) if low_52w > 0 else np.nan,
        "ext_atr": ext,
        "up_days_20": up_days,
        "max_gap_21d": max_gap,
        "bars": len(df),
    }


def build_factor_frame(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sym, df in bars.items():
        f = factors(df)
        if f:
            rows.append({"symbol": sym, **f})
    return pd.DataFrame(rows)


# --- gating ------------------------------------------------------------------

def apply_gates(fr: pd.DataFrame, g: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the frame into (passing, rejected-with-reason)."""
    if fr.empty:
        return fr, fr.assign(reject_reason=pd.Series(dtype=str))

    checks = {
        "price_too_low": fr["price"] < g["min_price"],
        "price_too_high": fr["price"] > g["max_price"],
        "illiquid": fr["avg_dollar_vol"] < g["min_avg_dollar_volume"],
        "too_quiet": fr["atr_pct"] < g["min_atr_pct"],
        "too_wild": fr["atr_pct"] > g["max_atr_pct"],
        "insufficient_history": fr["bars"] < g["min_bars"],
        "gap_driven": fr["max_gap_21d"] > g["max_single_gap"],
    }
    reason = pd.Series("", index=fr.index)
    for name, failed in checks.items():
        failed = failed.fillna(True)
        reason = reason.where(~(failed & (reason == "")), name)

    passing = fr[reason == ""].copy()
    rejected = fr[reason != ""].copy()
    rejected["reject_reason"] = reason[reason != ""]
    return passing.reset_index(drop=True), rejected.reset_index(drop=True)


# --- scoring -----------------------------------------------------------------

def zscore(s: pd.Series, clip: float = 3.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).clip(-clip, clip).fillna(0.0)


def score(fr: pd.DataFrame, side: str, weights: dict,
          extension_penalty_atr: float = 4.0) -> pd.DataFrame:
    """
    Composite momentum score for one side.

    `side` is "long" or "short". For shorts every directional factor flips sign,
    so a short score of 2.0 means the same strength of downside structure that a
    long score of 2.0 means on the upside. Volume factors do not flip —
    participation is good evidence either way.
    """
    if fr.empty:
        return fr.assign(score=pd.Series(dtype=float))

    out = fr.copy()
    sign = 1.0 if side == "long" else -1.0

    z = {
        "slope_q": zscore(out["slope_q"] * sign),
        "ret_21d": zscore(out["ret_21d"] * sign),
        "ret_63d": zscore(out["ret_63d"] * sign),
        "rvol": zscore(out["rvol"]),
        "up_days_20": zscore((out["up_days_20"] - 0.5) * sign),
    }

    if side == "long":
        z["trend_align"] = zscore(out["trend_align"])
        # Nearer the 52w high is better; the value is <= 0, so higher is closer.
        z["range_position"] = zscore(out["pct_from_52w_high"])
    else:
        z["trend_align"] = zscore(1.0 - out["trend_align"])
        # Nearer the 52w low is better; the value is >= 0, so lower is closer.
        z["range_position"] = zscore(-out["pct_from_52w_low"])

    composite = pd.Series(0.0, index=out.index)
    total_w = 0.0
    for key, w in weights.items():
        if key in z:
            composite += z[key] * w
            total_w += w
            out[f"z_{key}"] = z[key]
    if total_w > 0:
        composite /= total_w

    # Penalize chasing: distance from the 20-EMA beyond the threshold, in ATR
    # units, scaled at 0.25 score points per excess ATR.
    stretch = (out["ext_atr"] * sign).fillna(0.0)
    excess = (stretch - extension_penalty_atr).clip(lower=0.0)
    out["extension_penalty"] = excess * 0.25
    out["score"] = composite - out["extension_penalty"]
    out["side"] = side

    return out.sort_values("score", ascending=False).reset_index(drop=True)
