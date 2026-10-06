"""
Options tradability layer.

A high momentum score on an underlying whose options are 18% wide with 40 open
interest is not a tradable idea. This module answers three questions per
candidate:

1. Can you get filled?     -> bid/ask spread as a percentage of mid, open interest
2. What does it cost?      -> ATM debit as a percentage of the underlying
3. Buy or sell premium?    -> implied vol against the underlying's realized vol

Point 3 is the one that most often flips a decision. A clean uptrend with IV at
twice realized vol is an argument for a spread rather than a long call, because
you are paying a large premium for movement the stock has not historically
delivered.

Webull gates options data behind a separate market-data entitlement. If the
account is not entitled, every function here degrades gracefully: the scan still
produces its momentum ranking, each row is flagged `options: unavailable`, and
the realized-vol column still gives you a volatility reference.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .wb import WebullData, EntitlementError, WebullDataError

log = logging.getLogger(__name__)


def _target_window(dte_min: int, dte_max: int) -> tuple[str, str]:
    today = date.today()
    return (
        (today + timedelta(days=dte_min)).isoformat(),
        (today + timedelta(days=dte_max)).isoformat(),
    )


def _mid(bid: float, ask: float, last: float) -> float:
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return last if np.isfinite(last) else np.nan


def _spread_pct(bid: float, ask: float) -> float:
    if not (np.isfinite(bid) and np.isfinite(ask)) or ask <= 0 or bid <= 0:
        return np.nan
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid if mid > 0 else np.nan


def evaluate(wb: WebullData, symbol: str, spot: float, side: str,
             cfg: dict, realized_vol: float = np.nan) -> dict:
    """
    Evaluate the near-the-money contract for one underlying.

    Returns a dict of option metrics, or a dict with `opt_status` explaining why
    metrics are absent. Never raises for a single bad symbol.
    """
    blank = {
        "opt_status": "unavailable",
        "opt_symbol": None,
        "opt_expiry": None,
        "opt_strike": np.nan,
        "opt_mid": np.nan,
        "opt_spread_pct": np.nan,
        "opt_oi": np.nan,
        "opt_volume": np.nan,
        "opt_iv": np.nan,
        "iv_hv_ratio": np.nan,
        "debit_pct_of_spot": np.nan,
        "opt_verdict": "",
    }

    if not np.isfinite(spot) or spot <= 0:
        return {**blank, "opt_status": "no spot price"}

    dte_min, dte_max = cfg["dte_min"], cfg["dte_max"]
    start, end = _target_window(dte_min, dte_max)
    band = cfg.get("strike_band_pct", 0.08)

    try:
        contracts = wb.option_contracts(
            symbol, start, end,
            strike_low=round(spot * (1 - band), 2),
            strike_high=round(spot * (1 + band), 2),
        )
    except EntitlementError:
        raise
    except WebullDataError as exc:
        log.debug("%s: contract lookup failed: %s", symbol, exc)
        return {**blank, "opt_status": "contract lookup failed"}

    if contracts.empty:
        return {**blank, "opt_status": "no contracts in window"}

    want = "CALL" if side == "long" else "PUT"
    subset = contracts[contracts["option_type"].str.startswith(want[0], na=False)]
    if subset.empty:
        subset = contracts

    # Nearest strike to spot, then nearest expiry to the midpoint of the window.
    target_dte = (dte_min + dte_max) / 2
    target_expiry = pd.Timestamp(date.today() + timedelta(days=int(target_dte)))
    subset = subset.assign(
        strike_gap=(subset["strike"] - spot).abs(),
        expiry_gap=(subset["expiry"] - target_expiry).abs(),
    ).sort_values(["expiry_gap", "strike_gap"])

    picks = subset.head(cfg.get("candidates_per_symbol", 3))
    try:
        snaps = wb.option_snapshot(picks["option_symbol"].tolist())
    except EntitlementError:
        raise
    except WebullDataError as exc:
        log.debug("%s: option snapshot failed: %s", symbol, exc)
        return {**blank, "opt_status": "snapshot failed"}

    if snaps.empty:
        return {**blank, "opt_status": "no quotes returned"}

    merged = picks.merge(snaps, on="option_symbol", how="inner")
    if merged.empty:
        return {**blank, "opt_status": "no quotes returned"}

    merged["mid"] = [
        _mid(r.opt_bid, r.opt_ask, r.opt_last) for r in merged.itertuples()
    ]
    merged["spread_pct"] = [
        _spread_pct(r.opt_bid, r.opt_ask) for r in merged.itertuples()
    ]

    # Prefer the tightest quote among the picks; open interest breaks ties.
    merged = merged.sort_values(
        ["spread_pct", "open_interest"], ascending=[True, False], na_position="last"
    )
    best = merged.iloc[0]

    iv = float(best.get("iv", np.nan))
    if np.isfinite(iv) and iv > 3:  # some feeds return percent, not decimal
        iv = iv / 100.0
    ratio = iv / realized_vol if np.isfinite(iv) and np.isfinite(realized_vol) and realized_vol > 0 else np.nan

    mid = float(best["mid"])
    debit_pct = mid / spot if np.isfinite(mid) and spot > 0 else np.nan

    result = {
        "opt_status": "ok",
        "opt_symbol": best["option_symbol"],
        "opt_expiry": best["expiry"].date().isoformat() if pd.notna(best["expiry"]) else None,
        "opt_strike": float(best["strike"]),
        "opt_mid": mid,
        "opt_spread_pct": float(best["spread_pct"]) if np.isfinite(best["spread_pct"]) else np.nan,
        "opt_oi": float(best["open_interest"]) if np.isfinite(best["open_interest"]) else np.nan,
        "opt_volume": float(best["opt_volume"]) if np.isfinite(best["opt_volume"]) else np.nan,
        "opt_iv": iv,
        "iv_hv_ratio": ratio,
        "debit_pct_of_spot": debit_pct,
    }
    result["opt_verdict"] = verdict(result, cfg)
    return result


def verdict(m: dict, cfg: dict) -> str:
    """One-line read on structure, based on liquidity and vol pricing."""
    notes = []
    spread, oi = m.get("opt_spread_pct"), m.get("opt_oi")

    if np.isfinite(spread) and spread > cfg["max_spread_pct"]:
        notes.append(f"wide ({spread:.0%})")
    if np.isfinite(oi) and oi < cfg["min_open_interest"]:
        notes.append(f"thin OI ({oi:.0f})")

    ratio = m.get("iv_hv_ratio")
    if np.isfinite(ratio):
        if ratio >= cfg.get("iv_rich_ratio", 1.35):
            notes.append(f"IV rich {ratio:.2f}x HV — favor spreads")
        elif ratio <= cfg.get("iv_cheap_ratio", 0.85):
            notes.append(f"IV cheap {ratio:.2f}x HV — long premium ok")
        else:
            notes.append(f"IV fair {ratio:.2f}x HV")

    return "; ".join(notes) if notes else "liquid"


def tradable(m: dict, cfg: dict) -> bool:
    """Hard filter for the tradable column."""
    if m.get("opt_status") != "ok":
        return False
    spread, oi = m.get("opt_spread_pct"), m.get("opt_oi")
    if np.isfinite(spread) and spread > cfg["max_spread_pct"]:
        return False
    if np.isfinite(oi) and oi < cfg["min_open_interest"]:
        return False
    return True


def enrich(wb: WebullData, ranked: pd.DataFrame, side: str, cfg: dict,
           limit: int) -> pd.DataFrame:
    """Attach option metrics to the top `limit` rows of a ranked frame."""
    if ranked.empty:
        return ranked

    head = ranked.head(limit).copy()
    records = []
    entitled = True

    for row in head.itertuples():
        if not entitled:
            records.append({"opt_status": "not entitled"})
            continue
        try:
            records.append(
                evaluate(wb, row.symbol, float(row.price), side, cfg,
                         realized_vol=float(getattr(row, "hv20", np.nan)))
            )
        except EntitlementError as exc:
            log.warning("Options data not entitled on this account: %s", exc)
            log.warning("Continuing with momentum ranking only.")
            entitled = False
            records.append({"opt_status": "not entitled"})
        except Exception as exc:
            log.debug("%s: option evaluation error: %s", row.symbol, exc)
            records.append({"opt_status": f"error: {type(exc).__name__}"})

    opt = pd.DataFrame(records, index=head.index)
    head = pd.concat([head, opt], axis=1)
    head["opt_tradable"] = [tradable(r, cfg) for r in opt.to_dict("records")]

    tail = ranked.iloc[limit:].copy()
    if not tail.empty:
        tail["opt_status"] = "not evaluated"
        tail["opt_tradable"] = False

    return pd.concat([head, tail], ignore_index=True)
