"""
Run with:  python -m pytest tests/ -q

These cover the parts that don't need network: the factor math, the gates, the
scoring symmetry between sides, and the response normalizers.
"""

import numpy as np
import pandas as pd
import pytest

from scanner import momentum, options
from scanner.wb import pick, rows_of, _num, _to_ts


def synth(n=300, daily_drift=0.001, vol=0.01, start=100.0, seed=0):
    rng = np.random.default_rng(seed)
    logret = daily_drift + rng.normal(0, vol, n)
    close = start * np.exp(np.cumsum(logret))
    return pd.DataFrame({
        "date": pd.bdate_range("2025-01-02", periods=n),
        "open": np.concatenate([[start], close[:-1]]),
        "high": close * (1 + abs(rng.normal(0, 0.004, n))),
        "low": close * (1 - abs(rng.normal(0, 0.004, n))),
        "close": close,
        "volume": rng.integers(3_000_000, 8_000_000, n).astype(float),
    })


# --- factor math -------------------------------------------------------------

def test_exp_slope_matches_analytic_value():
    closes = pd.Series(100 * np.exp(0.001 * np.arange(200)))
    slope, r2 = momentum.exp_slope_r2(closes, 90)
    assert slope == pytest.approx(np.exp(0.001 * 252) - 1, rel=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_exp_slope_sign_flips_on_downtrend():
    down = pd.Series(100 * np.exp(-0.001 * np.arange(200)))
    slope, _ = momentum.exp_slope_r2(down, 90)
    assert slope < 0


def test_noise_lowers_r2_without_destroying_slope():
    rng = np.random.default_rng(1)
    noisy = pd.Series(100 * np.exp(0.001 * np.arange(200) + rng.normal(0, 0.02, 200)))
    slope, r2 = momentum.exp_slope_r2(noisy, 90)
    assert 0.0 < r2 < 0.95
    assert slope > 0


def test_atr_is_positive_and_tracks_range():
    df = synth(seed=3)
    a = momentum.atr(df, 14)
    assert (a.dropna() > 0).all()
    assert a.iloc[-1] < df["close"].iloc[-1]  # sanity: ATR well below price


def test_short_history_returns_empty_factor_block():
    assert momentum.factors(synth(n=30)) == {}


def test_factor_block_has_finite_core_fields():
    f = momentum.factors(synth(seed=5))
    for key in ("price", "ret_21d", "ret_63d", "atr_pct", "rvol", "trend_align"):
        assert np.isfinite(f[key]), key


# --- gates -------------------------------------------------------------------

GATES = dict(min_price=10, max_price=800, min_avg_dollar_volume=25e6,
             min_atr_pct=0.015, max_atr_pct=0.15, min_bars=120, max_single_gap=0.25)


def test_gates_reject_penny_and_illiquid_names():
    bars = {
        "GOOD": synth(start=150, vol=0.02, seed=11),
        "PENNY": synth(start=2, vol=0.02, seed=12),
        "QUIET": synth(start=150, vol=0.001, seed=13),
    }
    fr = momentum.build_factor_frame(bars)
    passing, rejected = momentum.apply_gates(fr, GATES)
    reasons = dict(zip(rejected["symbol"], rejected["reject_reason"]))
    assert "GOOD" in set(passing["symbol"])
    assert reasons["PENNY"] == "price_too_low"
    assert reasons["QUIET"] == "too_quiet"


def test_gates_on_empty_frame_do_not_raise():
    passing, rejected = momentum.apply_gates(pd.DataFrame(), GATES)
    assert passing.empty and rejected.empty


# --- scoring -----------------------------------------------------------------

WEIGHTS = dict(slope_q=0.30, ret_21d=0.15, ret_63d=0.15, trend_align=0.15,
               range_position=0.10, rvol=0.10, up_days_20=0.05)


def _uptrend_downtrend_frame():
    # vol chosen so ATR% clears the min_atr_pct gate; drift dominates the noise.
    bars = {
        # Start prices chosen so that after 300 sessions of drift each path
        # still finishes inside the price gate.
        "UP": synth(daily_drift=0.009, vol=0.020, start=40, seed=21),
        "DOWN": synth(daily_drift=-0.009, vol=0.020, start=400, seed=22),
        "FLAT": synth(daily_drift=0.0, vol=0.020, start=100, seed=23),
    }
    fr = momentum.build_factor_frame(bars)
    passing, rejected = momentum.apply_gates(fr, {**GATES, "min_avg_dollar_volume": 0})
    assert len(passing) == 3, f"fixture gated out: {dict(zip(rejected.symbol, rejected.reject_reason))}"
    return passing.sort_values("symbol").reset_index(drop=True)


def test_uptrend_ranks_first_long_and_last_short():
    passing = _uptrend_downtrend_frame()
    longs = momentum.score(passing, "long", WEIGHTS)
    shorts = momentum.score(passing, "short", WEIGHTS)
    assert longs.iloc[0]["symbol"] == "UP"
    assert shorts.iloc[0]["symbol"] == "DOWN"
    assert longs.iloc[-1]["symbol"] == "DOWN"


def test_scores_are_sorted_descending():
    ranked = momentum.score(_uptrend_downtrend_frame(), "long", WEIGHTS)
    assert ranked["score"].is_monotonic_decreasing


def test_extension_penalty_applies_to_stretched_names():
    passing = _uptrend_downtrend_frame().copy()
    # Frame is sorted by symbol: DOWN, FLAT, UP. Stretch UP by 8 ATRs.
    passing.loc[:, "ext_atr"] = [0.0, 0.0, 8.0]
    ranked = momentum.score(passing, "long", WEIGHTS, extension_penalty_atr=4.0)
    penalized = ranked.set_index("symbol")["extension_penalty"]
    assert penalized.loc["UP"] == pytest.approx(1.0)   # (8 - 4) * 0.25
    assert penalized.loc["FLAT"] == 0.0


def test_extension_penalty_uses_the_correct_sign_for_shorts():
    passing = _uptrend_downtrend_frame().copy()
    passing.loc[:, "ext_atr"] = [-8.0, 0.0, 0.0]  # DOWN is 8 ATRs *below* its 20-EMA
    ranked = momentum.score(passing, "short", WEIGHTS, extension_penalty_atr=4.0)
    assert ranked.set_index("symbol")["extension_penalty"].loc["DOWN"] == pytest.approx(1.0)


def test_score_on_empty_frame_returns_empty():
    assert momentum.score(pd.DataFrame(), "long", WEIGHTS).empty


# --- response normalization --------------------------------------------------

def test_rows_of_unwraps_common_containers():
    assert rows_of({"data": [{"a": 1}]}) == [{"a": 1}]
    assert rows_of({"items": {"list": [{"b": 2}]}}) == [{"b": 2}]
    assert rows_of([{"c": 3}]) == [{"c": 3}]
    assert rows_of(None) == []
    assert rows_of({"x": 1}) == [{"x": 1}]


def test_pick_prefers_first_present_key():
    assert pick({"closePrice": 5}, ["close", "closePrice"]) == 5
    assert pick({"close": None, "closePrice": 5}, ["close", "closePrice"]) == 5
    assert pick({}, ["close"], default=0) == 0


def test_num_coerces_strings_and_nulls():
    assert _num("12.5") == 12.5
    assert np.isnan(_num(None)) and np.isnan(_num("")) and np.isnan(_num("abc"))


def test_to_ts_handles_millis_seconds_and_strings():
    assert _to_ts(1750000000000).year == 2025      # millis
    assert _to_ts(1750000000).year == 2025          # seconds
    assert _to_ts("2025-06-15").strftime("%Y-%m-%d") == "2025-06-15"
    assert pd.isna(_to_ts("not a date"))


# --- options layer -----------------------------------------------------------

OPT_CFG = dict(dte_min=25, dte_max=50, strike_band_pct=0.08, candidates_per_symbol=3,
               max_spread_pct=0.08, min_open_interest=250,
               iv_rich_ratio=1.35, iv_cheap_ratio=0.85)


def test_verdict_flags_wide_spreads_and_thin_oi():
    v = options.verdict({"opt_spread_pct": 0.20, "opt_oi": 30, "iv_hv_ratio": np.nan}, OPT_CFG)
    assert "wide" in v and "thin OI" in v


def test_verdict_reads_iv_against_realized_vol():
    rich = options.verdict({"opt_spread_pct": 0.02, "opt_oi": 5000, "iv_hv_ratio": 1.8}, OPT_CFG)
    cheap = options.verdict({"opt_spread_pct": 0.02, "opt_oi": 5000, "iv_hv_ratio": 0.6}, OPT_CFG)
    assert "favor spreads" in rich
    assert "long premium ok" in cheap


def test_tradable_requires_ok_status_and_clean_liquidity():
    good = {"opt_status": "ok", "opt_spread_pct": 0.03, "opt_oi": 1000}
    assert options.tradable(good, OPT_CFG)
    assert not options.tradable({**good, "opt_spread_pct": 0.25}, OPT_CFG)
    assert not options.tradable({**good, "opt_oi": 10}, OPT_CFG)
    assert not options.tradable({**good, "opt_status": "not entitled"}, OPT_CFG)
