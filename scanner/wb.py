"""
Thin wrapper over the official Webull OpenAPI Python SDK.

Handles the three things the SDK doesn't: rate limiting, retries, and
normalizing response payloads into pandas-friendly structures.

FIELD MAPPING
-------------
Webull's JSON field names are documented but occasionally shift between API
versions. Every field this app depends on is mapped in FIELD_MAP below, so if a
key changes you edit one dict instead of hunting through the codebase.

Run `python run.py --dump-shape AAPL` to print the raw JSON for a bar, snapshot,
and option contract response, then reconcile against FIELD_MAP.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import pandas as pd

from webull.core.client import ApiClient
from webull.data.data_client import DataClient
from webull.data.common.category import Category
from webull.data.common.timespan import Timespan

log = logging.getLogger(__name__)

# --- Field mapping -----------------------------------------------------------
# Left side = the name this app uses. Right side = candidate keys in Webull's
# payload, tried in order. First hit wins.
FIELD_MAP = {
    "bar": {
        "date": ["tradeTime", "trade_time", "timestamp", "time"],
        "open": ["open", "openPrice", "open_price"],
        "high": ["high", "highPrice", "high_price"],
        "low": ["low", "lowPrice", "low_price"],
        "close": ["close", "closePrice", "close_price"],
        "volume": ["volume", "vol"],
    },
    "rank": {
        "symbol": ["symbol", "disSymbol", "ticker"],
        "name": ["name", "shortName"],
        "price": ["price", "close", "lastPrice"],
        "change_ratio": ["changeRatio", "change_ratio"],
        "volume": ["volume", "vol"],
        "turnover": ["turnover", "amount", "turnover_rate"],  # reconciled 2026-09-18: live payload uses turnover_rate
        "market_value": ["marketValue", "market_value", "totalMarketValue"],
        "rvol_10d": ["relativeVolume10d", "relative_volume_10d", "rvol10d"],
    },
    "snapshot": {
        "symbol": ["symbol", "disSymbol"],
        "price": ["close", "price", "lastPrice"],
        "bid": ["bidPrice", "bid_price", "bid"],
        "ask": ["askPrice", "ask_price", "ask"],
        "volume": ["volume", "vol"],
    },
    "contract": {
        "symbol": ["symbol", "optionSymbol", "option_symbol"],
        "underlying": ["underlyingSymbol", "unSymbol", "underlying_symbol"],
        "strike": ["strikePrice", "strike_price", "strike"],
        "expiry": ["expireDate", "expiration_date", "expDate", "expirationDate"],
        "option_type": ["optionType", "direction", "option_type", "callOrPut"],
        "def_type": ["defType", "def_type"],
    },
    "opt_snapshot": {
        "symbol": ["symbol", "optionSymbol"],
        "bid": ["bidPrice", "bid_price", "bid"],
        "ask": ["askPrice", "ask_price", "ask"],
        "last": ["close", "lastPrice", "price"],
        "volume": ["volume", "vol"],
        "open_interest": ["openInterest", "open_interest", "openInt"],
        "iv": ["impliedVolatility", "implied_volatility", "impVol", "iv", "imp_vol"],  # reconciled 2026-09-18: live payload uses imp_vol
        "delta": ["delta"],
    },
}

# Containers Webull wraps list payloads in, tried in order.
LIST_KEYS = ("data", "items", "list", "records", "result", "rows", "contracts")


class WebullDataError(RuntimeError):
    pass


class EntitlementError(WebullDataError):
    """Raised on 403 — the account lacks the market-data subscription."""


def pick(obj: dict, spec: Sequence[str], default=None):
    """Return the first present key from `spec`."""
    for key in spec:
        if isinstance(obj, dict) and key in obj and obj[key] is not None:
            return obj[key]
    return default


def rows_of(payload: Any) -> list[dict]:
    """Coerce a Webull JSON payload into a flat list of dicts."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in LIST_KEYS:
            if key in payload:
                return rows_of(payload[key])
        # A single record.
        return [payload]
    return []


class RateLimiter:
    """Single-token bucket. Webull documents per-App-Key per-second limits."""

    def __init__(self, rps: float = 1.0):
        self.interval = 1.0 / max(rps, 0.01)
        self._last = 0.0

    def wait(self) -> None:
        gap = self.interval - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


@dataclass
class WebullConfig:
    app_key: str
    app_secret: str
    region: str = "us"
    endpoint: str = "api.webull.com"
    requests_per_second: float = 1.0
    max_retries: int = 4
    bars_batch_size: int = 20
    snapshot_batch_size: int = 100
    option_snapshot_batch_size: int = 20


class WebullData:
    def __init__(self, cfg: WebullConfig):
        self.cfg = cfg
        api_client = ApiClient(cfg.app_key, cfg.app_secret, cfg.region)
        api_client.add_endpoint(cfg.region, cfg.endpoint)
        self.client = DataClient(api_client)
        self.limiter = RateLimiter(cfg.requests_per_second)
        self.last_raw: dict[str, Any] = {}  # for --dump-shape

    # --- transport -----------------------------------------------------------

    def _call(self, fn, *args, label: str = "", **kwargs):
        last_exc = None
        for attempt in range(self.cfg.max_retries):
            self.limiter.wait()
            try:
                res = fn(*args, **kwargs)
            except Exception as exc:  # SDK raises on transport failures
                last_exc = exc
                time.sleep(min(2**attempt, 8))
                continue

            code = getattr(res, "status_code", 200)
            if code == 200:
                payload = res.json()
                if label:
                    self.last_raw[label] = payload
                return payload
            if code == 403:
                raise EntitlementError(
                    f"403 from Webull for {label or fn.__name__}. This endpoint needs an "
                    "active OpenAPI market-data subscription on your account."
                )
            if code in (429, 500, 502, 503, 504):
                time.sleep(min(2**attempt, 8))
                last_exc = WebullDataError(f"HTTP {code} from {label or fn.__name__}")
                continue
            raise WebullDataError(
                f"HTTP {code} from {label or fn.__name__}: {getattr(res, 'text', '')[:300]}"
            )
        raise WebullDataError(f"{label or fn.__name__} failed after retries: {last_exc}")

    @staticmethod
    def _chunks(seq: Sequence, size: int) -> Iterable[Sequence]:
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    # --- screener ------------------------------------------------------------

    def gainers_losers(self, rank_type: str, sort_by: str = "CHANGE_RATIO",
                       direction: str = "DESC") -> pd.DataFrame:
        payload = self._call(
            self.client.screener.list_gainers_losers,
            rank_type, Category.US_STOCK.name, sort_by, direction,
            label=f"gainers_losers:{rank_type}:{direction}",
        )
        return self._rank_frame(payload)

    def most_active(self, rank_type: str = "RELATIVE_VOLUME_10D",
                    sort_by: str = "RELATIVE_VOLUME_10D") -> pd.DataFrame:
        payload = self._call(
            self.client.screener.list_most_active,
            Category.US_STOCK.name, rank_type, sort_by, "DESC",
            label=f"most_active:{rank_type}",
        )
        return self._rank_frame(payload)

    def week52_hl(self, rank_type: str = "NEW_HIGH",
                  sort_by: str = "RELATIVE_VOLUME_10D") -> pd.DataFrame:
        payload = self._call(
            self.client.screener.list_52whl,
            Category.US_STOCK.name, rank_type, sort_by, "DESC",
            label=f"52whl:{rank_type}",
        )
        return self._rank_frame(payload)

    def _rank_frame(self, payload) -> pd.DataFrame:
        spec = FIELD_MAP["rank"]
        out = []
        for row in rows_of(payload):
            sym = pick(row, spec["symbol"])
            if not sym:
                continue
            out.append({
                "symbol": str(sym).upper(),
                "name": pick(row, spec["name"], ""),
                "wb_price": _num(pick(row, spec["price"])),
                "wb_change_ratio": _num(pick(row, spec["change_ratio"])),
                "wb_volume": _num(pick(row, spec["volume"])),
                "wb_turnover": _num(pick(row, spec["turnover"])),
                "wb_market_value": _num(pick(row, spec["market_value"])),
                "wb_rvol_10d": _num(pick(row, spec["rvol_10d"])),
            })
        return pd.DataFrame(out)

    # --- bars ----------------------------------------------------------------

    def daily_bars(self, symbols: Sequence[str], count: int = 300) -> dict[str, pd.DataFrame]:
        """Daily OHLCV per symbol, oldest first. Missing symbols are omitted."""
        result: dict[str, pd.DataFrame] = {}
        symbols = [s.upper() for s in symbols]
        for batch in self._chunks(symbols, self.cfg.bars_batch_size):
            try:
                payload = self._call(
                    self.client.market_data.get_batch_history_bar,
                    list(batch), Category.US_STOCK.name, Timespan.D.name,
                    count=str(count),
                    label="bars",
                )
            except WebullDataError as exc:
                # 2026-09-18: a single bad symbol in a batch (delisted,
                # renamed, or a screener artifact - seen live with "UZX")
                # makes Webull reject the WHOLE batch with INVALID_SYMBOL,
                # not just that symbol. Don't let one bad symbol in 20 take
                # down the whole run - retry the batch one symbol at a time
                # and only drop the ones that individually fail.
                log.warning("bar batch failed (%s), retrying %d symbols individually: %s",
                            len(batch), len(batch), exc)
                for sym in batch:
                    try:
                        payload = self._call(
                            self.client.market_data.get_batch_history_bar,
                            [sym], Category.US_STOCK.name, Timespan.D.name,
                            count=str(count),
                            label="bars",
                        )
                    except WebullDataError as sym_exc:
                        log.warning("dropping %s: %s", sym, sym_exc)
                        continue
                    result.update(self._bar_frames(payload, {sym}))
                continue
            result.update(self._bar_frames(payload, set(batch)))
        return result

    def _bar_frames(self, payload, wanted: set[str]) -> dict[str, pd.DataFrame]:
        spec = FIELD_MAP["bar"]
        out: dict[str, pd.DataFrame] = {}
        for group in rows_of(payload):
            sym = pick(group, FIELD_MAP["rank"]["symbol"])
            nested = None
            for key in ("bars", "barList", "data", "candles", "klines", "result"):  # reconciled 2026-09-18: live per-symbol bar payload nests under "result"
                if isinstance(group.get(key), list):
                    nested = group[key]
                    break
            if nested is None:
                continue
            sym = str(sym).upper() if sym else None
            if sym is None or (wanted and sym not in wanted):
                continue
            recs = []
            for bar in nested:
                if not isinstance(bar, dict):
                    continue
                recs.append({
                    "date": _to_ts(pick(bar, spec["date"])),
                    "open": _num(pick(bar, spec["open"])),
                    "high": _num(pick(bar, spec["high"])),
                    "low": _num(pick(bar, spec["low"])),
                    "close": _num(pick(bar, spec["close"])),
                    "volume": _num(pick(bar, spec["volume"])),
                })
            if not recs:
                continue
            df = pd.DataFrame(recs).dropna(subset=["date", "close"])
            df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
            out[sym] = df
        return out

    # --- snapshots -----------------------------------------------------------

    def snapshot(self, symbols: Sequence[str]) -> pd.DataFrame:
        spec = FIELD_MAP["snapshot"]
        out = []
        for batch in self._chunks([s.upper() for s in symbols], self.cfg.snapshot_batch_size):
            payload = self._call(
                self.client.market_data.get_snapshot,
                list(batch), Category.US_STOCK.name,
                label="snapshot",
            )
            for row in rows_of(payload):
                sym = pick(row, spec["symbol"])
                if not sym:
                    continue
                out.append({
                    "symbol": str(sym).upper(),
                    "snap_price": _num(pick(row, spec["price"])),
                    "snap_bid": _num(pick(row, spec["bid"])),
                    "snap_ask": _num(pick(row, spec["ask"])),
                    "snap_volume": _num(pick(row, spec["volume"])),
                })
        return pd.DataFrame(out)

    # --- options -------------------------------------------------------------

    def option_contracts(self, underlying: str, start: str, end: str,
                         strike_low: float, strike_high: float) -> pd.DataFrame:
        spec = FIELD_MAP["contract"]
        # 2026-09-25: confirmed live against the sandbox that the SDK's
        # "end_date" param is actually an inclusive UPPER bound on expiry,
        # the opposite of what its own docstring claims ("lower bound").
        # end_date=<today> returned ONLY today's 0-DTE contracts; end_date=None
        # returned everything out to 2029. There is no server-side lower-bound
        # param at all, so `end` (our dte_max target) goes here, and `start`
        # (our dte_min target) is enforced client-side below instead.
        payload = self._call(
            self.client.instrument.list_option_contracts,
            Category.US_OPTION.name, None, underlying.upper(), "LISTING",
            None, end,
            None, None, None,
            strike_low, strike_high,
            label="option_contracts",
        )
        out = []
        for row in rows_of(payload):
            sym = pick(row, spec["symbol"])
            if not sym:
                continue
            def_type = str(pick(row, spec["def_type"], "STANDARD")).upper()
            if def_type != "STANDARD":
                # 2026-09-18: live responses also include FLEX contracts (root
                # symbol prefixed like "2AAPL", def_type FLEX/EUROPEAN-style).
                # These come back INVALID_SYMBOL from the option-snapshot
                # endpoint, and since candidates are snapshotted in a batch,
                # one FLEX symbol in the batch was failing the whole batch's
                # quotes. Standard swing-options candidates only.
                continue
            out.append({
                "option_symbol": str(sym).upper(),
                "underlying": str(pick(row, spec["underlying"], underlying)).upper(),
                "strike": _num(pick(row, spec["strike"])),
                "expiry": _to_ts(pick(row, spec["expiry"])),
                "option_type": str(pick(row, spec["option_type"], "")).upper(),
            })
        df = pd.DataFrame(out)
        if not df.empty and "expiry" in df:
            # 2026-09-25: confirmed live that Webull's "end_date" request param
            # (documented as an inclusive LOWER bound on expiry) is not
            # actually enforced by the sandbox - a request with start=+25d
            # still returned contracts expiring the same day (0 DTE). The SDK
            # also has no upper-bound request param at all. So both bounds of
            # the DTE window have to be enforced client-side here, not just
            # the upper one that was already being applied.
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            df = df[df["expiry"].notna() & (df["expiry"] >= start_ts) & (df["expiry"] <= end_ts)]
        return df.reset_index(drop=True)

    def option_snapshot(self, option_symbols: Sequence[str]) -> pd.DataFrame:
        spec = FIELD_MAP["opt_snapshot"]
        out = []
        for batch in self._chunks(list(option_symbols), self.cfg.option_snapshot_batch_size):
            payload = self._call(
                self.client.option_market_data.get_option_snapshot,
                list(batch), Category.US_OPTION.name,
                label="option_snapshot",
            )
            for row in rows_of(payload):
                sym = pick(row, spec["symbol"])
                if not sym:
                    continue
                out.append({
                    "option_symbol": str(sym).upper(),
                    "opt_bid": _num(pick(row, spec["bid"])),
                    "opt_ask": _num(pick(row, spec["ask"])),
                    "opt_last": _num(pick(row, spec["last"])),
                    "opt_volume": _num(pick(row, spec["volume"])),
                    "open_interest": _num(pick(row, spec["open_interest"])),
                    "iv": _num(pick(row, spec["iv"])),
                    "delta": _num(pick(row, spec["delta"])),
                })
        return pd.DataFrame(out)


# --- coercion helpers --------------------------------------------------------

def _num(v):
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _to_ts(v):
    if v is None or v == "":
        return pd.NaT
    # Epoch millis or seconds.
    if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()):
        n = float(v)
        unit = "ms" if n > 1e11 else "s"
        try:
            return pd.to_datetime(n, unit=unit, utc=True).tz_convert(None).normalize()
        except Exception:
            return pd.NaT
    try:
        return pd.to_datetime(v, utc=True, format="mixed").tz_convert(None).normalize()
    except Exception:
        try:
            return pd.to_datetime(v).normalize()
        except Exception:
            return pd.NaT
