#!/usr/bin/env python3
"""
Momentum scanner for swing-horizon options candidates.

    python run.py                       # both sides, default config
    python run.py --side long --top 15
    python run.py --no-options          # momentum ranking only, fewer API calls
    python run.py --symbols NVDA,AMD    # score a specific list
    python run.py --dump-shape AAPL     # print raw API payloads

This ranks candidates. It does not size positions, pick expirations for you, or
tell you what to trade.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from scanner import cache, momentum, options, report, universe
from scanner.wb import WebullConfig, WebullData, EntitlementError, WebullDataError

log = logging.getLogger("scan")


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_client(cfg: dict) -> WebullData:
    load_dotenv()
    key, secret = os.getenv("WEBULL_APP_KEY"), os.getenv("WEBULL_APP_SECRET")
    if not key or not secret:
        sys.exit(
            "Missing credentials. Copy .env.example to .env and fill in your\n"
            "Webull OpenAPI App Key and App Secret."
        )
    api = cfg["api"]
    return WebullData(WebullConfig(
        app_key=key,
        app_secret=secret,
        region=os.getenv("WEBULL_REGION", api.get("region", "us")),
        endpoint=os.getenv("WEBULL_ENDPOINT", api.get("endpoint", "api.webull.com")),
        requests_per_second=api.get("requests_per_second", 1.0),
        max_retries=api.get("max_retries", 4),
        bars_batch_size=api.get("bars_batch_size", 20),
        snapshot_batch_size=api.get("snapshot_batch_size", 100),
        option_snapshot_batch_size=api.get("option_snapshot_batch_size", 20),
    ))


def dump_shape(wb: WebullData, symbol: str) -> None:
    """Print raw payloads so FIELD_MAP can be reconciled against reality."""
    from webull.data.common.category import Category

    print(f"\n=== daily bars: {symbol} ===")
    try:
        wb.daily_bars([symbol], count=3)
        print(json.dumps(wb.last_raw.get("bars"), indent=2)[:3000])
    except Exception as exc:
        print(f"failed: {exc}")

    print(f"\n=== snapshot: {symbol} ===")
    try:
        wb.snapshot([symbol])
        print(json.dumps(wb.last_raw.get("snapshot"), indent=2)[:2000])
    except Exception as exc:
        print(f"failed: {exc}")

    print("\n=== screener: 1m gainers ===")
    try:
        wb.gainers_losers("MONTH_1")
        raw = next((v for k, v in wb.last_raw.items() if k.startswith("gainers")), None)
        print(json.dumps(raw, indent=2)[:2000])
    except Exception as exc:
        print(f"failed: {exc}")

    print(f"\n=== option contracts: {symbol} ===")
    try:
        from datetime import date, timedelta
        today = date.today()
        wb.option_contracts(
            symbol,
            (today + timedelta(days=25)).isoformat(),
            (today + timedelta(days=50)).isoformat(),
            1.0, 100000.0,
        )
        print(json.dumps(wb.last_raw.get("option_contracts"), indent=2)[:3000])
    except Exception as exc:
        print(f"failed: {exc}")


def run_side(side: str, bars: dict, cfg: dict, wb: WebullData, args) -> pd.DataFrame:
    frame = momentum.build_factor_frame(bars)
    passing, rejected = momentum.apply_gates(frame, cfg["gates"])
    ranked = momentum.score(
        passing, side, cfg["weights"],
        extension_penalty_atr=cfg["scoring"].get("extension_penalty_atr", 4.0),
    )

    report.summary(len(bars), len(passing), rejected, side)

    if not args.no_options and not ranked.empty:
        depth = max(args.top, cfg["options"].get("evaluate_top_n", 20))
        ranked = options.enrich(wb, ranked, side, cfg["options"], limit=depth)
        if cfg["options"].get("require_tradable", False):
            keep = ranked["opt_tradable"] | (ranked["opt_status"] == "not entitled")
            ranked = ranked[keep].reset_index(drop=True)

    label = "LONG candidates (call side)" if side == "long" else "SHORT candidates (put side)"
    report.table(ranked, label, args.top, show_options=not args.no_options,
                 opt_cfg=cfg["options"])
    return ranked


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--side", choices=["long", "short", "both"], default="both")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--no-options", action="store_true",
                   help="skip the options layer (much faster, no options entitlement needed)")
    p.add_argument("--symbols", help="comma-separated list; skips the screener universe")
    p.add_argument("--refresh", action="store_true", help="ignore the bar cache")
    p.add_argument("--out", default="output", help="directory for CSV output")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--dump-shape", metavar="SYMBOL", help="print raw API payloads and exit")
    p.add_argument("--demo", action="store_true",
                   help="run on synthetic data, no credentials or network needed")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname).1s %(message)s",
    )
    logging.getLogger("webull").setLevel(logging.WARNING)

    cfg = load_config(args.config)

    if args.demo:
        from scanner.demo import DemoClient
        wb = DemoClient()
        report.console.print("[yellow]DEMO MODE — synthetic data. These are not real securities.[/yellow]")
        from scanner.demo import TICKERS
        bars = wb.daily_bars([s for s, *_ in TICKERS])
        report.legend(show_options=not args.no_options)
        sides = ["long", "short"] if args.side == "both" else [args.side]
        results = {side: run_side(side, bars, cfg, wb, args) for side in sides}
        if not args.no_csv:
            for path in report.write_csv(results, args.out):
                report.console.print(f"[dim]wrote {path}[/dim]")
        return

    wb = build_client(cfg)

    if args.dump_shape:
        dump_shape(wb, args.dump_shape.upper())
        return

    # --- universe ---
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        log.info("scoring %d supplied symbols", len(symbols))
    else:
        try:
            uni = universe.build(wb, args.side)
        except EntitlementError as exc:
            sys.exit(f"\n{exc}\n\nSubscribe under OpenAPI market data, then rerun.")
        if uni.empty:
            sys.exit("Universe came back empty. Run with --dump-shape AAPL to inspect payloads.")
        symbols = uni["symbol"].tolist()
        cap = cfg["universe"].get("max_symbols", 600)
        if len(symbols) > cap:
            log.info("universe %d symbols, capping at %d", len(symbols), cap)
            symbols = symbols[:cap]
        log.info("universe: %d unique symbols", len(symbols))

    # --- bars ---
    lookback = cfg["universe"].get("bar_count", 300)
    with cache.BarCache(cfg["universe"].get("cache_path", ".cache/bars.sqlite")) as bc:
        through = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        need = symbols if args.refresh else bc.stale(symbols, through)
        if need:
            log.info("fetching bars for %d symbols (%d cached)", len(need), len(symbols) - len(need))
            try:
                fetched = wb.daily_bars(need, count=lookback)
            except EntitlementError as exc:
                sys.exit(f"\n{exc}")
            except WebullDataError as exc:
                sys.exit(f"\nBar fetch failed: {exc}")
            bc.put(fetched)
            log.info("cached bars for %d symbols", len(fetched))
        else:
            log.info("all %d symbols served from cache", len(symbols))
        bars = bc.get(symbols, min_bars=cfg["gates"]["min_bars"])

    if not bars:
        sys.exit("No usable bar history. Try --refresh, or --dump-shape AAPL to check field mapping.")
    log.info("scoring %d symbols with usable history", len(bars))

    # --- score ---
    report.legend(show_options=not args.no_options)
    sides = ["long", "short"] if args.side == "both" else [args.side]
    results = {side: run_side(side, bars, cfg, wb, args) for side in sides}

    if not args.no_csv:
        written = report.write_csv(results, args.out)
        for path in written:
            report.console.print(f"[dim]wrote {path}[/dim]")


if __name__ == "__main__":
    main()
