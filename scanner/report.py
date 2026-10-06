"""Terminal tables and file output.

The terminal view is deliberately narrow: the columns you need to eyeball a
ranking, and nothing else. Columns drop out automatically as the terminal gets
narrower. Every computed factor — including the z-score breakdown behind each
composite — goes to CSV, which is where you actually want it when tuning weights.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

# (key, header, justify, min terminal width at which the column is shown)
CORE_COLS = [
    ("symbol", "Sym", "left", 0),
    ("score", "Score", "right", 0),
    ("price", "Price", "right", 0),
    ("ret_21d", "21d", "right", 0),
    ("ret_63d", "63d", "right", 90),
    ("slope_ann", "Slope", "right", 100),
    ("slope_r2", "R2", "right", 110),
    ("atr_pct", "ATR%", "right", 120),
    ("rvol", "RVol", "right", 130),
    ("ext_atr", "Ext", "right", 140),
]

OPT_COLS = [
    ("opt_strike", "Strike", "right", 0),
    ("opt_expiry", "Expiry", "left", 105),
    ("opt_spread_pct", "Spr", "right", 0),
    ("opt_oi", "OI", "right", 115),
    ("iv_hv_ratio", "IV/HV", "right", 0),
    ("opt_tag", "Flag", "left", 0),
]

SIGNED_PCT = {"ret_5d", "ret_21d", "ret_63d", "ret_126d", "slope_ann", "pct_from_52w_high"}
PLAIN_PCT = {"atr_pct", "opt_spread_pct", "hv20", "hv60", "opt_iv", "debit_pct_of_spot"}
TWO_DP = {"score", "slope_r2", "rvol", "ext_atr", "iv_hv_ratio", "trend_align", "extension_penalty"}


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and np.isfinite(v)


def _fmt(col: str, val) -> str:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "-"
    if isinstance(val, str):
        return val
    if col in SIGNED_PCT:
        return f"{val:+.0%}"
    if col in PLAIN_PCT:
        return f"{val:.1%}"
    if col in TWO_DP:
        return f"{val:.2f}"
    if col in ("price", "opt_strike"):
        return f"{val:,.1f}"
    if col in ("opt_oi", "opt_volume"):
        return f"{val:,.0f}" if val < 1000 else f"{val/1000:,.0f}k"
    return str(val)


def compact_tag(row: dict, cfg: dict | None = None) -> str:
    """Short flag for the terminal. The full note lives in opt_verdict in the CSV."""
    cfg = cfg or {}
    status = row.get("opt_status")
    if status == "not entitled":
        return "no data"
    if status and status != "ok":
        return str(status)[:14]

    bits = []
    spread, oi = row.get("opt_spread_pct"), row.get("opt_oi")
    if _finite(spread) and spread > cfg.get("max_spread_pct", 0.08):
        bits.append("wide")
    if _finite(oi) and oi < cfg.get("min_open_interest", 250):
        bits.append("thin")
    ratio = row.get("iv_hv_ratio")
    if _finite(ratio):
        bits.append(
            "rich" if ratio >= cfg.get("iv_rich_ratio", 1.35)
            else "cheap" if ratio <= cfg.get("iv_cheap_ratio", 0.85)
            else "fair"
        )
    return " ".join(bits) if bits else "ok"


def table(df: pd.DataFrame, title: str, top: int, show_options: bool,
          opt_cfg: dict | None = None) -> None:
    if df.empty:
        console.print(f"[yellow]{title}: nothing passed the filters.[/yellow]")
        return

    width = console.width or 100
    cols = [c for c in CORE_COLS if c[3] <= width]

    if show_options and "opt_status" in df.columns:
        df = df.copy()
        df["opt_tag"] = [compact_tag(r, opt_cfg) for r in df.to_dict("records")]
        cols += [c for c in OPT_COLS if c[3] <= width]

    cols = [c for c in cols if c[0] in df.columns]

    t = Table(title=title, title_style="bold", header_style="bold cyan",
              show_lines=False, pad_edge=False, expand=False)
    for _, label, justify, _ in cols:
        t.add_column(label, justify=justify, no_wrap=True, overflow="ellipsis")

    for row in df.head(top).to_dict("records"):
        cells = []
        for key, _, _, _ in cols:
            text = _fmt(key, row.get(key))
            if key == "score":
                v = row.get("score") or 0
                color = ("bold green" if v > 0.8 else "green" if v > 0.3
                         else "yellow" if v > 0 else "dim")
                text = f"[{color}]{text}[/{color}]"
            elif key == "opt_tag":
                color = "red" if ("wide" in text or "thin" in text) else \
                        "magenta" if "rich" in text else \
                        "green" if "cheap" in text else "dim"
                text = f"[{color}]{text}[/{color}]"
            cells.append(text)
        t.add_row(*cells)

    console.print(t)


def summary(n_universe: int, n_scored: int, rejected: pd.DataFrame, side: str) -> None:
    console.print(
        f"[dim]{side}: {n_universe} symbols -> {n_scored} scored, {len(rejected)} filtered[/dim]"
    )
    if not rejected.empty and "reject_reason" in rejected.columns:
        counts = rejected["reject_reason"].value_counts().to_dict()
        console.print("[dim]  " + "  ".join(f"{k}={v}" for k, v in counts.items()) + "[/dim]")


def legend(show_options: bool) -> None:
    lines = [
        "Score  composite of z-scored factors; weights live in config.yaml",
        "Slope  annualized log-regression slope over 90 sessions. R2 is how well it fits",
        "Ext    distance from the 20-EMA in ATR units. High means extended, not strong",
    ]
    if show_options:
        lines += [
            "Spr    bid/ask spread as a share of mid on the near-ATM contract",
            "IV/HV  implied vs realized vol. 'rich' argues for spreads over long premium",
        ]
    console.print("[dim]" + "\n".join(lines) + "[/dim]\n")


def write_csv(frames: dict[str, pd.DataFrame], outdir: str | Path) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    written = []
    for name, df in frames.items():
        if df is None or df.empty:
            continue
        path = outdir / f"{stamp}-{name}.csv"
        df.to_csv(path, index=False)
        written.append(path)
    return written
