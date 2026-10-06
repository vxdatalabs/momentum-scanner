"""
SQLite cache for daily bars.

Rate limits are the real cost of this scan, so we only fetch what we don't
already have. A symbol whose cached bars already run through the last completed
session is skipped entirely; everything else is refetched in full (Webull's bar
endpoint returns the last N bars rather than an arbitrary range, so partial
top-ups aren't cheaper than a clean refetch).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,
    open   REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_bars_symbol ON bars(symbol);
"""


class BarCache:
    def __init__(self, path: str | Path = ".cache/bars.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        # 2026-09-18: default (DELETE) journal mode raises "disk I/O error" on
        # this machine's connected-folder mount (no proper POSIX file locking
        # over the bridge) - TRUNCATE journal mode still persists safely to
        # disk but doesn't rely on the same locking dance, and works here.
        self.conn.execute("PRAGMA journal_mode=TRUNCATE")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def latest_dates(self, symbols: list[str]) -> dict[str, str]:
        if not symbols:
            return {}
        out: dict[str, str] = {}
        for i in range(0, len(symbols), 500):
            chunk = symbols[i : i + 500]
            marks = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT symbol, MAX(date), COUNT(*) FROM bars "
                f"WHERE symbol IN ({marks}) GROUP BY symbol",
                chunk,
            ).fetchall()
            for sym, maxdate, n in rows:
                if n >= 60:  # too little history to be worth reusing
                    out[sym] = maxdate
        return out

    def stale(self, symbols: list[str], through: pd.Timestamp) -> list[str]:
        """Symbols needing a fetch: absent, short, or not current through `through`."""
        latest = self.latest_dates(symbols)
        cutoff = through.strftime("%Y-%m-%d")
        return [s for s in symbols if latest.get(s, "") < cutoff]

    def put(self, bars: dict[str, pd.DataFrame]) -> int:
        rows = []
        for sym, df in bars.items():
            for r in df.itertuples():
                rows.append((
                    sym, pd.Timestamp(r.date).strftime("%Y-%m-%d"),
                    r.open, r.high, r.low, r.close, r.volume,
                ))
        if rows:
            self.conn.executemany(
                "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?)", rows
            )
            self.conn.commit()
        return len(rows)

    def get(self, symbols: list[str], min_bars: int = 60) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for i in range(0, len(symbols), 500):
            chunk = symbols[i : i + 500]
            marks = ",".join("?" * len(chunk))
            df = pd.read_sql_query(
                f"SELECT * FROM bars WHERE symbol IN ({marks}) ORDER BY symbol, date",
                self.conn, params=chunk,
            )
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"])
            for sym, grp in df.groupby("symbol"):
                if len(grp) >= min_bars:
                    out[sym] = grp.drop(columns=["symbol"]).reset_index(drop=True)
        return out
