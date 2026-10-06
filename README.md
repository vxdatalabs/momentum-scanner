# Momentum Scanner

Ranks US equities by swing-horizon momentum (days to weeks) on both the long and
short side, then checks whether each candidate's options are actually worth
trading — spread, open interest, and implied vol against the stock's own
realized vol.

Data comes from the official Webull OpenAPI. Runs locally, no server needed.

---

## Try it before you have credentials

Webull's OpenAPI application takes a day or two to approve. You can run the whole
pipeline right now on synthetic data:

```bash
pip install -r requirements.txt
python run.py --demo
```

That exercises every code path — universe, scoring, options layer, output — with
fabricated bars and quotes, so you can tune `config.yaml` while you wait.

---

## Real setup

1. **Apply for OpenAPI access.** In your Webull account, go to OpenAPI
   Management and request an App Key and App Secret. Approval usually takes 1–2
   business days.
2. **Subscribe to market data.** API access itself is free, but historical and
   real-time data for US stocks requires an active OpenAPI market-data
   subscription, billed separately from the app. Without it every data call
   returns 403.
3. **Configure.**
   ```bash
   cp .env.example .env
   # paste your App Key and App Secret into .env
   ```
4. **Run.**
   ```bash
   python run.py                    # both sides
   python run.py --side long --top 15
   python run.py --no-options       # momentum only, far fewer API calls
   python run.py --symbols NVDA,AMD,TSLA
   ```

Results print to the terminal and write to `output/` as CSV with every factor and
z-score, which is what you want when adjusting weights.

---

## How the ranking works

**Universe.** Pulling bars for every listed equity would take hours against a
per-second rate limit, so the scan seeds from Webull's own ranked lists — 1-month
and 3-month movers on both sides, 52-week highs and lows, relative-volume and
turnover leaders — and dedupes the union. That typically lands around 300–600
names, which is a few minutes of batched requests.

**Gates.** Before anything is scored, candidates must clear hard filters: price
range, minimum average dollar volume (a proxy for a usable options chain), an ATR
band that excludes both dead names and gap-driven chaos, enough history, and no
single 25%+ gap in the last month. A "trend" that is one earnings print is not a
trend you can swing.

**Factors.** Each survivor gets a factor block, then every factor is z-scored
*across the candidate pool* before weighting. Cross-sectional normalization is
what makes a composite mean anything: a 12% one-month return is strong in a flat
tape and unremarkable in a melt-up.

| Factor | Weight | What it captures |
|---|---|---|
| `slope_q` | 0.30 | Annualized log-regression slope over 90 sessions × R². Trend *quality*, not just size |
| `ret_21d` | 0.15 | One-month return |
| `ret_63d` | 0.15 | Three-month return |
| `trend_align` | 0.15 | Price vs 20-EMA / 50-SMA / 200-SMA structure |
| `range_position` | 0.10 | Proximity to the 52-week extreme on the relevant side |
| `rvol` | 0.10 | 20-day volume vs 60-day. Participation |
| `up_days_20` | 0.05 | Steadiness rather than one big print |

The slope×R² term carries the most weight on purpose. Raw return rewards a single
gap; slope×R² rewards names that grind steadily in one direction, which is what
you need when you're holding an option through several sessions of theta.

**Short side.** Every directional factor flips sign, so a short score of 2.0
means the same structural strength that a long 2.0 means. Volume factors don't
flip — participation is evidence either way.

**Extension penalty.** Distance from the 20-EMA beyond 4 ATRs costs 0.25 score
points per excess ATR. A name 8 ATRs extended is a chase, not a setup.

---

## The options layer

For the top N per side, the scanner finds the near-the-money contract in your
target DTE window and reports:

- **Spread** as a share of mid — can you get filled without donating the edge
- **Open interest** — can you get out
- **IV / HV** — implied vol against the stock's own 20-day realized vol

That last one flips decisions more often than the others. A clean uptrend with IV
at twice realized vol means you're paying a large premium for movement the stock
hasn't historically delivered; the flag says `rich` and points you toward a
defined-risk spread instead of a long call.

**Constraint worth knowing:** options data on Webull's OpenAPI sits behind a
separate paid entitlement that is still being rolled out. If your account isn't
entitled, the scan doesn't break — it flags every row `no data`, keeps the
momentum ranking, and still gives you realized vol as a reference. Run with
`--no-options` to skip the calls entirely.

---

## Tuning

Everything adjustable lives in `config.yaml`; nothing in `scanner/` needs editing.
The parameters most worth your attention:

- `gates.min_avg_dollar_volume` — the single biggest lever on options quality.
  Raise it to $50M if you're getting too many names with unusable chains.
- `gates.max_atr_pct` — lower it if the list is full of names that gap through
  your stops.
- `weights` — they're normalized, so relative size is what matters. If you want a
  faster list, shift weight from `ret_63d` toward `ret_21d` and `rvol`.
- `options.dte_min` / `dte_max` — 25–50 is the usual swing window.

---

## Rate limits and caching

Bars are cached in `.cache/bars.sqlite`. A symbol already current through the
last completed session is skipped on the next run, so a second scan the same day
is nearly free. `--refresh` forces a full refetch.

`api.requests_per_second` defaults to 1.0, which is conservative — Webull
documents per-App-Key per-second limits and some option endpoints are explicitly
one call per second. Raise it gradually and watch for 429s.

---

## If the data looks wrong

Webull's JSON field names occasionally shift between API versions. Every field
this app depends on is mapped in `FIELD_MAP` at the top of `scanner/wb.py`, so a
rename is a one-line fix rather than a hunt. To see what the API is actually
returning:

```bash
python run.py --dump-shape AAPL
```

That prints raw payloads for bars, snapshots, the screener, and option contracts.
Compare against `FIELD_MAP` and adjust.

---

## Tests

```bash
python -m pytest tests/ -q
```

Covers the factor math (the regression slope is verified against its analytic
value), the gates, scoring symmetry between long and short, the response
normalizers, and the options verdict logic. No network required.

---

## What this doesn't do

It ranks candidates. It doesn't size positions, choose expirations, decide which
side to take, or tell you what to trade — and a high score is a starting point for
your own analysis, not a signal. There's no backtest here either: the factor
weights are reasonable priors, not fitted parameters, and you should treat them
as a starting configuration to evaluate against your own results rather than
something already validated.

Not investment advice, and I'm not a licensed advisor.
