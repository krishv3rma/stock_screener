# Holy Grail Screener/Signal System

Python port of the "Holy Grail — Daily Trend Edition" TradingView Pine Script
indicator (currently tracking **v9.6**), combined with a Finviz-style
screener, for free daily signal scanning and backtesting. See
`.claude/plans` (or ask Claude) for the full project plan.

## Indicator version: v9.6

v9.6 added an **entry confirmation stack** on top of v9.4/v9.5's signal
sources — Stochastic/RSI/MACD agreement plus a stochastic exhaustion veto,
controlled by one `strictness` setting (`loose` / `balanced` / `strict` /
`custom`, default **`balanced`**). It gates NEW ENTRIES ONLY; every exit,
stop, trail, break-even latch, and chandelier ratchet is byte-for-byte
unchanged from v9.4. See `HolyGrailParams` in `holy_grail/engine.py` for
every field, and `_resolve_strictness()` for exactly what each preset
resolves to.

Note: `use_adx_filter` defaults to `True` here, which is *not* the Pine
default (`False`) — that's a deliberate tuning choice from backtesting for
this project's multi-week swing-trading goal, not part of the v9.6 port
itself. See git history for the backtest evidence.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

## Usage

Check a ticker's current indicator state (compare against your TradingView chart):

```bash
.venv/Scripts/python -m holy_grail.engine AAPL MSFT NVDA
```

Backtest one or more tickers over history:

```bash
.venv/Scripts/python -m holy_grail.backtest AAPL MSFT NVDA --period 5y
```

Trade logs land in `backtest_output/*.csv`, equity curve in `backtest_output/*.png`.

Run the Finviz-equivalent screener (defaults to S&P 500 members over $100B —
US-domiciled household names; the full-market list is still available via
`universe.get_full_market_tickers()`):

```bash
.venv/Scripts/python -m holy_grail.screener
```

Run the full daily scan (screener + engine) and update the dashboard's data file:

```bash
.venv/Scripts/python -m holy_grail.scan
```

**Daily workflow is automated** via a Windows Scheduled Task
(`HolyGrailDailyScan`, runs `scripts/daily_run.ps1` at 9:00 AM and 6:00 PM
local time — only while your PC is on and you're logged in). It runs the
scan and pushes `docs/data/signals.json` only if something changed. The
9 AM run is provisional (today's daily candle is still forming, so signals
can differ from end of day); the 6 PM run reflects the final close. Logs
land in `logs/run_*.log`. Manage the task with:

```powershell
Get-ScheduledTask -TaskName HolyGrailDailyScan          # check status
Start-ScheduledTask -TaskName HolyGrailDailyScan        # run it right now
Disable-ScheduledTask -TaskName HolyGrailDailyScan      # pause it
```

To run it by hand instead:

```bash
.venv/Scripts/python -m holy_grail.scan
git add docs/data/signals.json
git commit -m "Update signals"
git push
```

GitHub Pages republishes `docs/` automatically a minute or two after the push
— check the dashboard from anywhere at your Pages URL.

To preview locally before pushing (fetch() of a local JSON file is blocked
by the browser over `file://`, so it needs a tiny server):

```bash
cd docs && "../.venv/Scripts/python.exe" -m http.server 8000
```

...and browse to `http://localhost:8000`.

## Status

- [x] Stage A — core engine (`holy_grail/engine.py`, `indicators.py`, `weekly.py`)
- [x] Stage B — backtester (`holy_grail/backtest.py`)
- [x] Stage C — Finviz-equivalent screener (`holy_grail/screener.py`, `universe.py`)
- [x] Stage D — daily scan orchestrator (`holy_grail/scan.py` → `docs/data/signals.json`)
- [x] Stage E — dashboard (`docs/index.html`, static, no build step)
- [x] Stage F — public GitHub repo + Pages hosting (source runs locally, on demand)

Live dashboard: https://krishv3rma.github.io/stock_screener/

## Known approximations vs. the live Pine script

- EMA/RMA warm-up uses SMA-seeded Wilder-style smoothing for both `ta.ema`
  and `ta.rma` for consistency; converges to the same values as TradingView
  after the smoothing window, but the very first `length` bars of a brand
  new series may differ slightly.
- Weekly EMA(50) resamples daily bars to Friday-anchored weeks — matches
  TradingView for tickers with a normal Mon-Fri trading week.
- Divergence pivot detection and R-multiple/stop mechanics are ported
  1:1 from the script's own logic and priority order.
- v9.6's entry confirmation stack (stoch/RSI/MACD agreement + exhaustion
  veto) is ported 1:1 including all three strictness presets; the
  `pendingLong`/`pendingShort` "blocked setup" markers are dashboard/alert
  cosmetics in Pine with no effect on trading logic, so they're not ported.
