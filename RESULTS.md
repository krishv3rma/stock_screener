# Backtest results — Holy Grail Daily Trend Edition

Basket: AAPL, MSFT, NVDA, JPM, XOM · daily bars · 2022-07-12 → 2026-07-06 · 214 trades

All results are in R-multiples, where 1R is the initial stop distance at entry
(entry − 2.0 × ATR(14), locked at entry). No commissions, slippage, or modeled
fills. Signals are evaluated on bar close.

---

## Headline numbers

| Metric | Value |
|---|---|
| Trades | 214 |
| Net result | **+25.19R** |
| Expectancy | +0.118R per trade |
| Win rate | 32.2% (69W / 145L) |
| Average win | +1.71R |
| Average loss | −0.64R |
| Profit factor | 1.27 |
| Best trade | +14.92R |
| Worst trade | −1.87R |
| Average hold | 16 bars |
| Longest losing streak | 11 |
| Maximum drawdown | ≈ −13R |

This is a recognisable trend-following distribution: a low win rate carried by
a win/loss ratio near 2.7:1. It is not a high-accuracy system and was never
designed to be one.

---

## Finding 1 — the short side loses money

| Direction | Trades | Net R |
|---|---|---|
| Long | 132 | **+56.1R** |
| Short | 82 | **−30.9R** |

Shorts gave back 55% of what longs earned. Disabling the short side entirely
would have more than doubled the system's net result over this period.

**Caveat, and it is the important one:** the test window is 2022–2026 on
US large-cap equities, which is close to a best case for long-only trend
following. The honest reading is not "shorts don't work" but "this system's
short side was not tested in a regime where shorts should work." A test that
included 2008 or 2000–2002 would be needed before drawing a general
conclusion. The finding is real for this window; the generalisation is not
supported yet.

## Finding 2 — the result depends on a handful of trades

The single best trade (+14.92R) is 59% of the net +25.19R. The equity curve
advances in two step-changes — mid-2023 and early 2024 — and drifts sideways
or downward between them.

Remove the top three trades and the system is roughly flat. This is a normal
property of trend following rather than a defect, but it means the expectancy
figure is fragile: it describes the average of a distribution that is
dominated by its tail, so a 214-trade sample is small for the claim it makes.

## Finding 3 — the first ten months were underwater

The curve does not cross zero until roughly April 2023 and reaches −13R before
it does. Anyone trading this live would have spent three quarters losing money
before the system did anything.

That is a practical result, not a statistical one, and it belongs in the
headline: a system with a positive long-run expectancy can still be
untradeable if the drawdown arrives before the returns.

## Finding 4 — two features barely fired

| Exit reason | Count | Share |
|---|---|---|
| Full exit (weight of evidence) | 168 | 78.5% |
| Hard stop | 39 | 18.2% |
| Trailing stop (EMA50) | 5 | 2.3% |
| Flip | 2 | 0.9% |

The weight-of-evidence full exit is doing nearly all the work. The EMA50
trailing stop fired 5 times in 214 trades and the flip mechanism fired twice.

Both are substantial pieces of logic carrying real complexity, and on this
evidence neither earns its keep. The v9.3 guide predicted exactly this —
*"the honest expectation is that the backtest will reveal some enabled
features earn their keep and some don't."* This is which ones.

## Finding 5 — stops gap

The worst trade closed at −1.87R against a stop placed at −1R. On daily bars
the stop level is a plan, not a fill. Any forward expectancy estimate should
assume the loss side is worse than −1R on average, which makes the +0.118R
expectancy above an optimistic figure rather than a conservative one.

---

## What this does not establish

- **One basket, five tickers, four years.** 214 trades across five correlated
  US large caps is not an out-of-sample test; it is one observation with
  internal repetition.
- **No transaction costs.** At 214 trades and a 16-bar average hold,
  commissions are minor but slippage on the stop side is not.
- **The parameters were not held out.** The system carries roughly 38 inputs.
  Any of them tuned while looking at this window is curve-fitting, and this
  test cannot distinguish a tuned parameter from a sound one.
- **Survivorship.** AAPL, MSFT, NVDA, JPM and XOM were selected in 2026 as
  names that exist and are liquid in 2026.

## What would make it stronger

1. Re-run long-only and report the difference as the primary result.
2. Split 2022–2024 as in-sample and 2024–2026 as out-of-sample; report both.
3. Extend the window back through 2007–2009 to test the short side in a
   regime that should favour it.
4. Ablate the trailing stop and the flip logic and confirm the result is
   unchanged. If it is, remove them — two fewer degrees of freedom.
5. Add a slippage assumption of 0.25R on stop exits and re-report expectancy.
6. Widen the basket to the full screener universe rather than five names.

---

*Generated from `backtest_output/trades_AAPL_MSFT_NVDA_JPM_XOM.csv`.
Reproduce with `python -m holy_grail.backtest AAPL MSFT NVDA JPM XOM --period 5y`.*
