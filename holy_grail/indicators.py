"""
Hand-built technical indicators matching TradingView Pine Script's ta.* builtins
bar-for-bar. Pine's ta.rsi / ta.atr / ta.dmi all use Wilder's RMA smoothing
(alpha = 1/length), which is NOT the same as pandas .ewm(span=length)
(alpha = 2/(length+1)) and not the same as most pip `ta`/`pandas-ta` defaults.

All functions take/return pandas Series aligned to the input index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def stdev(series: pd.Series, length: int) -> pd.Series:
    # Pine's ta.stdev defaults to the biased (population) estimator.
    return series.rolling(length, min_periods=length).std(ddof=0)


def _seeded_ma(series: pd.Series, length: int, alpha: float) -> pd.Series:
    """
    Shared warm-up logic for ta.ema/ta.rma: NaN for the first length-1 bars,
    first valid value = SMA of the first `length` values, then recurse with
    the given alpha. (This is what makes a young listing's weekly EMA(50)
    read as na for ~50 weeks, per the indicator's own PATCH-1 note.)
    """
    values = series.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    n = len(values)
    if n < length:
        return pd.Series(out, index=series.index)

    start = None
    for i in range(length - 1, n):
        window = values[i - length + 1 : i + 1]
        if not np.isnan(window).any():
            out[i] = window.mean()
            start = i
            break
    if start is None:
        return pd.Series(out, index=series.index)

    for i in range(start + 1, n):
        x = values[i]
        if np.isnan(x):
            out[i] = np.nan
            continue
        prev = out[i - 1]
        if np.isnan(prev):
            # gap after seed (shouldn't normally happen) — reseed
            window = values[max(0, i - length + 1) : i + 1]
            out[i] = np.nanmean(window)
        else:
            out[i] = alpha * x + (1 - alpha) * prev
    return pd.Series(out, index=series.index)


def ema(series: pd.Series, length: int) -> pd.Series:
    """Pine ta.ema: alpha = 2/(length+1), SMA-seeded (see _seeded_ma)."""
    return _seeded_ma(series, length, alpha=2.0 / (length + 1))


def rma(series: pd.Series, length: int) -> pd.Series:
    """Pine ta.rma (Wilder's smoothing): alpha = 1/length, SMA-seeded."""
    return _seeded_ma(series, length, alpha=1.0 / length)


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out[(avg_loss == 0.0) & (avg_gain > 0.0)] = 100.0
    out[(avg_loss == 0.0) & (avg_gain == 0.0)] = 50.0
    return out


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # first bar: no previous close, Pine falls back to high-low
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    return rma(true_range(high, low, close), length)


def dmi(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
    """Returns (plus_di, minus_di, adx), matching Pine ta.dmi(length, length)."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    tr_rma = rma(true_range(high, low, close), length)
    plus_di = 100.0 * rma(plus_dm, length) / tr_rma.replace(0.0, np.nan)
    minus_di = 100.0 * rma(minus_dm, length) / tr_rma.replace(0.0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = rma(dx, length)
    return plus_di, minus_di, adx


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def stochastic(close: pd.Series, high: pd.Series, low: pd.Series, length: int = 14, smooth_d: int = 3, smooth_k: int = 1):
    """
    Matches Pine v9.6's two-stage smoothing:
        stochRaw = ta.stoch(close, high, low, length)
        stochK   = ta.sma(stochRaw, smooth_k)   # smooth_k=1 is a no-op (v9.4 behavior)
        stochD   = ta.sma(stochK, smooth_d)
    """
    lowest_low = low.rolling(length, min_periods=length).min()
    highest_high = high.rolling(length, min_periods=length).max()
    raw = 100.0 * (close - lowest_low) / (highest_high - lowest_low).replace(0.0, np.nan)
    k = sma(raw, smooth_k)
    d = sma(k, smooth_d)
    return k, d


def bollinger(close: pd.Series, length: int = 20, mult: float = 2.0):
    basis = sma(close, length)
    dev = stdev(close, length) * mult
    return basis, basis + dev, basis - dev


def linreg(series: pd.Series, length: int, offset: int = 0) -> pd.Series:
    """
    Pine ta.linreg: least-squares fit over the trailing `length` bars,
    evaluated `offset` bars back from the current bar (offset=0 -> current).
    """
    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _fit(window: np.ndarray) -> float:
        if np.isnan(window).any():
            return np.nan
        y_mean = window.mean()
        slope = ((x - x_mean) * (window - y_mean)).sum() / x_var
        intercept = y_mean - slope * x_mean
        return slope * (length - 1 - offset) + intercept

    return series.rolling(length, min_periods=length).apply(_fit, raw=True)


def pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
    """
    Pine ta.pivothigh(source, left, right): confirmed `right` bars after the
    pivot bar. Value at index i is source[i-right] if that bar is the max
    over the window [i-right-left, i-right+right], else NaN.
    """
    return _pivot_high_impl(series, left, right)


def _pivot_high_impl(series: pd.Series, left: int, right: int) -> pd.Series:
    window = left + right + 1
    vals = series.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    for end in range(window - 1, n):
        w = vals[end - window + 1 : end + 1]
        if np.isnan(w).any():
            continue
        pivot_idx_in_window = left  # the candidate pivot bar position within window
        if w[pivot_idx_in_window] == w.max():
            out[end - right] = w[pivot_idx_in_window]
    return pd.Series(out, index=series.index)


def _pivot_low_impl(series: pd.Series, left: int, right: int) -> pd.Series:
    window = left + right + 1
    vals = series.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    for end in range(window - 1, n):
        w = vals[end - window + 1 : end + 1]
        if np.isnan(w).any():
            continue
        pivot_idx_in_window = left
        if w[pivot_idx_in_window] == w.min():
            out[end - right] = w[pivot_idx_in_window]
    return pd.Series(out, index=series.index)


def pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
    return _pivot_low_impl(series, left, right)


def crossover(a: pd.Series, b) -> pd.Series:
    """Pine ta.crossover(a, b): a[1] < b[1] and a > b (strict both sides)."""
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a > b) & (a.shift(1) < b.shift(1))


def crossunder(a: pd.Series, b) -> pd.Series:
    """Pine ta.crossunder(a, b): a[1] > b[1] and a < b (strict both sides)."""
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a < b) & (a.shift(1) > b.shift(1))


def bars_since(cond: pd.Series, sentinel: int = 9999) -> np.ndarray:
    """
    Pine ta.barssince(cond), with Pine's nz(..., sentinel) applied for bars
    before the condition has ever been true (ta.barssince is na there).
    On the bar cond is true, this returns 0 (matches Pine).
    """
    arr = cond.to_numpy(dtype=bool)
    out = np.full(arr.shape, sentinel, dtype=np.int64)
    last_true = -1
    for i in range(len(arr)):
        if arr[i]:
            last_true = i
        if last_true >= 0:
            out[i] = i - last_true
    return out
