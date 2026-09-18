"""
Python port of the "Holy Grail — Daily Trend Edition v9.4" Pine Script v5
indicator. This is a bar-by-bar state machine (open trade direction, entry
price, ratcheting stop, partial-exit flag, cooldown, divergence pivots),
so it's ported the same way Pine executes it: vectorized indicator math
up front, then one sequential per-bar loop for everything that depends on
trade state — matching the script's own top-to-bottom mutation order.

Running this over full history == backtesting. Running it up through the
latest bar == today's live signal. Same engine, same code path, on purpose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from . import indicators as ta
from .weekly import non_repainting_weekly_ema


@dataclass
class HolyGrailParams:
    # Daily Trend Settings
    ema_fast_len: int = 21
    ema_slow_len: int = 50
    rsi_len: int = 14
    rsi_ob: float = 75
    rsi_os: float = 25
    cooldown_bars: int = 4
    grace_len: int = 2

    # Oscillator Engine (v9.6) - configurable, but defaults reproduce the
    # v9.4 hardcoded values exactly (stoch_smooth_k=1 is a no-op SMA).
    macd_fast_len: int = 12
    macd_slow_len: int = 26
    macd_sig_len: int = 9
    stoch_len: int = 14
    stoch_smooth_k: int = 1
    stoch_d_len: int = 3
    stoch_ob_lvl: float = 80
    stoch_os_lvl: float = 20

    # Signal Sources
    use_ema_cross: bool = False
    use_mom_break: bool = True
    use_rsi_reversal: bool = False
    use_squeeze: bool = True
    use_breakout: bool = True
    use_continuation: bool = True

    # Signal Filters
    require_trend: bool = True
    require_macd: bool = False
    require_weekly: bool = True
    # Enabled (Pine default is off) - blocks entries when ADX shows chop
    # rather than a real trend. Backtested across two different large-cap
    # name sets it was the single change that flipped expectancy positive,
    # and it lengthens the average hold toward the multi-week range.
    use_adx_filter: bool = True
    adx_min_level: float = 20.0
    rsi_min_long: float = 40
    rsi_max_short: float = 60
    rsi_ob_breakout: float = 85
    use_sqz_rsi_override: bool = True
    sqz_vol_mult: float = 2.0
    allow_reversal_bars: int = 3
    allow_rev_no_weekly: bool = True

    # Entry Confirmation Stack (v9.6, ADD-5/CHG-5/CHG-6). Gates NEW ENTRIES
    # ONLY - exits/stops/trails never touch this. "balanced" is the Pine
    # default and what actually runs unless the indicator is set otherwise.
    #   loose    - stack off entirely (v9.4 entry behaviour)
    #   balanced - all 3 confirmations active, ANY 2 of 3 must pass
    #   strict   - all 3 must pass, MACD must be a fresh cross (<=10 bars)
    #   custom   - every use_*_confirm / *_mode / *_lookback field below is
    #              honoured exactly as set
    strictness: str = "balanced"  # "loose" | "balanced" | "strict" | "custom"

    use_confirm_stack: bool = True
    confirm_logic: str = "at_least_n"  # "all" | "at_least_n"
    confirm_min_count: int = 2
    apply_confirm_to_rev: bool = True

    use_stoch_confirm: bool = True
    stoch_confirm_mode: str = "k_vs_d"  # "k_vs_d" | "fresh_cross" | "reclaim"
    stoch_lookback: int = 3

    use_rsi_confirm: bool = True
    rsi_confirm_mode: str = "midline"  # "avoid_extremes" | "midline" | "reclaim"
    rsi_midline: float = 50
    rsi_lookback: int = 5

    use_macd_confirm: bool = True
    macd_confirm_mode: str = "line_vs_signal"  # "line_vs_signal" | "hist_sign" | "fresh_cross"
    macd_lookback: int = 10

    veto_mode: str = "extreme_rolling_over"  # "off" | "extreme_only" | "extreme_rolling_over"

    # Continuation Settings
    cont_mom_bars: int = 3
    cont_pullback_pct: float = 1.0
    cont_rsi_max: float = 80
    cont_vol_min: float = 1.0

    # Exit Sensitivity
    exit_threshold: int = 2
    require_core_exit: bool = True
    div_piv_left: int = 3
    div_piv_right: int = 2
    div_active_bars: int = 5
    div_max_spacing: int = 30
    use_partial_at_r: bool = False
    partial_target_r: float = 1.5
    use_time_exit: bool = False
    time_exit_bars: int = 15
    time_exit_max_r: float = 0.5

    # Stop Loss
    use_hard_stop: bool = True
    atr_stop_mult: float = 2.0
    use_intrabar_stop: bool = False
    use_trail_stop: bool = True
    use_breakeven_stop: bool = True
    use_chandelier: bool = False
    chand_mult: float = 3.0
    chand_arm_r: float = 1.0

    # ATR
    atr_len: int = 14


_NA = float("nan")


@dataclass
class _EffectiveStrictness:
    """Resolved (e_*) confirmation-stack settings, per Pine's CHG-5 table."""
    use_stack: bool
    logic: str
    min_count: int
    apply_to_rev: bool
    stoch_on: bool
    stoch_mode: str
    stoch_lb: int
    rsi_on: bool
    rsi_mode: str
    rsi_mid: float
    rsi_lb: int
    macd_on: bool
    macd_mode: str
    macd_lb: int
    veto_mode: str


def _resolve_strictness(p: "HolyGrailParams") -> _EffectiveStrictness:
    is_custom = p.strictness == "custom"
    is_loose = p.strictness == "loose"
    is_strict = p.strictness == "strict"
    return _EffectiveStrictness(
        use_stack=p.use_confirm_stack if is_custom else not is_loose,
        logic=p.confirm_logic if is_custom else ("all" if is_strict else "at_least_n"),
        min_count=p.confirm_min_count if is_custom else 2,
        apply_to_rev=p.apply_confirm_to_rev if is_custom else not is_loose,
        stoch_on=p.use_stoch_confirm if is_custom else True,
        stoch_mode=p.stoch_confirm_mode if is_custom else "k_vs_d",
        stoch_lb=p.stoch_lookback if is_custom else 3,
        rsi_on=p.use_rsi_confirm if is_custom else True,
        rsi_mode=p.rsi_confirm_mode if is_custom else "midline",
        rsi_mid=p.rsi_midline if is_custom else 50,
        rsi_lb=p.rsi_lookback if is_custom else 5,
        macd_on=p.use_macd_confirm if is_custom else True,
        macd_mode=p.macd_confirm_mode if is_custom else ("fresh_cross" if is_strict else "line_vs_signal"),
        macd_lb=p.macd_lookback if is_custom else 10,
        veto_mode=p.veto_mode if is_custom else ("off" if is_loose else "extreme_rolling_over"),
    )


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


class HolyGrailEngine:
    def __init__(self, params: HolyGrailParams | None = None):
        self.p = params or HolyGrailParams()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df must have columns open, high, low, close, volume, indexed by a
        sorted DatetimeIndex of daily bars. Returns a DataFrame aligned to
        df.index with one row per bar of dashboard-equivalent state and
        entry/exit flags.
        """
        p = self.p
        o, h, l, c, v = (df[col].astype(float) for col in ("open", "high", "low", "close", "volume"))
        n = len(df)

        # ------------------------------------------------------------------
        # Vectorized indicators (no trade-state dependency)
        # ------------------------------------------------------------------
        ema_fast = ta.ema(c, p.ema_fast_len)
        ema_slow = ta.ema(c, p.ema_slow_len)
        ema200 = ta.ema(c, 200)
        atr = ta.atr(h, l, c, p.atr_len)
        rsi = ta.rsi(c, p.rsi_len)
        macd_line, macd_sig, macd_hist = ta.macd(c, p.macd_fast_len, p.macd_slow_len, p.macd_sig_len)
        stoch_k, stoch_d = ta.stochastic(c, h, l, p.stoch_len, p.stoch_d_len, p.stoch_smooth_k)
        bb_basis, bb_upper, bb_lower = ta.bollinger(c, 20, 2.0)
        is_squeeze = (bb_upper - bb_lower) < atr * 2.5
        sqz_dir = ta.linreg(c - (bb_upper + bb_lower) / 2.0, 14, 0)
        di_plus, di_minus, adx_val = ta.dmi(h, l, c, 14)
        vol_avg = ta.sma(v, 20)
        vol_spike = v > vol_avg * 2.0
        weekly_ema = non_repainting_weekly_ema(c, 50)
        weekly_bull = (not p.require_weekly) | (weekly_ema.notna() & (c > weekly_ema))
        weekly_bear = (not p.require_weekly) | (weekly_ema.notna() & (c < weekly_ema))
        adx_ok = (not p.use_adx_filter) | (adx_val >= p.adx_min_level)

        macd_long = (not p.require_macd) | (macd_hist > 0)
        macd_short = (not p.require_macd) | (macd_hist < 0)

        # Raw (toggle-independent) versions of the same conditions, for
        # scoring how "set up" a currently-flat ticker is for a future
        # entry - regardless of whether that filter is actually required
        # to gate a real entry right now.
        weekly_bull_raw = weekly_ema.notna() & (c > weekly_ema)
        weekly_bear_raw = weekly_ema.notna() & (c < weekly_ema)
        trend_bull_raw = (c > ema_slow) & (ema_fast > ema_slow)
        trend_bear_raw = (c < ema_slow) & (ema_fast < ema_slow)
        macd_bull_raw = macd_hist > 0
        macd_bear_raw = macd_hist < 0

        # signal sources
        long_ema_cross = p.use_ema_cross & ta.crossover(ema_fast, ema_slow)
        short_ema_cross = p.use_ema_cross & ta.crossunder(ema_fast, ema_slow)
        mom_long = p.use_mom_break & ta.crossover(c, ema_fast) & (c > ema_slow)
        mom_short = p.use_mom_break & ta.crossunder(c, ema_fast) & (c < ema_slow)
        rsi_bull_rev = p.use_rsi_reversal & ta.crossover(rsi, p.rsi_os) & (stoch_k > stoch_d)
        rsi_bear_rev = p.use_rsi_reversal & ta.crossunder(rsi, p.rsi_ob) & (stoch_k < stoch_d)
        was_squeezing = is_squeeze.shift(1).fillna(False) & is_squeeze.shift(2).fillna(False)
        sqz_long = p.use_squeeze & was_squeezing & (~is_squeeze) & (sqz_dir > 0)
        sqz_short = p.use_squeeze & was_squeezing & (~is_squeeze) & (sqz_dir < 0)

        brk_long = (
            p.use_breakout
            & (c > ema200) & (c.shift(1) <= ema200.shift(1))
            & (v > vol_avg * 1.5) & (macd_hist > 0) & (c > ema_fast) & (c > ema_slow)
        )
        brk_short = (
            p.use_breakout
            & (c < ema200) & (c.shift(1) >= ema200.shift(1))
            & (v > vol_avg * 1.5) & (macd_hist < 0) & (c < ema_fast) & (c < ema_slow)
        )

        # continuation
        cont_vol_ok = v > vol_avg * p.cont_vol_min
        pull_dist_low = (l - ema_fast).abs() / ema_fast * 100.0
        pull_dist_high = (h - ema_fast).abs() / ema_fast * 100.0
        min_pull_dist_low = pull_dist_low.shift(1).rolling(p.cont_mom_bars, min_periods=p.cont_mom_bars).min()
        min_pull_dist_high = pull_dist_high.shift(1).rolling(p.cont_mom_bars, min_periods=p.cont_mom_bars).min()
        had_pullback = min_pull_dist_low <= p.cont_pullback_pct
        had_pullback_sh = min_pull_dist_high <= p.cont_pullback_pct
        cur_bar_pullback_ok_long = pull_dist_low <= p.cont_pullback_pct * 2
        cur_bar_pullback_ok_short = pull_dist_high <= p.cont_pullback_pct * 2

        cont_pull_long_raw = (
            p.use_continuation & had_pullback & cur_bar_pullback_ok_long
            & (c > ema_fast) & (c > ema_slow) & (c > o) & (macd_hist > macd_hist.shift(1)) & cont_vol_ok
        )
        cont_pull_short_raw = (
            p.use_continuation & had_pullback_sh & cur_bar_pullback_ok_short
            & (c < ema_fast) & (c < ema_slow) & (c < o) & (macd_hist < macd_hist.shift(1)) & cont_vol_ok
        )

        above_fast_arr = (c - ema_fast).rolling(p.cont_mom_bars, min_periods=p.cont_mom_bars).min() > 0
        above_slow_arr = (c - ema_slow).rolling(p.cont_mom_bars, min_periods=p.cont_mom_bars).min() > 0
        below_fast_arr = (c - ema_fast).rolling(p.cont_mom_bars, min_periods=p.cont_mom_bars).max() < 0
        below_slow_arr = (c - ema_slow).rolling(p.cont_mom_bars, min_periods=p.cont_mom_bars).max() < 0

        cont_rsi_floor_short = 100 - p.cont_rsi_max
        mom_cont_long_raw = (
            p.use_continuation & above_fast_arr & above_slow_arr & (macd_hist > 0)
            & cont_vol_ok & (rsi < p.cont_rsi_max) & (c > o)
        )
        mom_cont_short_raw = (
            p.use_continuation & below_fast_arr & below_slow_arr & (macd_hist < 0)
            & cont_vol_ok & (rsi > cont_rsi_floor_short) & (c < o)
        )

        # RSI gates + squeeze override
        rsi_good_long = (rsi > p.rsi_min_long) & (rsi < p.rsi_ob)
        rsi_good_short = (rsi < p.rsi_max_short) & (rsi > p.rsi_os)
        rsi_good_brk_long = (rsi > p.rsi_min_long) & (rsi < p.rsi_ob_breakout)
        rsi_good_brk_short = (rsi < p.rsi_max_short) & (rsi > p.rsi_os)

        sqz_vol_ok = v > vol_avg * p.sqz_vol_mult
        sqz_override_long = p.use_sqz_rsi_override & sqz_long & sqz_vol_ok & (rsi > p.rsi_min_long)
        sqz_override_short = p.use_sqz_rsi_override & sqz_short & sqz_vol_ok & (rsi < p.rsi_max_short)

        std_long_sig = long_ema_cross | mom_long | rsi_bull_rev | sqz_long
        std_short_sig = short_ema_cross | mom_short | rsi_bear_rev | sqz_short

        # exit-against components
        macd_cross_under = ta.crossunder(macd_line, macd_sig)
        macd_cross_over = ta.crossover(macd_line, macd_sig)
        rsi_cross_under_50 = ta.crossunder(rsi, 50.0)
        rsi_cross_over_50 = ta.crossover(rsi, 50.0)
        stoch_cross_under = ta.crossunder(stoch_k, stoch_d)
        stoch_cross_over = ta.crossover(stoch_k, stoch_d)

        # ------------------------------------------------------------------
        # Entry Confirmation Stack (v9.6, ADD-5/CHG-5/CHG-6) - resolved once
        # from the strictness preset, evaluated vectorized since none of it
        # depends on trade state. Gates NEW ENTRIES ONLY.
        # ------------------------------------------------------------------
        e = _resolve_strictness(p)

        bars_since_macd_up = ta.bars_since(macd_cross_over)
        bars_since_macd_dn = ta.bars_since(macd_cross_under)
        bars_since_stoch_up = ta.bars_since(stoch_cross_over)
        bars_since_stoch_dn = ta.bars_since(stoch_cross_under)
        bars_since_stoch_leave_os = ta.bars_since(ta.crossover(stoch_k, p.stoch_os_lvl))
        bars_since_stoch_leave_ob = ta.bars_since(ta.crossunder(stoch_k, p.stoch_ob_lvl))
        bars_since_rsi_leave_os = ta.bars_since(ta.crossover(rsi, p.rsi_os))
        bars_since_rsi_leave_ob = ta.bars_since(ta.crossunder(rsi, p.rsi_ob))

        stoch_k_a0, stoch_d_a0 = stoch_k.to_numpy(dtype=float), stoch_d.to_numpy(dtype=float)
        if e.stoch_mode == "k_vs_d":
            stoch_pass_long = stoch_k_a0 > stoch_d_a0
            stoch_pass_short = stoch_k_a0 < stoch_d_a0
        elif e.stoch_mode == "fresh_cross":
            stoch_pass_long = bars_since_stoch_up <= e.stoch_lb
            stoch_pass_short = bars_since_stoch_dn <= e.stoch_lb
        else:  # "reclaim"
            stoch_pass_long = (bars_since_stoch_leave_os <= e.stoch_lb) & (stoch_k_a0 > stoch_d_a0)
            stoch_pass_short = (bars_since_stoch_leave_ob <= e.stoch_lb) & (stoch_k_a0 < stoch_d_a0)

        rsi_a0 = rsi.to_numpy(dtype=float)
        if e.rsi_mode == "avoid_extremes":
            rsi_pass_long = rsi_a0 < p.rsi_ob
            rsi_pass_short = rsi_a0 > p.rsi_os
        elif e.rsi_mode == "midline":
            rsi_pass_long = rsi_a0 > e.rsi_mid
            rsi_pass_short = rsi_a0 < e.rsi_mid
        else:  # "reclaim" - no stoch dependency in Pine's rsiPassLong/Short
            rsi_pass_long = bars_since_rsi_leave_os <= e.rsi_lb
            rsi_pass_short = bars_since_rsi_leave_ob <= e.rsi_lb

        macd_line_a0, macd_sig_a0 = macd_line.to_numpy(dtype=float), macd_sig.to_numpy(dtype=float)
        macd_hist_a0 = macd_hist.to_numpy(dtype=float)
        if e.macd_mode == "line_vs_signal":
            macd_pass_long = macd_line_a0 > macd_sig_a0
            macd_pass_short = macd_line_a0 < macd_sig_a0
        elif e.macd_mode == "hist_sign":
            macd_pass_long = macd_hist_a0 > 0
            macd_pass_short = macd_hist_a0 < 0
        else:  # "fresh_cross"
            macd_pass_long = bars_since_macd_up <= e.macd_lb
            macd_pass_short = bars_since_macd_dn <= e.macd_lb

        confirm_active = int(e.stoch_on) + int(e.rsi_on) + int(e.macd_on)
        long_pass_count = (
            (stoch_pass_long.astype(int) if e.stoch_on else 0)
            + (rsi_pass_long.astype(int) if e.rsi_on else 0)
            + (macd_pass_long.astype(int) if e.macd_on else 0)
        )
        short_pass_count = (
            (stoch_pass_short.astype(int) if e.stoch_on else 0)
            + (rsi_pass_short.astype(int) if e.rsi_on else 0)
            + (macd_pass_short.astype(int) if e.macd_on else 0)
        )
        confirm_need = confirm_active if e.logic == "all" else min(e.min_count, confirm_active)

        if e.veto_mode == "off":
            stoch_veto_long = np.zeros(n, dtype=bool)
            stoch_veto_short = np.zeros(n, dtype=bool)
        elif e.veto_mode == "extreme_only":
            stoch_veto_long = stoch_k_a0 > p.stoch_ob_lvl
            stoch_veto_short = stoch_k_a0 < p.stoch_os_lvl
        else:  # "extreme_rolling_over"
            stoch_veto_long = (stoch_k_a0 > p.stoch_ob_lvl) & (stoch_k_a0 < stoch_d_a0)
            stoch_veto_short = (stoch_k_a0 < p.stoch_os_lvl) & (stoch_k_a0 > stoch_d_a0)

        if e.use_stack:
            confirm_long = (long_pass_count >= confirm_need) & ~stoch_veto_long
            confirm_short = (short_pass_count >= confirm_need) & ~stoch_veto_short
        else:
            confirm_long = np.ones(n, dtype=bool)
            confirm_short = np.ones(n, dtype=bool)

        # divergence pivots (confirmation-lagged, spacing-capped) — precompute
        # the raw pivot points; the running ph1/ph2/pl1/pl2 bookkeeping is
        # inherently sequential and handled in the pre-pass loop below.
        ph_price = ta.pivot_high(h, p.div_piv_left, p.div_piv_right)
        pl_price = ta.pivot_low(l, p.div_piv_left, p.div_piv_right)
        rsi_at_pivot_confirm = rsi.shift(p.div_piv_right)

        # to numpy for the sequential loops
        arrs = {
            name: s.to_numpy(dtype=float)
            for name, s in {
                "open": o, "high": h, "low": l, "close": c, "volume": v,
                "ema_fast": ema_fast, "ema_slow": ema_slow, "ema200": ema200,
                "atr": atr, "rsi": rsi, "macd_hist": macd_hist,
                "stoch_k": stoch_k, "adx_val": adx_val,
            }.items()
        }
        bools = {
            name: s.fillna(False).to_numpy(dtype=bool)
            for name, s in {
                "weekly_bull": weekly_bull, "weekly_bear": weekly_bear, "adx_ok": adx_ok,
                "macd_long": macd_long, "macd_short": macd_short,
                "std_long_sig": std_long_sig, "std_short_sig": std_short_sig,
                "brk_long": brk_long, "brk_short": brk_short,
                "cont_pull_long_raw": cont_pull_long_raw, "cont_pull_short_raw": cont_pull_short_raw,
                "mom_cont_long_raw": mom_cont_long_raw, "mom_cont_short_raw": mom_cont_short_raw,
                "rsi_good_long": rsi_good_long, "rsi_good_short": rsi_good_short,
                "rsi_good_brk_long": rsi_good_brk_long, "rsi_good_brk_short": rsi_good_brk_short,
                "sqz_override_long": sqz_override_long, "sqz_override_short": sqz_override_short,
                "sqz_vol_ok": sqz_vol_ok, "vol_spike": vol_spike,
                "macd_cross_under": macd_cross_under, "macd_cross_over": macd_cross_over,
                "rsi_cross_under_50": rsi_cross_under_50, "rsi_cross_over_50": rsi_cross_over_50,
                "stoch_cross_under": stoch_cross_under, "stoch_cross_over": stoch_cross_over,
                "is_squeeze": is_squeeze,
            }.items()
        }
        # already plain numpy bool arrays (computed from other numpy arrays
        # above, not pandas Series), so no .fillna()/.to_numpy() needed
        bools["confirm_long"] = confirm_long
        bools["confirm_short"] = confirm_short
        ph_price_arr = ph_price.to_numpy(dtype=float)
        pl_price_arr = pl_price.to_numpy(dtype=float)
        rsi_at_pivot_arr = rsi_at_pivot_confirm.to_numpy(dtype=float)

        weekly_na = weekly_ema.isna().to_numpy()

        # ------------------------------------------------------------------
        # Pre-pass: RSI divergence pivot tracking (independent of trade state)
        # ------------------------------------------------------------------
        bear_div_active = np.zeros(n, dtype=bool)
        bull_div_active = np.zeros(n, dtype=bool)
        ph1 = ph2 = ph1_rsi = ph2_rsi = None
        ph1_bar = ph2_bar = None
        pl1 = pl2 = pl1_rsi = pl2_rsi = None
        pl1_bar = pl2_bar = None
        last_bear_div_bar = -999
        last_bull_div_bar = -999

        for i in range(n):
            if not _isnan(ph_price_arr[i]):
                ph2, ph2_rsi, ph2_bar = ph1, ph1_rsi, ph1_bar
                ph1 = ph_price_arr[i]
                ph1_rsi = rsi_at_pivot_arr[i]
                ph1_bar = i - p.div_piv_right
                if (
                    ph2 is not None and ph2_bar is not None
                    and (ph1_bar - ph2_bar) <= p.div_max_spacing
                    and ph1 > ph2 and not _isnan(ph1_rsi) and not _isnan(ph2_rsi) and ph1_rsi < ph2_rsi
                ):
                    last_bear_div_bar = i

            if not _isnan(pl_price_arr[i]):
                pl2, pl2_rsi, pl2_bar = pl1, pl1_rsi, pl1_bar
                pl1 = pl_price_arr[i]
                pl1_rsi = rsi_at_pivot_arr[i]
                pl1_bar = i - p.div_piv_right
                if (
                    pl2 is not None and pl2_bar is not None
                    and (pl1_bar - pl2_bar) <= p.div_max_spacing
                    and pl1 < pl2 and not _isnan(pl1_rsi) and not _isnan(pl2_rsi) and pl1_rsi > pl2_rsi
                ):
                    last_bull_div_bar = i

            bear_div_active[i] = (i - last_bear_div_bar) <= p.div_active_bars
            bull_div_active[i] = (i - last_bull_div_bar) <= p.div_active_bars

        # ------------------------------------------------------------------
        # Main sequential trade-state loop
        # ------------------------------------------------------------------
        out = {
            key: np.full(n, np.nan)
            for key in (
                "trade_dir", "entry_price", "stop_price", "current_r", "signal_count",
            )
        }
        out_bool = {
            key: np.zeros(n, dtype=bool)
            for key in (
                "enter_long", "enter_short", "is_flip_long", "is_flip_short",
                "partial_exit", "full_exit", "hard_stop_hit", "trail_stop_hit",
                "time_exit", "atr_warning", "conflict_both_fire", "be_active",
                "weekly_bull_state", "weekly_bear_state", "weekly_warming_up",
            )
        }

        trade_dir = 0
        entry_price = entry_atr = entry_stop_mult = _NA
        entry_bar = -100
        partial_done = False
        atr_warn_fired = False
        stop_price = _NA
        track_extreme = _NA
        had_exit = False
        last_exit_bar = -999
        last_exit_dir = 0
        last_any_bar = -999
        last_long_bar = -999
        last_short_bar = -999

        close_a, open_a, high_a, low_a = arrs["close"], arrs["open"], arrs["high"], arrs["low"]
        ema_fast_a, ema_slow_a, atr_a, rsi_a, macd_hist_a = (
            arrs["ema_fast"], arrs["ema_slow"], arrs["atr"], arrs["rsi"], arrs["macd_hist"]
        )
        stoch_k_a, adx_val_a = arrs["stoch_k"], arrs["adx_val"]

        for i in range(n):
            dir0 = trade_dir
            is_long_live = dir0 == 1
            is_short_live = dir0 == -1
            is_live = dir0 != 0
            past_grace = is_live and (i - entry_bar) > p.grace_len

            bars_from_exit = i - last_exit_bar
            in_rev_window = had_exit and p.allow_reversal_bars > 0 and bars_from_exit <= p.allow_reversal_bars

            trend_long = (
                (not p.require_trend)
                or (close_a[i] > ema_slow_a[i] and ema_fast_a[i] > ema_slow_a[i])
                or (in_rev_window and last_exit_dir == -1)
            )
            trend_short = (
                (not p.require_trend)
                or (close_a[i] < ema_slow_a[i] and ema_fast_a[i] < ema_slow_a[i])
                or (in_rev_window and last_exit_dir == 1)
            )
            macd_long_i = bools["macd_long"][i]
            macd_short_i = bools["macd_short"][i]

            long_ready = (i - last_any_bar) > p.cooldown_bars
            short_ready = (i - last_any_bar) > p.cooldown_bars

            std_long = (
                bools["std_long_sig"][i] and trend_long and macd_long_i and bools["rsi_good_long"][i]
            ) or (bools["sqz_override_long"][i] and trend_long and macd_long_i)
            std_short = (
                bools["std_short_sig"][i] and trend_short and macd_short_i and bools["rsi_good_short"][i]
            ) or (bools["sqz_override_short"][i] and trend_short and macd_short_i)

            brk_long_full = bools["brk_long"][i] and (
                bools["rsi_good_brk_long"][i]
                or (p.use_sqz_rsi_override and bools["sqz_vol_ok"][i] and rsi_a[i] > p.rsi_min_long)
            )
            brk_short_full = bools["brk_short"][i] and (
                bools["rsi_good_brk_short"][i]
                or (p.use_sqz_rsi_override and bools["sqz_vol_ok"][i] and rsi_a[i] < p.rsi_max_short)
            )

            cont_long = (
                (bools["cont_pull_long_raw"][i] or bools["mom_cont_long_raw"][i])
                and trend_long and macd_long_i and bools["rsi_good_long"][i]
            )
            cont_short = (
                (bools["cont_pull_short_raw"][i] or bools["mom_cont_short_raw"][i])
                and trend_short and macd_short_i and bools["rsi_good_short"][i]
            )

            # --- exit-against signal components
            macd_cross_against = (is_long_live and bools["macd_cross_under"][i]) or (
                is_short_live and bools["macd_cross_over"][i]
            )
            rsi_cross_against = (is_long_live and bools["rsi_cross_under_50"][i]) or (
                is_short_live and bools["rsi_cross_over_50"][i]
            )
            price_against_now = (is_long_live and close_a[i] < ema_fast_a[i]) or (
                is_short_live and close_a[i] > ema_fast_a[i]
            )
            if i >= 1:
                price_against_prev = (dir0 == 1 and close_a[i - 1] >= ema_fast_a[i - 1]) or (
                    dir0 == -1 and close_a[i - 1] <= ema_fast_a[i - 1]
                )
            else:
                price_against_prev = False
            price_against_edge = price_against_now and price_against_prev
            vol_spike_against = (is_long_live and bools["vol_spike"][i] and close_a[i] < open_a[i]) or (
                is_short_live and bools["vol_spike"][i] and close_a[i] > open_a[i]
            )

            signal_count = (
                int(macd_cross_against) + int(rsi_cross_against) + int(price_against_edge) + int(vol_spike_against)
            )
            core_exit_sig = macd_cross_against or price_against_edge
            core_exit_met = (not p.require_core_exit) or core_exit_sig

            # --- partial-exit signal components
            if i >= 2:
                macd_shrink_against = (
                    is_long_live
                    and macd_hist_a[i] < macd_hist_a[i - 1] < macd_hist_a[i - 2]
                    and macd_hist_a[i] > 0
                ) or (
                    is_short_live
                    and macd_hist_a[i] > macd_hist_a[i - 1] > macd_hist_a[i - 2]
                    and macd_hist_a[i] < 0
                )
            else:
                macd_shrink_against = False
            div_against = (is_long_live and bear_div_active[i]) or (is_short_live and bull_div_active[i])
            stoch_against = (is_long_live and bools["stoch_cross_under"][i] and stoch_k_a[i] > 60) or (
                is_short_live and bools["stoch_cross_over"][i] and stoch_k_a[i] < 40
            )

            partial_sig_count = int(macd_shrink_against) + int(div_against) + int(stoch_against)
            partial_signal_raw = partial_sig_count >= 2

            trend_still_intact = (
                is_long_live and close_a[i] > ema_slow_a[i] and ema_fast_a[i] > ema_slow_a[i] and macd_hist_a[i] > 0
            ) or (
                is_short_live and close_a[i] < ema_slow_a[i] and ema_fast_a[i] < ema_slow_a[i] and macd_hist_a[i] < 0
            )

            r_denom = entry_atr * entry_stop_mult if not _isnan(entry_atr) and not _isnan(entry_stop_mult) else _NA
            if is_live and not _isnan(r_denom) and r_denom > 0:
                current_r = (
                    (close_a[i] - entry_price) / r_denom if dir0 == 1 else (entry_price - close_a[i]) / r_denom
                )
            else:
                current_r = 0.0
            min_r_reached = current_r >= 1.0

            full_exit_sig = is_live and past_grace and signal_count >= p.exit_threshold and core_exit_met

            partial_sig_based = (
                is_live and past_grace and not partial_done and not full_exit_sig and partial_signal_raw
                and signal_count < p.exit_threshold and (not trend_still_intact or min_r_reached)
            )
            partial_at_target = (
                p.use_partial_at_r and is_live and past_grace and not partial_done and not full_exit_sig
                and current_r >= p.partial_target_r
            )
            partial_any = partial_sig_based or partial_at_target

            be_arm_same_bar = partial_any and not p.use_intrabar_stop
            if i >= 1 and is_live and not _isnan(r_denom) and r_denom > 0:
                prev_close_r = (
                    (close_a[i - 1] - entry_price) / r_denom
                    if dir0 == 1
                    else (entry_price - close_a[i - 1]) / r_denom
                )
            else:
                prev_close_r = 0.0
            be_intrabar_ok = (not p.use_intrabar_stop) or prev_close_r >= 1.0

            if (
                p.use_hard_stop and p.use_breakeven_stop and is_live and not _isnan(stop_price)
                and (partial_done or be_arm_same_bar) and min_r_reached and be_intrabar_ok
            ):
                stop_price = max(stop_price, entry_price) if dir0 == 1 else min(stop_price, entry_price)

            chand_atr = (atr_a[i - 1] if i >= 1 else atr_a[i]) if p.use_intrabar_stop else atr_a[i]
            if (
                p.use_hard_stop and p.use_chandelier and is_live and not _isnan(stop_price)
                and not _isnan(track_extreme) and current_r >= p.chand_arm_r
            ):
                chand_stop = (
                    track_extreme - p.chand_mult * chand_atr
                    if dir0 == 1
                    else track_extreme + p.chand_mult * chand_atr
                )
                stop_price = max(stop_price, chand_stop) if dir0 == 1 else min(stop_price, chand_stop)

            stop_breached_long = is_long_live and not _isnan(stop_price) and (
                low_a[i] <= stop_price if p.use_intrabar_stop else close_a[i] < stop_price
            )
            stop_breached_short = is_short_live and not _isnan(stop_price) and (
                high_a[i] >= stop_price if p.use_intrabar_stop else close_a[i] > stop_price
            )
            hard_stop_hit = p.use_hard_stop and is_live and (stop_breached_long or stop_breached_short)

            if i >= 1:
                trail_raw = (
                    is_long_live and close_a[i] < ema_slow_a[i] and close_a[i - 1] < ema_slow_a[i - 1]
                ) or (
                    is_short_live and close_a[i] > ema_slow_a[i] and close_a[i - 1] > ema_slow_a[i - 1]
                )
            else:
                trail_raw = False
            trail_stop_hit = p.use_trail_stop and is_live and past_grace and partial_done and trail_raw and not hard_stop_hit

            time_exit_raw = (
                p.use_time_exit and is_live and past_grace and (i - entry_bar) >= p.time_exit_bars
                and abs(current_r) < p.time_exit_max_r
            )
            time_exit = time_exit_raw and not hard_stop_hit and not trail_stop_hit and not full_exit_sig

            full_exit = full_exit_sig and not hard_stop_hit and not trail_stop_hit
            partial_exit = partial_any and not hard_stop_hit and not trail_stop_hit and not time_exit
            exit_event = hard_stop_hit or trail_stop_hit or full_exit or time_exit

            if is_live:
                if dir0 == 1:
                    track_extreme = max(track_extreme, high_a[i]) if not _isnan(track_extreme) else high_a[i]
                else:
                    track_extreme = min(track_extreme, low_a[i]) if not _isnan(track_extreme) else low_a[i]

            atr_below_now = is_live and not _isnan(entry_atr) and atr_a[i] < entry_atr * 0.75
            atr_above_prev = is_live and not _isnan(entry_atr) and i >= 1 and atr_a[i - 1] >= entry_atr * 0.75
            atr_warning = (
                is_live and past_grace and atr_below_now and atr_above_prev
                and not atr_warn_fired and not exit_event and not partial_exit
            )

            # --- entry signals
            weekly_ok_long = bools["weekly_bull"][i] or (p.allow_rev_no_weekly and in_rev_window and last_exit_dir == -1)
            weekly_ok_short = bools["weekly_bear"][i] or (p.allow_rev_no_weekly and in_rev_window and last_exit_dir == 1)
            adx_ok_i = bools["adx_ok"][i]
            confirm_long_i = bools["confirm_long"][i]
            confirm_short_i = bools["confirm_short"][i]

            setup_long = std_long or brk_long_full or cont_long
            setup_short = std_short or brk_short_full or cont_short

            # v9.6: the confirmation stack gates NEW entries only (raw_long/
            # raw_short), same as Pine's rawLong = rawLongPre and confirmLong.
            raw_long_pre = setup_long and weekly_ok_long and long_ready and adx_ok_i
            raw_short_pre = setup_short and weekly_ok_short and short_ready and adx_ok_i
            raw_long = raw_long_pre and confirm_long_i
            raw_short = raw_short_pre and confirm_short_i

            prior_bar_exited = had_exit and last_exit_bar == i - 1
            reversal_to_long_pre = (
                prior_bar_exited and last_exit_dir == -1 and setup_long and weekly_ok_long and adx_ok_i
            )
            reversal_to_short_pre = (
                prior_bar_exited and last_exit_dir == 1 and setup_short and weekly_ok_short and adx_ok_i
            )
            reversal_to_long = reversal_to_long_pre and (confirm_long_i or not e.apply_to_rev)
            reversal_to_short = reversal_to_short_pre and (confirm_short_i or not e.apply_to_rev)

            long_wants = raw_long or reversal_to_long
            short_wants = raw_short or reversal_to_short
            dir_bias = 1 if long_wants else (-1 if short_wants else 0)
            conflict_both_fire = long_wants and short_wants

            enter_long = dir_bias == 1 and dir0 != 1 and not exit_event and not partial_exit
            enter_short = dir_bias == -1 and dir0 != -1 and not exit_event and not partial_exit
            is_flip_long = enter_long and dir0 == -1
            is_flip_short = enter_short and dir0 == 1

            # --- state mutations, in script order
            if exit_event:
                had_exit = True
                last_exit_bar = i
                last_exit_dir = dir0
                trade_dir = 0
                partial_done = False
                entry_price = entry_atr = entry_stop_mult = _NA
                stop_price = _NA
                track_extreme = _NA
                atr_warn_fired = False

            if partial_exit:
                partial_done = True

            if atr_warning:
                atr_warn_fired = True

            if enter_long:
                if dir0 == -1:
                    had_exit = True
                    last_exit_bar = i
                    last_exit_dir = -1
                trade_dir = 1
                entry_price = close_a[i]
                entry_atr = atr_a[i]
                entry_stop_mult = p.atr_stop_mult
                entry_bar = i
                partial_done = False
                atr_warn_fired = False
                stop_price = close_a[i] - atr_a[i] * p.atr_stop_mult
                track_extreme = high_a[i]
                last_long_bar = i
                last_any_bar = i

            if enter_short:
                if dir0 == 1:
                    had_exit = True
                    last_exit_bar = i
                    last_exit_dir = 1
                trade_dir = -1
                entry_price = close_a[i]
                entry_atr = atr_a[i]
                entry_stop_mult = p.atr_stop_mult
                entry_bar = i
                partial_done = False
                atr_warn_fired = False
                stop_price = close_a[i] + atr_a[i] * p.atr_stop_mult
                track_extreme = low_a[i]
                last_short_bar = i
                last_any_bar = i

            # --- record post-mutation state for this bar
            out["trade_dir"][i] = trade_dir
            out["entry_price"][i] = entry_price
            out["stop_price"][i] = stop_price
            out["current_r"][i] = current_r
            out["signal_count"][i] = signal_count
            out_bool["enter_long"][i] = enter_long
            out_bool["enter_short"][i] = enter_short
            out_bool["is_flip_long"][i] = is_flip_long
            out_bool["is_flip_short"][i] = is_flip_short
            out_bool["partial_exit"][i] = partial_exit
            out_bool["full_exit"][i] = full_exit
            out_bool["hard_stop_hit"][i] = hard_stop_hit
            out_bool["trail_stop_hit"][i] = trail_stop_hit
            out_bool["time_exit"][i] = time_exit
            out_bool["atr_warning"][i] = atr_warning
            out_bool["conflict_both_fire"][i] = conflict_both_fire
            out_bool["be_active"][i] = (
                p.use_breakeven_stop and trade_dir != 0 and not _isnan(entry_price) and not _isnan(stop_price)
                and ((stop_price >= entry_price) if trade_dir == 1 else (stop_price <= entry_price))
            )
            out_bool["weekly_warming_up"][i] = weekly_na[i] and p.require_weekly

        # ------------------------------------------------------------------
        # Assemble result frame
        # ------------------------------------------------------------------
        result = pd.DataFrame(index=df.index)
        result["close"] = c
        result["ema_fast"] = ema_fast
        result["ema_slow"] = ema_slow
        result["ema200"] = ema200
        result["weekly_ema"] = weekly_ema
        result["atr"] = atr
        result["rsi"] = rsi
        result["macd_hist"] = macd_hist
        result["stoch_k"] = stoch_k
        result["stoch_d"] = stoch_d
        result["adx"] = adx_val
        result["di_plus"] = di_plus
        result["di_minus"] = di_minus
        result["is_squeeze"] = is_squeeze
        result["sqz_dir"] = sqz_dir
        result["bear_div_active"] = bear_div_active
        result["bull_div_active"] = bull_div_active
        result["weekly_bull_raw"] = weekly_bull_raw
        result["weekly_bear_raw"] = weekly_bear_raw
        result["trend_bull_raw"] = trend_bull_raw
        result["trend_bear_raw"] = trend_bear_raw
        result["macd_bull_raw"] = macd_bull_raw
        result["macd_bear_raw"] = macd_bear_raw
        result["rsi_good_long"] = rsi_good_long
        result["rsi_good_short"] = rsi_good_short
        for key, arr in out.items():
            result[key] = arr
        for key, arr in out_bool.items():
            result[key] = arr

        result["exit_event"] = (
            out_bool["hard_stop_hit"] | out_bool["trail_stop_hit"] | out_bool["full_exit"] | out_bool["time_exit"]
        )

        result["trade_status"] = np.select(
            [
                out_bool["hard_stop_hit"], out_bool["trail_stop_hit"], out_bool["full_exit"],
                out_bool["time_exit"], out_bool["partial_exit"], out_bool["atr_warning"],
                out["trade_dir"] == 1, out["trade_dir"] == -1,
            ],
            [
                "HARD STOP", "TRAIL STOP", "FULL EXIT", "TIME EXIT", "PARTIAL 50%",
                "ATR WARNING", "LONG LIVE", "SHORT LIVE",
            ],
            default="FLAT",
        )
        return result


def load_daily(ticker: str, period: str = "3y") -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _print_latest_state(ticker: str, result: pd.DataFrame) -> None:
    row = result.iloc[-1]
    date = result.index[-1].date()
    print(f"\n=== {ticker}  ({date}) ===")
    print(f"Close            {row['close']:.2f}")
    print(f"Trend (200)      {'BULL' if row['close'] > row['ema200'] else 'BEAR'}")
    if row["weekly_warming_up"]:
        print("Weekly EMA 50    warming up (blocks entries)")
    else:
        print(f"Weekly EMA 50    {'BULL' if row['close'] > row['weekly_ema'] else 'BEAR'}  ({row['weekly_ema']:.2f})")
    print(f"EMA 21/50        {'BULL' if row['ema_fast'] > row['ema_slow'] else 'BEAR'}")
    print(f"MACD             {'BULL' if row['macd_hist'] > 0 else 'BEAR'} ({row['macd_hist']:.3f})")
    print(f"ADX (14)         {row['adx']:.1f}")
    print(f"RSI              {row['rsi']:.1f}")
    stoch_ob_os = "  OB" if row["stoch_k"] > 80 else "  OS" if row["stoch_k"] < 20 else ""
    print(f"Stoch %K         {row['stoch_k']:.0f}{stoch_ob_os}")
    if row["is_squeeze"]:
        sqz_txt = f"COILING {'UP' if row['sqz_dir'] > 0 else 'DOWN'}"
    else:
        sqz_txt = f"FIRED {'UP' if row['sqz_dir'] > 0 else 'DOWN'}"
    print(f"BB Squeeze       {sqz_txt}")
    if row["bear_div_active"]:
        print("RSI Divergence   BEARISH (vs longs)")
    elif row["bull_div_active"]:
        print("RSI Divergence   BULLISH (vs shorts)")
    else:
        print("RSI Divergence   none")
    print(f"ATR              {row['atr']:.2f}")
    print(f"Stop price       {row['stop_price']:.2f}" if not math.isnan(row["stop_price"]) else "Stop price       none")
    print(f"Trade status     {row['trade_status']}")
    print(f"Signals against  {int(row['signal_count'])} / 4")

    recent = result.tail(15)
    events = recent[
        recent["enter_long"] | recent["enter_short"] | recent["full_exit"]
        | recent["hard_stop_hit"] | recent["trail_stop_hit"] | recent["time_exit"] | recent["partial_exit"]
    ]
    if len(events):
        print("\nRecent signals (last 15 bars):")
        for idx, r in events.iterrows():
            label = (
                "ENTER LONG" if r["enter_long"] else
                "ENTER SHORT" if r["enter_short"] else
                "HARD STOP" if r["hard_stop_hit"] else
                "TRAIL STOP" if r["trail_stop_hit"] else
                "FULL EXIT" if r["full_exit"] else
                "TIME EXIT" if r["time_exit"] else
                "PARTIAL EXIT"
            )
            print(f"  {idx.date()}  {label}  close={r['close']:.2f}")


if __name__ == "__main__":
    import sys

    tickers = sys.argv[1:] or ["AAPL"]
    engine = HolyGrailEngine()
    for t in tickers:
        data = load_daily(t)
        res = engine.run(data)
        _print_latest_state(t, res)
