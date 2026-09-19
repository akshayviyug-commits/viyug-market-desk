"""
Block-wise features for the G-DAM / RTM price model.

Every feature for a delivery date D is built only from information the desk
has at 09:00 on D-1, under an explicit information policy:

  * G-DAM and DAM prices for D-1 were cleared on D-2, so they are known up to D-1.
  * RTM prices, ACP and metered actuals are only complete up to D-2.
  * The delivery day's own prices and actuals are NEVER read - only its
    calendar, its wind forecast and its available capacity.

The cutoffs live in `Policy`, so the same code can be run under a stricter
rule (everything to D-2) to show that nothing depends on the looser one.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

CAP = 10000.0           # exchange price cap, Rs/MWh - a real ceiling, not a placeholder
CAP_TOL = 1.0
N_BLOCKS = 96
ENERGY_FACTOR = 0.25

HOLIDAY_FILE = Path(__file__).resolve().parent.parent / "data" / "external" / "holidays_in.csv"

# Seven windows, as on the demo page (start hour, end hour).
WINDOWS = [(0, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21), (21, 24)]
WINDOW_LABELS = [f"{a:02d}:00-{b:02d}:00" for a, b in WINDOWS]
EVENING = slice(72, 88)     # blocks 73-88 = 18:00-22:00


@dataclass(frozen=True)
class Policy:
    lag_gdam: int = 1   # G-DAM / DAM prices known up to D-lag_gdam
    lag_rtm: int = 2    # RTM prices, ACP and actuals known up to D-lag_rtm


@dataclass
class Panel:
    """Day x block matrices (NaN where unknown)."""
    dates: pd.DatetimeIndex
    gdam: np.ndarray
    dam: np.ndarray
    rtm: np.ndarray
    acp: np.ndarray
    actual: np.ndarray      # metered injection after line loss, NaN on unsettled days
    final: np.ndarray       # desk's final schedule
    fc: np.ndarray          # day-ahead wind forecast
    avc: np.ndarray
    settled: np.ndarray     # bool per day
    extra: dict = field(default_factory=dict)   # optional further day x block matrices


def block_window(block: int) -> int:
    hour = (int(block) - 1) * 0.25
    for i, (a, b) in enumerate(WINDOWS):
        if a <= hour < b:
            return i
    return len(WINDOWS) - 1


def block_time(block: int) -> str:
    m = (int(block) - 1) * 15
    return f"{m // 60:02d}:{m % 60:02d}"


def build_panel(df: pd.DataFrame, settled_dates: list) -> Panel:
    """`df` is the validated frame from engine.loader (canonical column names)."""
    settled_set = {pd.Timestamp(d) for d in settled_dates}

    def mat(col):
        p = df.pivot(index="date", columns="block", values=col).sort_index()
        return p.reindex(columns=range(1, N_BLOCKS + 1)).to_numpy(dtype=float, copy=True)

    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    settled = np.array([d in settled_set for d in dates])
    actual = mat("actual_ll")
    final = mat("final_schedule")
    actual[~settled] = np.nan     # 0 on a forecast-only day means "not yet", not "zero output"
    return Panel(
        dates=dates, gdam=mat("gdam_mcp"), dam=mat("dam_mcp"), rtm=mat("rtm_mcp"), acp=mat("acp"),
        actual=actual, final=final, fc=mat("day_ahead_schedule"), avc=mat("avc"), settled=settled,
    )


@lru_cache(maxsize=1)
def _holiday_set() -> frozenset:
    if not HOLIDAY_FILE.exists():
        return frozenset()
    h = pd.read_csv(HOLIDAY_FILE, parse_dates=["date"])
    return frozenset(pd.to_datetime(h["date"]).dt.normalize())


def holiday_name(ts) -> str | None:
    if not HOLIDAY_FILE.exists():
        return None
    h = pd.read_csv(HOLIDAY_FILE, parse_dates=["date"])
    row = h[h["date"] == pd.Timestamp(ts).normalize()]
    return None if row.empty else str(row["name"].iloc[0])


def is_holiday(ts) -> bool:
    return pd.Timestamp(ts).normalize() in _holiday_set()


def season_active(panel: Panel) -> bool:
    """A month/season feature is only learnable once the file spans more than one month."""
    return len({(d.year, d.month) for d in panel.dates}) >= 2


# ----------------------------------------------------------------------------- helpers
def _nanmean(x, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(x, axis=axis)


def _day_row(panel: Panel, arr: np.ndarray, day) -> np.ndarray:
    m = panel.dates == pd.Timestamp(day)
    return arr[m][0].copy() if m.any() else np.full(N_BLOCKS, np.nan)


def _window(panel: Panel, arr: np.ndarray, cutoff, n_days: int) -> np.ndarray:
    lo = pd.Timestamp(cutoff) - pd.Timedelta(days=n_days - 1)
    m = (panel.dates >= lo) & (panel.dates <= pd.Timestamp(cutoff))
    return arr[m]


def _upto(panel: Panel, arr: np.ndarray, cutoff) -> np.ndarray:
    return arr[panel.dates <= pd.Timestamp(cutoff)]


def _smooth(v: np.ndarray, k: int = 2) -> np.ndarray:
    out = np.full(v.shape, np.nan)
    for i in range(len(v)):
        out[i] = _nanmean(v[max(0, i - k): i + k + 1])
    return out


def _runs(flag: np.ndarray, known: bool) -> tuple[np.ndarray, np.ndarray]:
    """Length of the run of consecutive True blocks containing each block, and
    the block's position inside it (1 = first). Unknown history -> NaN."""
    if not known:
        return np.full(N_BLOCKS, np.nan), np.full(N_BLOCKS, np.nan)
    length, pos = np.zeros(N_BLOCKS), np.zeros(N_BLOCKS)
    i = 0
    while i < N_BLOCKS:
        if flag[i]:
            j = i
            while j < N_BLOCKS and flag[j]:
                j += 1
            length[i:j] = j - i
            pos[i:j] = np.arange(1, j - i + 1)
            i = j
        else:
            i += 1
    return length, pos


def _capped(v: np.ndarray) -> np.ndarray:
    out = (v >= CAP - CAP_TOL).astype(float)
    out[np.isnan(v)] = np.nan
    return out


def _level_ratio(panel: Panel, arr: np.ndarray, cutoff, target) -> float:
    """Mean daily price level on the same weekday as `target`, relative to all days
    up to `cutoff`. NaN until that weekday has been seen."""
    m = panel.dates <= pd.Timestamp(cutoff)
    days, rows = panel.dates[m], arr[m]
    if len(days) == 0:
        return np.nan
    level = _nanmean(rows, axis=1)
    same = np.array([d.dayofweek == pd.Timestamp(target).dayofweek for d in days])
    if not same.any() or np.isnan(level).all():
        return np.nan
    return float(_nanmean(level[same]) / _nanmean(level))


# ----------------------------------------------------------------------------- features
FEATURES = [
    # calendar
    "block", "blk_sin", "blk_cos", "window", "dow", "is_weekend", "is_holiday", "hol_prev", "hol_next",
    "dow_level_ratio",
    # G-DAM / DAM history (to D-1)
    "g_l1", "g_m3", "g_m7", "g_all", "g_nb", "g_day_l1", "g_trend", "dm_l1", "dm_m3", "g_minus_dm_l1",
    "g_cap_l1", "g_cap_rate7", "g_run_len", "g_run_pos", "g_maxrun", "g_cap_run_len",
    # RTM history (to D-2)
    "r_l2", "r_m3", "r_all", "r_nb", "r_day_l2", "r_cap_l2", "r_cap_rate7", "r_run_len", "sp_m3",
    # DSM / plant
    "acp_l2", "acp_day_l2", "dev_pct_l2", "eve_short_rate", "cf_l2",
    # tomorrow's wind forecast
    "fc", "fc_pct", "fc_day_mwh", "fc_ramp4", "fc_shape",
]

# Features that exist only where the plant's own data does (its forecast, actuals and settlement).
PLANT_FEATURES = ["acp_l2", "acp_day_l2", "dev_pct_l2", "eve_short_rate", "cf_l2",
                  "fc", "fc_pct", "fc_day_mwh", "fc_ramp4", "fc_shape"]
# Price-history features only: they need no plant data, so they work on any day that has prices.
MARKET_FEATURES = [f for f in FEATURES if f not in PLANT_FEATURES]


def features_for_date(panel: Panel, target, fc: np.ndarray | None = None, avc: np.ndarray | None = None,
                      policy: Policy = Policy()) -> pd.DataFrame:
    """96-row feature frame for delivery date `target`. Reads panel rows strictly
    before the policy cutoffs; the target row's own prices/actuals are never touched."""
    target = pd.Timestamp(target).normalize()
    cg = target - pd.Timedelta(days=policy.lag_gdam)     # last day G-DAM/DAM is known
    cr = target - pd.Timedelta(days=policy.lag_rtm)      # last day RTM/ACP/actuals are known
    if fc is None:
        fc = _day_row(panel, panel.fc, target)
    if avc is None:
        avc = _day_row(panel, panel.avc, target)
    fc = np.asarray(fc, dtype=float)
    avc = np.asarray(avc, dtype=float)
    blocks = np.arange(1, N_BLOCKS + 1)
    f = pd.DataFrame({"block": blocks})

    # calendar
    ang = 2 * np.pi * (blocks - 1) / N_BLOCKS
    f["blk_sin"], f["blk_cos"] = np.sin(ang), np.cos(ang)
    f["window"] = [block_window(b) for b in blocks]
    f["dow"] = target.dayofweek
    f["is_weekend"] = float(target.dayofweek >= 5)
    f["is_holiday"] = float(is_holiday(target))
    f["hol_prev"] = float(is_holiday(target - pd.Timedelta(days=1)))
    f["hol_next"] = float(is_holiday(target + pd.Timedelta(days=1)))
    f["dow_level_ratio"] = _level_ratio(panel, panel.gdam, cg, target)

    # G-DAM / DAM
    G = _upto(panel, panel.gdam, cg)
    g1 = _day_row(panel, panel.gdam, cg)
    g3, g7 = _window(panel, panel.gdam, cg, 3), _window(panel, panel.gdam, cg, 7)
    f["g_l1"] = g1
    f["g_m3"] = _nanmean(g3) if len(g3) else np.nan
    f["g_m7"] = _nanmean(g7) if len(g7) else np.nan
    f["g_all"] = _nanmean(G) if len(G) else np.nan
    f["g_nb"] = _smooth(g1)
    f["g_day_l1"] = _nanmean(g1)
    prior = _window(panel, panel.gdam, cg - pd.Timedelta(days=1), 3)
    f["g_trend"] = (_nanmean(g1) - _nanmean(prior)) if len(prior) else np.nan
    d1 = _day_row(panel, panel.dam, cg)
    dm3 = _window(panel, panel.dam, cg, 3)
    f["dm_l1"] = d1
    f["dm_m3"] = _nanmean(dm3) if len(dm3) else np.nan
    f["g_minus_dm_l1"] = g1 - d1
    f["g_cap_l1"] = _capped(g1)
    f["g_cap_rate7"] = _nanmean(_capped(g7)) if len(g7) else np.nan
    known = bool(np.isfinite(g1).any())
    thr = float(np.nanquantile(G, 0.75)) if len(G) and np.isfinite(G).any() else np.nan
    high = (g1 >= thr) if np.isfinite(thr) else np.zeros(N_BLOCKS, bool)
    f["g_run_len"], f["g_run_pos"] = _runs(high & np.isfinite(g1), known and np.isfinite(thr))
    f["g_maxrun"] = float(np.nanmax(f["g_run_len"])) if f["g_run_len"].notna().any() else np.nan
    f["g_cap_run_len"], _ = _runs(np.nan_to_num(_capped(g1)) > 0, known)

    # RTM (to D-2)
    R = _upto(panel, panel.rtm, cr)
    r2 = _day_row(panel, panel.rtm, cr)
    r3, r7 = _window(panel, panel.rtm, cr, 3), _window(panel, panel.rtm, cr, 7)
    f["r_l2"] = r2
    f["r_m3"] = _nanmean(r3) if len(r3) else np.nan
    f["r_all"] = _nanmean(R) if len(R) else np.nan
    f["r_nb"] = _smooth(r2)
    f["r_day_l2"] = _nanmean(r2)
    f["r_cap_l2"] = _capped(r2)
    f["r_cap_rate7"] = _nanmean(_capped(r7)) if len(r7) else np.nan
    rk = bool(np.isfinite(r2).any())
    rthr = float(np.nanquantile(R, 0.75)) if len(R) and np.isfinite(R).any() else np.nan
    rhigh = (r2 >= rthr) if np.isfinite(rthr) else np.zeros(N_BLOCKS, bool)
    f["r_run_len"], _ = _runs(rhigh & np.isfinite(r2), rk and np.isfinite(rthr))
    gs, rs = _window(panel, panel.gdam, cr, 3), _window(panel, panel.rtm, cr, 3)
    f["sp_m3"] = _nanmean(rs - gs) if len(rs) else np.nan

    # DSM / plant (to D-2)
    f["acp_l2"] = _day_row(panel, panel.acp, cr)
    f["acp_day_l2"] = _nanmean(f["acp_l2"].to_numpy())
    a2, fin2, avc2 = (_day_row(panel, x, cr) for x in (panel.actual, panel.final, panel.avc))
    with np.errstate(divide="ignore", invalid="ignore"):
        f["dev_pct_l2"] = _nanmean(np.abs(a2 - fin2) / avc2)
        f["cf_l2"] = _nanmean(a2) / _nanmean(avc2) if np.isfinite(avc2).any() else np.nan
    act7, fin7 = _window(panel, panel.actual, cr, 7), _window(panel, panel.final, cr, 7)
    if len(act7):
        dev = act7[:, EVENING] - fin7[:, EVENING]
        day_dev = _nanmean(dev, axis=1)
        ok = np.isfinite(day_dev)
        f["eve_short_rate"] = float((day_dev[ok] < 0).mean()) if ok.any() else np.nan
    else:
        f["eve_short_rate"] = np.nan

    # tomorrow's wind forecast
    f["fc"] = fc
    with np.errstate(divide="ignore", invalid="ignore"):
        f["fc_pct"] = fc / avc
        f["fc_shape"] = fc / _nanmean(fc)
    f["fc_day_mwh"] = float(np.nansum(fc) * ENERGY_FACTOR)
    ramp = np.zeros(N_BLOCKS)
    ramp[4:] = fc[4:] - fc[:-4]
    f["fc_ramp4"] = ramp
    return f[FEATURES]


def slice_panel(panel: Panel, start, end=None) -> Panel:
    """The panel restricted to dates in [start, end]."""
    m = panel.dates >= pd.Timestamp(start)
    if end is not None:
        m &= panel.dates <= pd.Timestamp(end)
    return Panel(
        dates=panel.dates[m], gdam=panel.gdam[m], dam=panel.dam[m], rtm=panel.rtm[m], acp=panel.acp[m],
        actual=panel.actual[m], final=panel.final[m], fc=panel.fc[m], avc=panel.avc[m], settled=panel.settled[m],
        extra={k: v[m] for k, v in panel.extra.items()},
    )


def build_training_frame(panel: Panel, policy: Policy = Policy(), extra_fn=None) -> pd.DataFrame:
    """One row per (date, block) for every date on file, with features built strictly
    from that date's own information cutoff, plus the realised prices as targets.
    `extra_fn(panel, date, policy)` may add further 96-row feature columns."""
    frames = []
    for ts in panel.dates:
        x = features_for_date(panel, ts, policy=policy)
        if extra_fn is not None:
            x = pd.concat([x, extra_fn(panel, ts, policy)], axis=1)
        x.insert(0, "date", ts)
        x["y_gdam"] = _day_row(panel, panel.gdam, ts)
        x["y_rtm"] = _day_row(panel, panel.rtm, ts)
        frames.append(x)
    return pd.concat(frames, ignore_index=True)
