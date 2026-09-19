"""
Market Desk — Part 1: LEARN
============================

What this file does, in one sentence: it looks at the days that have already
happened and settled, and turns them into three numbers the next-day decision
needs — how wrong the forecast usually is, how much of the forecast the desk
usually commits a day ahead, and which day-ahead market usually pays more —
each one broken down by time of day.

This is a faithful line-by-line port of the "learn" stage of the production
engine (engine/compare_app.js, functions learn / mirrorRatio / segmentChoice /
prepDay / asOfCutoff / priorDays), verified against the same engine's output
on the real 29-day reference workbook.

It has no knowledge of prices, settlement, or bidding — that is Part 2
(predict_split.py). This file only answers: "based on history, what should
tomorrow's decision expect?"

See Learn_and_Predict_Explained.md for the plain-language version of
everything below, with a picture.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import statistics

# --------------------------------------------------------------------------
# The four time-of-day clusters. Every one of the day's 96 fifteen-minute
# blocks (block 1 = 00:00-00:15 ... block 96 = 23:45-00:00) falls into
# exactly one of these. Grouping by cluster instead of by exact block is
# what lets five or six days of history produce a usable statistic per
# 15-minute slot — five days is only 5 samples for block 42 alone, but it is
# 5 days x 32 blocks = 160 samples for the whole C-2 cluster.
# --------------------------------------------------------------------------
CLUSTERS = ["C-1", "C-2", "C-3", "C-4"]
CLUSTER_LABEL = {
    "C-1": "C-1 - 06:00-10:00 (morning ramp)",
    "C-2": "C-2 - 10:00-18:00 (midday)",
    "C-3": "C-3 - 18:00-22:00 (evening peak)",
    "C-4": "C-4 - 22:00-06:00 (night)",
}


def cluster_of(block: int) -> str:
    """Which of the four time-of-day clusters a 15-minute block (1-96) falls in."""
    if 25 <= block <= 40:
        return "C-1"
    if 41 <= block <= 72:
        return "C-2"
    if 73 <= block <= 88:
        return "C-3"
    return "C-4"


# --------------------------------------------------------------------------
# One day's worth of block-wise data, already parsed from the workbook.
# This is the shared shape both Learn and Predict operate on. In production
# this is produced by the Ingest stage (not this file) from the customer's
# workbook; here it is a plain container so this file has no dependency
# on file parsing at all.
# --------------------------------------------------------------------------
@dataclass
class Day:
    date: str
    blk: list[int]                       # 1..96
    da_sch: list[float]                  # day-ahead forecast, MW, per block
    avc: list[float]                     # available capacity, MW, per block
    act: Optional[list[float]] = None    # actual injected power, MW (None for a day not yet run)
    dac: list[float] = field(default_factory=list)   # desk's bilateral DAC volume, MW (0 if none)
    dam: list[float] = field(default_factory=list)   # desk's DAM volume, MW (0 if unknown / not the desk's own day)
    gdam: list[float] = field(default_factory=list)  # desk's G-DAM volume, MW
    mcp_gdam: Optional[list[float]] = None  # G-DAM clearing price, Rs/MWh
    mcp_dam: Optional[list[float]] = None   # DAM clearing price, Rs/MWh
    mcp_rtm: Optional[list[float]] = None   # RTM clearing price, Rs/MWh
    has_split: bool = False              # does this day carry the desk's own DAC/DAM/GDAM split?
    has_prices: bool = False             # does this day carry all three day-ahead prices?

    # filled in by prep_day() below
    n: int = field(init=False)
    cl: list[str] = field(init=False)
    metered: Optional[list[float]] = None
    da_commit: Optional[list[float]] = None
    gdam_share: Optional[list[float]] = None
    pers: Optional[list[float]] = None
    err_da: Optional[list[float]] = None
    err_rtm: Optional[list[float]] = None

    def __post_init__(self):
        self.n = len(self.blk)
        self.cl = [cluster_of(b) for b in self.blk]
        if not self.dac:
            self.dac = [0.0] * self.n
        if not self.dam:
            self.dam = [0.0] * self.n
        if not self.gdam:
            self.gdam = [0.0] * self.n


LINE_LOSS = 0.034            # the desk's own line-loss factor, from the workbook header
REC_RS_PER_MWH = 300         # the REC uplift banked to a day-ahead trade, Rs/MWh


def _clip(x, lo, hi):
    return min(max(x, lo), hi)


def _gate_lag(block: int, lag: int, session_aware: bool) -> int:
    """The RTM gate looks back `lag` blocks (default 5 = 75 minutes) to find the
    most recent *settled* actual it can treat as a persistence estimate. On the
    second block of a 30-minute RTM session it has to look back one block further
    (a 'session-aware' +1), because that pair of blocks is gated together."""
    return lag + (1 if session_aware and block % 2 == 0 else 0)


def prep_day(day: Day, lag: int = 5, session_aware: bool = True) -> Day:
    """Turns raw actuals into the two error series Learn pools across days:
    err_da  = how far the metered output ended up from the day-ahead forecast
    err_rtm = how far the metered output ended up from a simple persistence
              estimate (the last actual seen `lag` blocks earlier) — this is
              the error a same-day, real-time decision would have made.
    Both stay None on a day with no actuals yet (a 'tomorrow' day being predicted)."""
    if day.act is None:
        return day
    day.metered = [a * (1 - LINE_LOSS) for a in day.act]
    day.da_commit = [day.dac[i] + day.dam[i] + day.gdam[i] for i in range(day.n)]
    day.gdam_share = [
        (day.gdam[i] / day.da_commit[i]) if day.da_commit[i] > 0 else 0.0
        for i in range(day.n)
    ]
    day.pers = []
    for i in range(day.n):
        lag_i = _gate_lag(day.blk[i], lag, session_aware)
        j = i - lag_i
        if j >= 0 and day.blk[i] - day.blk[j] == lag_i:
            day.pers.append(day.metered[j])
        else:
            day.pers.append(float("nan"))
    day.err_da = [
        (day.metered[i] - day.da_sch[i]) if day.da_sch[i] == day.da_sch[i] else float("nan")
        for i in range(day.n)
    ]
    day.err_rtm = [day.metered[i] - day.pers[i] for i in range(day.n)]
    return day


def as_of_cutoff(date: str) -> str:
    """What the desk (and the Lab) may use, as of 10:00 on D-1 for tomorrow's
    date D: everything up to and including D-2 has fully settled. D-1 is still
    running, so nothing from it is used. This one rule is what makes 'what we
    would have bid' and 'what we actually settled' numerically identical later —
    the information set never quietly grows between the two."""
    from datetime import date as _date, timedelta
    y, m, d = (int(x) for x in date.split("-"))
    cutoff = _date(y, m, d) - timedelta(days=2)
    return cutoff.isoformat()


def prior_days(all_days: dict[str, Day], date: str, lag: int = 5, session_aware: bool = True) -> list[Day]:
    """Every day on file up to and including the cutoff, prepared (prep_day applied).
    Walk-forward, always: nothing on or after D-1 ever enters this list."""
    cutoff = as_of_cutoff(date)
    return [
        prep_day(all_days[d], lag, session_aware)
        for d in sorted(all_days)
        if d <= cutoff
    ]


# --------------------------------------------------------------------------
# The built-in fallback sample. When fewer than 20 real block-observations
# exist for a cluster (e.g. the very first days of a new site, before enough
# history has accumulated), Learn borrows a starting distribution from a
# six-day sample on a comparable 102 MW wind SPV rather than making a
# statistic out of three or four numbers. This is not a permanent substitute
# — as soon as 20+ real observations exist for that cluster, real history
# takes over automatically, day by day.
# --------------------------------------------------------------------------
DEFAULT_ERR_DA_FRACTION = {  # fractions of AvC, at 19 fixed percentile points 5%..95%
    "C-1": [-0.4033, -0.3805, -0.3516, -0.3075, -0.2811, -0.2467, -0.2113, -0.1781, -0.1279,
            -0.0982, -0.0835, -0.0609, -0.0404, -0.026, -0.0128, 0.0011, 0.0117, 0.029, 0.1359],
    "C-2": [-0.3846, -0.2865, -0.2543, -0.1688, -0.1131, -0.0792, -0.053, -0.0371, -0.0295,
            -0.0154, -0.0048, 0.004, 0.0126, 0.0208, 0.0283, 0.033, 0.036, 0.0437, 0.051],
    "C-3": [-0.4054, -0.2392, -0.2127, -0.1777, -0.1537, -0.1306, -0.094, -0.0693, -0.026,
            -0.0154, -0.0045, 0.012, 0.0193, 0.0257, 0.0293, 0.044, 0.0503, 0.0748, 0.1019],
    "C-4": [-0.2791, -0.234, -0.2032, -0.1896, -0.1716, -0.1532, -0.1223, -0.1057, -0.089,
            -0.0745, -0.056, -0.0418, -0.0257, 0.0017, 0.0301, 0.0571, 0.0853, 0.1182, 0.1856],
}
DEFAULT_ERR_RTM_FRACTION = {
    "C-1": [-0.3113, -0.1619, -0.1036, -0.0861, -0.0411, -0.0216, -0.0025, 0.0148, 0.0428,
            0.0629, 0.0795, 0.0972, 0.118, 0.1249, 0.1381, 0.1674, 0.1885, 0.2255, 0.2794],
    "C-2": [-0.2448, -0.1399, -0.0856, -0.0363, -0.0184, -0.008, -0.0023, 0.0044, 0.0086,
            0.0178, 0.0282, 0.0334, 0.0417, 0.0607, 0.0944, 0.1265, 0.1498, 0.1938, 0.2784],
    "C-3": [-0.2574, -0.2206, -0.1858, -0.1723, -0.1613, -0.1397, -0.1293, -0.1148, -0.0807,
            -0.0651, -0.0539, -0.0399, -0.0341, -0.0118, 0.0029, 0.0197, 0.0439, 0.1209, 0.194],
    "C-4": [-0.2472, -0.2039, -0.1694, -0.1315, -0.1189, -0.0997, -0.0874, -0.0481, -0.0345,
            -0.0027, 0.0188, 0.0347, 0.0483, 0.0606, 0.0869, 0.1007, 0.1189, 0.1484, 0.1812],
}
DEFAULT_COMMIT_RATIO = {"C-1": 0.73, "C-2": 0.73, "C-3": 0.74, "C-4": 0.74}
DEFAULT_SEGMENT = {"C-1": 0, "C-2": 0, "C-3": 1, "C-4": 1}   # 0 = DAM, 1 = G-DAM
MIN_SAMPLES = 20


@dataclass
class LearnResult:
    err_da: dict[str, list[float]]        # per cluster: pooled (metered - forecast), MW
    err_rtm: dict[str, list[float]]       # per cluster: pooled (metered - persistence), MW
    fallback_da: list[str]                # clusters that had to borrow the built-in sample
    fallback_rtm: list[str]
    days_used: list[str]                  # which prior dates actually contributed data
    source: str                           # one-line, human-readable provenance string


def learn(prior: list[Day], target_day: Day) -> LearnResult:
    """The core of Part 1. For each of the four time clusters, pool every
    block-level forecast error and real-time error from every prior settled
    day, then fall back to the built-in sample for a cluster that is still
    too thin (fewer than 20 blocks of real history)."""
    err_da: dict[str, list[float]] = {}
    err_rtm: dict[str, list[float]] = {}
    fallback_da, fallback_rtm = [], []
    days_da, days_rtm = set(), set()

    avc_today = target_day.avc[0]
    for cl in CLUSTERS:
        eda, ert = [], []
        for o in prior:
            if o.err_da is None:
                continue
            for i in range(o.n):
                if o.cl[i] != cl:
                    continue
                if o.err_da[i] == o.err_da[i]:          # not NaN
                    eda.append(o.err_da[i]); days_da.add(o.date)
                if o.err_rtm[i] == o.err_rtm[i]:
                    ert.append(o.err_rtm[i]); days_rtm.add(o.date)
        if len(eda) < MIN_SAMPLES:
            eda = [x * avc_today for x in DEFAULT_ERR_DA_FRACTION[cl]]
            fallback_da.append(cl)
        if len(ert) < MIN_SAMPLES:
            ert = [x * avc_today for x in DEFAULT_ERR_RTM_FRACTION[cl]]
            fallback_rtm.append(cl)
        err_da[cl] = eda
        err_rtm[cl] = ert

    days_used = sorted(days_da)
    if days_used:
        span = days_used[0] if len(days_used) == 1 else f"{days_used[0]} → {days_used[-1]} ({len(days_used)} days)"
        source = f"prior day(s) {span}"
    else:
        source = "no prior day with forecast and actual - built-in sample (6 days, one 102 MW wind SPV)"
    if days_used and fallback_da:
        source += f"; built-in sample for {', '.join(fallback_da)} (too few blocks)"

    return LearnResult(err_da, err_rtm, fallback_da, fallback_rtm, days_used, source)


@dataclass
class MirrorRatio:
    ratio: dict[str, float]     # per cluster: desk's typical (day-ahead commit / forecast)
    days_used: list[str]
    used_fallback: bool


def mirror_ratio(prior: list[Day], k: int = 3) -> MirrorRatio:
    """The desk's own recent habit: of the forecast, what share does it usually
    commit a day ahead (versus leaving to real-time)? Computed per cluster, from
    whichever of the last `k` prior days actually carried the desk's own split."""
    pool = [o for o in prior if o.has_split and o.da_commit is not None][-k:]
    ratio = {}
    for cl in CLUSTERS:
        committed = forecast = 0.0
        for o in pool:
            for i in range(o.n):
                if o.cl[i] == cl and o.da_sch[i] > 0:
                    committed += o.da_commit[i]
                    forecast += o.da_sch[i]
        ratio[cl] = (committed / forecast) if forecast > 0 else DEFAULT_COMMIT_RATIO[cl]
    return MirrorRatio(ratio, [o.date for o in pool], used_fallback=not pool)


@dataclass
class SegmentChoice:
    g_flag: list[int]           # per block: 1 = G-DAM preferred, 0 = DAM preferred
    spread: list[float]         # per block: the learned average spread that produced the flag
    days_used: list[str]
    used_fallback: bool


def segment_choice(target_day: Day, prior: list[Day], k: int = 3) -> SegmentChoice:
    """Which day-ahead venue has recently paid more, block by block: G-DAM, or
    DAM plus the REC uplift? Averaged over the last `k` prior days that carry
    both prices, per exact block (not per cluster — the venue preference moves
    by time of day in a finer pattern than the four clusters capture)."""
    pool = [o for o in prior if o.has_prices][-k:]
    if not pool:
        return SegmentChoice(
            g_flag=[DEFAULT_SEGMENT[cl] for cl in target_day.cl],
            spread=[float("nan")] * target_day.n,
            days_used=[], used_fallback=True,
        )
    spread_sum: dict[int, float] = {}
    spread_ct: dict[int, int] = {}
    for o in pool:
        for i in range(o.n):
            b = o.blk[i]
            s = o.mcp_gdam[i] - (o.mcp_dam[i] + REC_RS_PER_MWH)
            if s == s:  # not NaN
                spread_sum[b] = spread_sum.get(b, 0.0) + s
                spread_ct[b] = spread_ct.get(b, 0) + 1
    g_flag, spread = [], []
    for b in target_day.blk:
        if spread_ct.get(b):
            avg = spread_sum[b] / spread_ct[b]
        else:
            avg = 0.0
        g_flag.append(1 if avg >= 0 else 0)
        spread.append(avg if spread_ct.get(b) else float("nan"))
    return SegmentChoice(g_flag, spread, [o.date for o in pool], used_fallback=False)


@dataclass
class LearnProfile:
    """Everything Part 2 (predict_split.py) needs, bundled with its own
    provenance so the profile can always say, on request, exactly which
    days it learned from."""
    date: str
    cutoff: str
    error: LearnResult
    commit: MirrorRatio
    segment: SegmentChoice


def build_learn_profile(all_days: dict[str, Day], target_day: Day, k: int = 3,
                         lag: int = 5, session_aware: bool = True) -> LearnProfile:
    """The single entry point: hand it every day on file and the day you are
    about to decide, and it returns the walk-forward profile Predict needs."""
    prior = prior_days(all_days, target_day.date, lag, session_aware)
    L = learn(prior, target_day)
    mir = mirror_ratio(prior, k)
    seg = segment_choice(target_day, prior, k)
    return LearnProfile(target_day.date, as_of_cutoff(target_day.date), L, mir, seg)
