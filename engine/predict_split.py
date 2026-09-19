"""
Market Desk — Part 2: PREDICT / SPLIT (the walk-forward statistical model)
============================================================================

What this file does, in one sentence: given tomorrow's forecast and the
profile Part 1 (learn_engine.py) built from history, it decides how much of
tomorrow's energy to commit a day ahead versus plan through real-time, and
which day-ahead market to route it through — block by block, for three
risk postures.

This is a faithful port of the "predict tomorrow's bid" stage of the
production engine (engine/compare_app.js, functions dayAheadVolume /
placeVolumes / DIALS), verified against that engine's output on the real
reference workbook.

The statistical part is `day_ahead_volume()`: for the Balanced and
Aggressive dials it does not predict a single number, it reads a *quantile*
off the learned error distribution built in Part 1 — e.g. "the level that
history's forecast error beat 25% of the time" — which is the standard
newsvendor technique for sizing a commitment under an asymmetric penalty
(the desk is charged differently for coming up short than for having a
surplus, so the "safest single guess" is not the average, it is a chosen
percentile of the error distribution).

Everything else in this file is arithmetic, not statistics: the "thumb
rule" in place_volumes() is a hard identity (day-ahead + real-time always
equals the forecast, exactly, block by block) enforced in code, not learned
or estimated.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .learn_engine import Day, LearnProfile, CLUSTERS


def _clip(x, lo, hi):
    return min(max(x, lo), hi)


def _quantile(values: list[float], p: float) -> float:
    """The exact quantile convention the production engine uses (numpy's
    default 'linear' interpolation) — the same p=0.25 must land on the same
    number here and in the JS engine, or the two systems would quietly
    disagree on what '25th percentile' means."""
    clean = [v for v in values if v == v]  # drop NaN
    if not clean:
        return float("nan")
    return float(np.quantile(clean, p, method="linear"))


# --------------------------------------------------------------------------
# The three dials. Each is nothing more than a choice of *method* for sizing
# the day-ahead commitment, plus (for the quantile method) which percentile
# to use. Nothing about the day-ahead/real-time boundary logic changes
# between dials — only these two parameters do.
# --------------------------------------------------------------------------
@dataclass
class Dial:
    label: str
    volume_method: str          # "mirror" or "quantile"
    alpha: Optional[float]      # the quantile to use, only when volume_method == "quantile"
    blurb: str


DIALS: dict[str, Dial] = {
    "conservative": Dial(
        "Conservative", "mirror", None,
        "Commits the forecast x the desk's own recent commit ratio, cluster by "
        "cluster; the rest of the same forecast is planned through real-time. "
        "Matches the desk's usual appetite without copying its exact schedule."
    ),
    "balanced": Dial(
        "Balanced", "quantile", 0.25,
        "Commits at the 25th percentile of the learned forecast-error "
        "distribution per cluster — a level history's error beat 1 day in 4. "
        "Day-ahead plus real-time always adds back to the forecast."
    ),
    "aggressive": Dial(
        "Aggressive", "quantile", 0.35,
        "Commits nearer the middle of the learned error distribution (35th "
        "percentile), leaving less to real-time. Day-ahead plus real-time "
        "always adds back to the forecast; only the boundary between them moves."
    ),
}


def day_ahead_volume(day: Day, profile: LearnProfile, dial: Dial) -> list[float]:
    """THE STATISTICAL STEP. Decide, block by block, how much of tomorrow's
    forecast to commit a day ahead (qda). Everything left over becomes the
    real-time leg once place_volumes() applies the thumb rule below.

    mirror   -> qda[i] = forecast[i] x (desk's own recent commit ratio for that block's cluster)
    quantile -> qda[i] = forecast[i] + (dial.alpha quantile of that cluster's learned forecast-error distribution)

    Both are clipped to [0, AvC] — a bid can never be negative or exceed the
    plant's available capacity for that block.
    """
    qda = []
    for i in range(day.n):
        cl = day.cl[i]
        f = day.da_sch[i]
        if dial.volume_method == "mirror":
            v = f * profile.commit.ratio[cl]
        else:
            v = f + _quantile(profile.error.err_da[cl], dial.alpha)
        qda.append(_clip(v, 0, day.avc[i]))
    return qda


@dataclass
class BidSheet:
    date: str
    dial: str
    blk: list[int]
    total: list[float]      # = the forecast, unchanged — the shared starting point
    day_ahead_qty: list[float]   # qda: the day-ahead leg actually filed (dac + dam + gdam)
    gdam: list[float]
    dam: list[float]
    dac: list[float]        # bilateral DAC — 0 for the Lab's own recommendation (see note below)
    rtm: list[float]        # what the thumb rule leaves for real-time = total - day_ahead_qty
    g_flag: list[int]       # which venue each block's day-ahead leg preferred (1=G-DAM, 0=DAM)

    @property
    def total_mwh(self) -> float:
        return sum(self.total) / 4

    @property
    def gdam_dam_mwh(self) -> float:
        return sum(self.gdam) / 4 + sum(self.dam) / 4

    @property
    def rtm_mwh(self) -> float:
        return sum(self.rtm) / 4


def place_volumes(day: Day, qda: list[float], g_flag: list[int]) -> BidSheet:
    """THE THUMB RULE. A hard, non-negotiable identity, enforced here in code
    rather than left to the model or a downstream step to get right:

        day-ahead leg (dac + dam + gdam) + real-time leg (rtm) == the forecast

    exactly, on every one of the 96 blocks, with nothing held back for line
    loss, forecaster optimism, or the DSM free band. Only WHERE the day-ahead
    energy goes (which venue) and HOW MUCH of the forecast is committed a day
    ahead at all are decided upstream, by day_ahead_volume() and
    segment_choice(); this function just places the fixed total.

    The Lab's own recommendation never books bilateral DAC (that is the
    desk's private trade, agreed after 13:00 on D-1 — unknown at bid time),
    so dac is always 0 here and every day-ahead unit goes to G-DAM or DAM.
    """
    total = day.da_sch
    dac_out, dam_out, gdam_out, rtm_out = [], [], [], []
    for i in range(day.n):
        t = max(total[i], 0.0) if total[i] == total[i] else day.avc[i]
        q = _clip(qda[i], 0, t)
        rest = q               # the Lab books no bilateral DAC of its own
        g = rest * g_flag[i]
        dac_out.append(0.0)
        gdam_out.append(g)
        dam_out.append(rest - g)
        rtm_out.append(t - q)
    return BidSheet(day.date, "", list(day.blk), list(total),
                     list(qda), gdam_out, dam_out, dac_out, rtm_out, list(g_flag))


def predict_tomorrow(day: Day, profile: LearnProfile, dial_name: str) -> BidSheet:
    """The single entry point for Part 2. Hand it tomorrow's forecast/AvC
    (as a Day with act=None), the LearnProfile Part 1 built for that date,
    and a dial name — get back the recommended 96-block bid sheet."""
    dial = DIALS[dial_name]
    qda = day_ahead_volume(day, profile, dial)
    sheet = place_volumes(day, qda, profile.segment.g_flag)
    sheet.dial = dial_name
    return sheet


def predict_all_dials(day: Day, profile: LearnProfile) -> dict[str, BidSheet]:
    """Convenience wrapper: run all three dials at once, exactly as the
    Split pane shows them side by side."""
    return {name: predict_tomorrow(day, profile, name) for name in DIALS}
