"""
Replay of the split on days that have already settled.

For every settled day T in a range the split is made exactly as it would have been then (price model
trained only on what was known, guardrail from Learn's walk-forward profile) and scored against what
actually happened. Same accounting for every strategy:

    revenue  = G-DAM volume x G-DAM MCP  +  RTM volume x (RTM MCP + REC)
    net      = revenue - DSM(actual vs scheduled total) - the day's historical charges

The desk row uses the desk's own legs and final schedule, DAC valued at the DAC price, and REC on every
market except G-DAM. Both legs of our plan are sized off the same forecast, so DSM is identical across our
strategies and to "all G-DAM": differences between them are pure price capture. The replay does not model
RTM revisions, so it cannot credit the forecast-error guardrail for the deviation it might save.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bundle import DataBundle
from .dsm import ENERGY_FACTOR, dsm_charge
from .learn_predict import plan_day
from .price_forecast import Forecaster
from .split_price import DIAL_LABEL, DIAL_TILT, REC_DEFAULT, price_split

ALL_G = "All G-DAM"
HABIT = "Desk habit, no price view"
VOLUME_ONLY = "Volume-only method (Balanced)"
DESK = "Desk as settled (incl. its DAC leg)"


def _row(T, name, day, g, r, da, rtm, total, rec, share, dam=None, dam_px=None):
    dsm = float(np.sum(dsm_charge(day["actual_ll"].to_numpy(), total, day["avc"].to_numpy(), day["acp"].to_numpy())["total_dsm"]))
    rev_g = float(np.sum(da * g) * ENERGY_FACTOR)
    rev_r = float(np.sum(rtm * (r + rec)) * ENERGY_FACTOR)
    if dam is not None:
        rev_g += float(np.sum(dam * (dam_px + rec)) * ENERGY_FACTOR)
    charges = float(day["total_charges"].sum())
    return {"date": T, "strategy": name, "net_inr": rev_g + rev_r - dsm - charges, "dsm_inr": dsm, "da_share": share,
            "rev_g_inr": rev_g, "rev_r_inr": rev_r, "charges_inr": charges}


def replay(bundle: DataBundle, fc: Forecaster, start, end, rec: float = REC_DEFAULT, guardrail: bool = True) -> pd.DataFrame:
    """One row per (date, strategy): net revenue and its parts in Rs. Only days settled in the workbook."""
    df = bundle.workbook.df
    rows = []
    for T in bundle.workbook.settled_dates:
        T = pd.Timestamp(T)
        if T < pd.Timestamp(start) or T > pd.Timestamp(end):
            continue
        day = df[df["date"] == T].sort_values("block")
        g, r = day["gdam_mcp"].to_numpy(), day["rtm_mcp"].to_numpy()
        forecast = day["day_ahead_schedule"].to_numpy()
        prices = fc.forecast(T)
        zeros = np.zeros(96)

        rows.append(_row(T, ALL_G, day, g, r, forecast, zeros, forecast, rec, 1.0))

        out = price_split(bundle, T, prices, "conservative", rec, guardrail=guardrail, tilt=0.0)
        p = out["plan"].sort_values("block")
        rows.append(_row(T, HABIT, day, g, r, p["gdam_mw"].to_numpy(), p["rtm_mw"].to_numpy(), forecast, rec, out["summary"]["gdam_share"]))

        for dial in DIAL_TILT:
            out = price_split(bundle, T, prices, dial, rec, guardrail=guardrail)
            p = out["plan"].sort_values("block")
            rows.append(_row(T, DIAL_LABEL[dial], day, g, r, p["gdam_mw"].to_numpy(), p["rtm_mw"].to_numpy(), forecast, rec,
                             out["summary"]["gdam_share"]))

        lp = plan_day(bundle.days, T.strftime("%Y-%m-%d"), "balanced")["plan"].sort_values("block")
        rows.append(_row(T, VOLUME_ONLY, day, g, r, lp["gdam_mw"].to_numpy(), lp["rtm_mw"].to_numpy(), forecast, rec,
                         float(lp["da_mw"].sum() / lp["forecast_mw"].sum()), dam=lp["dam_mw"].to_numpy(), dam_px=day["dam_mcp"].to_numpy()))

        # the desk as it actually settled: its own legs, its final schedule, DAC at the DAC price
        dac_px = day["dac_mcp"].fillna(day["gdam_mcp"]).to_numpy()
        final = day["final_schedule"].to_numpy()
        dsm_d = float(np.sum(dsm_charge(day["actual_ll"].to_numpy(), final, day["avc"].to_numpy(), day["acp"].to_numpy())["total_dsm"]))
        rev_g = float(np.sum(day["gdam_sched"].to_numpy() * g + day["dam_sched"].to_numpy() * (day["dam_mcp"].to_numpy() + rec)
                             + day["dac_sched"].to_numpy() * (dac_px + rec)) * ENERGY_FACTOR)
        rev_r = float(np.sum(day["rtm_sched"].to_numpy() * (r + rec)) * ENERGY_FACTOR)
        charges = float(day["total_charges"].sum())
        rows.append({"date": T, "strategy": DESK, "net_inr": rev_g + rev_r - dsm_d - charges, "dsm_inr": dsm_d,
                     "da_share": float((day["gdam_sched"] + day["dam_sched"] + day["dac_sched"]).sum() / max(final.sum(), 1e-9)),
                     "rev_g_inr": rev_g, "rev_r_inr": rev_r, "charges_inr": charges})
    return pd.DataFrame(rows)


def summarise(rep: pd.DataFrame) -> pd.DataFrame:
    """Per strategy: total, per-day P10/P50/P90, gain over all-G-DAM, and how often it beat all-G-DAM."""
    base = rep[rep["strategy"] == ALL_G].set_index("date")["net_inr"]
    out = []
    for name, g in rep.groupby("strategy", sort=False):
        s = g.set_index("date")["net_inr"]
        gain = s - base.reindex(s.index)
        out.append({
            "Strategy": name, "Days": len(s), "Total net (Rs lakh)": s.sum() / 1e5,
            "Gain vs all G-DAM (Rs lakh)": gain.sum() / 1e5,
            "Day P10 (Rs lakh)": s.quantile(0.10) / 1e5, "Day P50 (Rs lakh)": s.quantile(0.50) / 1e5, "Day P90 (Rs lakh)": s.quantile(0.90) / 1e5,
            "Days better than all G-DAM": int((gain > 0).sum()), "Avg G-DAM share": g["da_share"].mean(),
            "Avg DSM (Rs lakh/day)": g["dsm_inr"].mean() / 1e5,
            "G-DAM leg (Rs lakh/day)": g["rev_g_inr"].mean() / 1e5, "RTM leg (Rs lakh/day)": g["rev_r_inr"].mean() / 1e5,
            "Charges (Rs lakh/day)": g["charges_inr"].mean() / 1e5, "Net (Rs lakh/day)": s.mean() / 1e5,
        })
    return pd.DataFrame(out).set_index("Strategy")
