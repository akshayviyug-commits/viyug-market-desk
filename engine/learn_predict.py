"""
Adapter between the app's validated workbook frame and the Learn / Predict
engine (learn_engine.py + predict_split.py).

Those two modules are pure Python and know nothing about pandas or the
workbook. This module builds their `Day` inputs from the loader's DataFrame,
applies the forecast guardrails, and returns tidy DataFrames for the UI.

Walk-forward is enforced by the engine itself (`as_of_cutoff`): for a delivery
date D only days up to D-2 are ever read. The target day handed to Learn and
Predict is stripped down to what is knowable in advance - forecast and AvC -
so nothing from the delivery date's own settlement can reach the decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dsm import ENERGY_FACTOR
from .learn_engine import CLUSTER_LABEL, CLUSTERS, LINE_LOSS, Day, LearnProfile, build_learn_profile
from .planning import ForecastValidationError, validate_forecast
from .predict_split import DIALS, place_volumes, predict_tomorrow

LOOKBACK_K = 3   # days behind the desk-habit and venue-preference signals (spec default)


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def block_time(block: int) -> str:
    minutes = (int(block) - 1) * 15
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def build_days(df: pd.DataFrame, settled_dates: list) -> dict[str, Day]:
    """One `Day` per date in the workbook. Settled days carry actuals, prices and
    the desk's own legs; forecast-only days carry just forecast and AvC."""
    settled = {pd.Timestamp(d) for d in settled_dates}
    days: dict[str, Day] = {}
    for date, g in df.groupby("date"):
        g = g.sort_values("block")
        is_settled = pd.Timestamp(date) in settled
        days[_iso(date)] = Day(
            date=_iso(date),
            blk=g["block"].astype(int).tolist(),
            da_sch=g["day_ahead_schedule"].tolist(),
            avc=g["avc"].tolist(),
            # the engine applies the line-loss factor itself, so hand it the pre-loss figure
            act=(g["actual_ll"] / (1 - LINE_LOSS)).tolist() if is_settled else None,
            dac=g["dac_sched"].fillna(0).tolist(),
            dam=g["dam_sched"].fillna(0).tolist(),
            gdam=g["gdam_sched"].fillna(0).tolist(),
            mcp_gdam=g["gdam_mcp"].tolist() if is_settled else None,
            mcp_dam=g["dam_mcp"].tolist() if is_settled else None,
            mcp_rtm=g["rtm_mcp"].tolist() if is_settled else None,
            has_split=is_settled,
            has_prices=is_settled,
        )
    return days


def _target(days: dict[str, Day], delivery_iso: str) -> tuple[Day, list[str]]:
    """The delivery day as Learn/Predict may see it: forecast and AvC only."""
    if delivery_iso not in days:
        raise ForecastValidationError(f"No forecast on file for {delivery_iso}.")
    src = days[delivery_iso]
    forecast = pd.DataFrame({"block": src.blk, "forecast_mw": src.da_sch})
    if forecast["forecast_mw"].isna().any():
        missing = forecast.loc[forecast["forecast_mw"].isna(), "block"].tolist()
        raise ForecastValidationError(f"Forecast is missing for block(s) {missing} - it cannot be planned.")
    warnings = validate_forecast(forecast, dict(zip(src.blk, src.avc)))
    return Day(date=src.date, blk=list(src.blk), da_sch=list(src.da_sch), avc=list(src.avc)), warnings


def profile_for(days: dict[str, Day], delivery_iso: str, k: int = LOOKBACK_K) -> tuple[LearnProfile, Day, list[str]]:
    target, warnings = _target(days, delivery_iso)
    return build_learn_profile(days, target, k=k), target, warnings


def plan_day(days: dict[str, Day], delivery_iso: str, dial: str,
             overrides: dict | None = None, k: int = LOOKBACK_K) -> dict:
    """Recommended 96-block bid sheet for one delivery date and one dial.
    `overrides` maps block -> DA share (fraction of the forecast) and replaces the
    dial's own day-ahead commitment for that block; the thumb rule is re-applied."""
    if dial not in DIALS:
        raise ValueError(f"Unknown dial '{dial}'. Must be one of {list(DIALS)}.")
    overrides = {int(b): float(np.clip(s, 0.0, 1.0)) for b, s in (overrides or {}).items()}
    profile, target, warnings = profile_for(days, delivery_iso, k)

    sheet = predict_tomorrow(target, profile, dial)
    forecast = np.array(target.da_sch, dtype=float)
    base_da = np.array(sheet.gdam) + np.array(sheet.dam)
    base_share = np.divide(base_da, forecast, out=np.zeros_like(forecast), where=forecast > 0)

    if overrides:
        qda = [forecast[i] * overrides[b] if b in overrides else sheet.day_ahead_qty[i]
               for i, b in enumerate(target.blk)]
        sheet = place_volumes(target, qda, profile.segment.g_flag)
        sheet.dial = dial

    gdam, dam, rtm = np.array(sheet.gdam), np.array(sheet.dam), np.array(sheet.rtm)
    da = gdam + dam
    # the thumb rule, checked again on what is actually shown - fail the step, never file a broken split
    assert np.abs(da + rtm - forecast).max() < 1e-9, "Thumb-rule identity violated."
    assert (gdam >= 0).all() and (dam >= 0).all() and (rtm >= -1e-12).all(), "Negative volume produced."

    share = np.divide(da, forecast, out=np.zeros_like(forecast), where=forecast > 0)
    g_flag = np.array(profile.segment.g_flag)
    plan = pd.DataFrame({
        "block": target.blk,
        "time": [block_time(b) for b in target.blk],
        "cluster": target.cl,
        "forecast_mw": forecast,
        "avc_mw": target.avc,
        "gdam_mw": gdam,
        "dam_mw": dam,
        "da_mw": da,
        "rtm_mw": rtm,
        "da_share_applied": share,
        "base_da_share": base_share,
        "venue": np.where(g_flag == 1, "G-DAM", "DAM"),
        "override_flag": [b in overrides for b in target.blk],
    })

    total = forecast.sum() * ENERGY_FACTOR
    summary = {
        "total_forecast_mwh": round(total, 2),
        "total_da_mwh": round(da.sum() * ENERGY_FACTOR, 2),
        "total_gdam_mwh": round(gdam.sum() * ENERGY_FACTOR, 2),
        "total_dam_mwh": round(dam.sum() * ENERGY_FACTOR, 2),
        "total_rtm_mwh": round(rtm.sum() * ENERGY_FACTOR, 2),
        "overall_da_share": round(float(da.sum() * ENERGY_FACTOR / total), 4) if total else 0.0,
        "override_count": len(overrides),
    }

    # Indicative revenue: historical per-block average prices over every priced day the
    # profile may see, applied to this split. Not a forecast - tomorrow's prices are unknown.
    priced = [d for d in days.values() if d.date <= profile.cutoff and d.has_prices]
    if priced:
        def avg(attr):
            return np.nanmean(np.array([getattr(d, attr) for d in priced], dtype=float), axis=0)
        revenue = ((gdam * avg("mcp_gdam") + dam * avg("mcp_dam") + rtm * avg("mcp_rtm")) * ENERGY_FACTOR).sum()
        summary["indicative_revenue_inr"] = round(float(revenue))
        summary["indicative_revenue_note"] = (
            "Indicative only - historical per-block average prices, REC and fees excluded. "
            "Not a forecast of tomorrow's prices."
        )

    return {"plan": plan, "summary": summary, "warnings": warnings, "profile": profile,
            "provenance": provenance(profile)}


def provenance(profile: LearnProfile) -> dict:
    """Plain-language facts about where a profile came from."""
    return {
        "cutoff": profile.cutoff,
        "source": profile.error.source,
        "days_used": list(profile.error.days_used),
        "mirror_days": list(profile.commit.days_used),
        "venue_days": list(profile.segment.days_used),
        "fallback_error_clusters": list(profile.error.fallback_da),
        "commit_fallback": bool(profile.commit.used_fallback),
        "venue_fallback": bool(profile.segment.used_fallback),
    }


def learn_view(days: dict[str, Day], delivery_iso: str, k: int = LOOKBACK_K) -> dict:
    """Everything the Learn page shows for one delivery date."""
    profile, target, warnings = profile_for(days, delivery_iso, k)
    g_flag = np.array(profile.segment.g_flag)
    cl = np.array(target.cl)
    fallback = set(profile.error.fallback_da)

    rows = []
    for c in CLUSTERS:
        err = np.array(profile.error.err_da[c], dtype=float)
        in_c = cl == c
        rows.append({
            "Time window": CLUSTER_LABEL[c],
            "Blocks": int(in_c.sum()),
            "Desk day-ahead commit": profile.commit.ratio[c],
            "Forecast error P25 (MW)": float(np.quantile(err, 0.25)),
            "Forecast error P50 (MW)": float(np.quantile(err, 0.50)),
            "Forecast error P75 (MW)": float(np.quantile(err, 0.75)),
            "Samples": int(err.size),
            "Source": "built-in starter sample" if c in fallback else "real history",
            "G-DAM preferred": f"{int(g_flag[in_c].sum())} of {int(in_c.sum())} blocks",
        })
    return {
        "cluster_table": pd.DataFrame(rows),
        "error_by_cluster": {c: list(profile.error.err_da[c]) for c in CLUSTERS},
        "commit_ratio": dict(profile.commit.ratio),
        "blocks": list(target.blk),
        "times": [block_time(b) for b in target.blk],
        "spread": list(profile.segment.spread),
        "g_flag": g_flag.tolist(),
        "provenance": provenance(profile),
        "warnings": warnings,
    }
