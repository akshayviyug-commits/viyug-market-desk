"""
G-DAM vs RTM split, block by block, driven by the price forecast.

For each of the 96 blocks:

  1. price view      z = ( E[RTM] + REC - E[G-DAM] ) / spread-uncertainty
                     z > 0: RTM is expected to pay more (after the Rs300 REC RTM earns and G-DAM does not);
                     the uncertainty comes from the calibrated P10-P90 bands of both markets, treated as
                     independent, which overstates it a little and so errs towards caution.
  2. share           day-ahead share = desk's usual commit ratio  -  tilt x clip(z, -2, 2)
                     The dial sets the tilt: how many points of share one unit of price confidence moves.
  3. guardrail       day-ahead MW can never exceed the forecast-error-based volume for that dial
                     (engine.predict_split.day_ahead_volume). Where the wind forecast is unreliable, day-ahead
                     is held back whatever the price says.
  4. thumb rule      RTM = forecast - day-ahead, exactly, on every block, checked in code.

Day-ahead volume is all G-DAM (DAM and DAC are outside this scope). DSM is identical whatever the split
(both legs are sized off the same forecast), so it is a cost in the scoring and an input to the guardrail,
not a lever here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bundle import DataBundle
from .dsm import ENERGY_FACTOR
from .features import block_time, block_window
from .learn_predict import profile_for
from .predict_split import DIALS, day_ahead_volume

REC_DEFAULT = 300.0
DIAL_TILT = {"conservative": 0.10, "balanced": 0.20, "aggressive": 0.30}   # share points per unit of z
Z_CLIP = 2.0
BAND_Z = 2 * 1.2816            # a P10-P90 band spans 2.563 standard deviations
SIGMA_FLOOR = 100.0            # Rs/MWh, stops a zero-width band from producing infinite confidence
DIAL_LABEL = {k: v.label for k, v in DIALS.items()}


def price_split(bundle: DataBundle, T, prices: pd.DataFrame, dial: str, rec: float = REC_DEFAULT,
                overrides: dict | None = None, guardrail: bool = True, tilt: float | None = None) -> dict:
    """Recommended 96-block G-DAM / RTM plan for delivery date T. `prices` is the (calibrated) forecast
    frame from engine.price_forecast.Forecaster.forecast(T). `overrides` maps block -> day-ahead share.
    `guardrail=False` removes the forecast-error ceiling (day-ahead is then limited only by capacity);
    `tilt` overrides the dial's strength (0 = the desk's habit with no price view)."""
    if dial not in DIAL_TILT:
        raise ValueError(f"Unknown dial '{dial}'. Must be one of {list(DIAL_TILT)}.")
    T = pd.Timestamp(T)
    overrides = {int(b): float(np.clip(s, 0.0, 1.0)) for b, s in (overrides or {}).items()}
    profile, target, warnings = profile_for(bundle.days, T.strftime("%Y-%m-%d"))
    fc = np.array(target.da_sch, dtype=float)
    avc = np.array(target.avc, dtype=float)
    blocks = np.array(target.blk)
    p = prices.set_index("block").loc[blocks]

    base = np.array([profile.commit.ratio[c] for c in target.cl], dtype=float)
    ceiling = np.array(day_ahead_volume(target, profile, DIALS[dial]), dtype=float) if guardrail else avc.copy()
    tilt_k = DIAL_TILT[dial] if tilt is None else float(tilt)

    mu_g = p["g_p50"].to_numpy()
    mu_r = p["r_p50"].to_numpy() + rec
    sg = (p["g_p90"] - p["g_p10"]).to_numpy() / BAND_Z
    sr = (p["r_p90"] - p["r_p10"]).to_numpy() / BAND_Z
    sigma = np.maximum(np.sqrt(sg ** 2 + sr ** 2), SIGMA_FLOOR)
    spread = mu_r - mu_g
    z = spread / sigma
    tilt_pts = tilt_k * np.clip(z, -Z_CLIP, Z_CLIP)

    da_share_price = np.clip(base - tilt_pts, 0.0, 1.0)
    da_mw = np.minimum(np.minimum(da_share_price * fc, ceiling), avc)
    da_mw = np.maximum(da_mw, 0.0)
    capped = da_mw < da_share_price * fc - 1e-9

    for b, s in overrides.items():
        i = int(np.where(blocks == b)[0][0])
        da_mw[i] = min(max(s * fc[i], 0.0), avc[i], fc[i])
    rtm_mw = fc - da_mw
    # the thumb rule, checked on what is actually shown - fail the step, never file a broken split
    assert np.abs(da_mw + rtm_mw - fc).max() < 1e-9, "Thumb-rule identity violated."
    assert (da_mw >= -1e-12).all() and (rtm_mw >= -1e-9).all(), "Negative volume produced."

    share = np.divide(da_mw, fc, out=np.zeros_like(fc), where=fc > 0)
    reason = np.select(
        [np.isin(blocks, list(overrides)), capped, z <= -0.5, z >= 0.5],
        ["Override", "Held back by forecast-error guardrail", "Price favours G-DAM", "Price favours RTM"],
        default="No clear price edge - desk habit")
    plan = pd.DataFrame({
        "block": blocks, "time": [block_time(b) for b in blocks], "window": [block_window(b) for b in blocks],
        "cluster": list(target.cl),
        "forecast_mw": fc, "avc_mw": avc,
        "g_p10": p["g_p10"].to_numpy(), "g_p50": mu_g, "g_p90": p["g_p90"].to_numpy(),
        "r_p10": p["r_p10"].to_numpy(), "r_p50": p["r_p50"].to_numpy(), "r_p90": p["r_p90"].to_numpy(),
        "rtm_edge": spread, "confidence_z": z,
        "habit_da_share": base, "price_da_share": da_share_price, "ceiling_da_share": np.divide(ceiling, fc, out=np.zeros_like(fc), where=fc > 0),
        "da_share_applied": share, "gdam_mw": da_mw, "dam_mw": np.zeros_like(fc), "rtm_mw": rtm_mw,
        "reason": reason, "override_flag": np.isin(blocks, list(overrides)),
    })

    total = fc.sum() * ENERGY_FACTOR
    da_mwh = da_mw.sum() * ENERGY_FACTOR
    exp_rev = float(((da_mw * mu_g + rtm_mw * mu_r) * ENERGY_FACTOR).sum())
    exp_all_g = float((fc * mu_g * ENERGY_FACTOR).sum())
    summary = {
        "total_forecast_mwh": round(total, 2), "gdam_mwh": round(da_mwh, 2), "rtm_mwh": round(total - da_mwh, 2),
        "gdam_share": float(da_mwh / total) if total else 0.0, "rtm_share": float(1 - da_mwh / total) if total else 0.0,
        "override_count": len(overrides), "guardrail": bool(guardrail), "blocks_price_rtm": int((reason == "Price favours RTM").sum()),
        "blocks_price_gdam": int((reason == "Price favours G-DAM").sum()), "blocks_held_back": int((reason == "Held back by forecast-error guardrail").sum()),
        "indicative_revenue_inr": round(exp_rev), "indicative_gain_vs_all_gdam_inr": round(exp_rev - exp_all_g),
        "indicative_note": "Indicative only: the plan valued at the model's own median price forecast (RTM includes the REC). "
                           "It is not a forecast of tomorrow's earnings.",
    }
    return {"plan": plan, "summary": summary, "warnings": warnings, "profile": profile, "rec": rec, "dial": dial}


CLUSTER_ORDER = ["C-1", "C-2", "C-3", "C-4"]
CLUSTER_SLOT = {"C-1": "06-10am", "C-2": "10am-6pm", "C-3": "6-10pm", "C-4": "10pm-6am"}   # the desk's four slots
CLUSTER_NAME = {c: c.lower() + " \u00b7 " + CLUSTER_SLOT[c] for c in CLUSTER_ORDER}


def cluster_table(plan: pd.DataFrame) -> pd.DataFrame:
    """The desk's four time-of-day slots (c-1 06-10am, c-2 10am-6pm, c-3 6-10pm, c-4 10pm-6am)."""
    g = plan.assign(fc_mwh=plan["forecast_mw"] * ENERGY_FACTOR, g_mwh=plan["gdam_mw"] * ENERGY_FACTOR,
                    r_mwh=plan["rtm_mw"] * ENERGY_FACTOR).groupby("cluster")[["fc_mwh", "g_mwh", "r_mwh"]].sum()
    g = g.reindex(CLUSTER_ORDER)
    return pd.DataFrame({
        "cluster": CLUSTER_ORDER, "Time of day": [CLUSTER_NAME[c] for c in CLUSTER_ORDER],
        "Forecast (MWh)": g["fc_mwh"].to_numpy(), "G-DAM (MWh)": g["g_mwh"].to_numpy(), "RTM (MWh)": g["r_mwh"].to_numpy(),
        "G-DAM %": (100 * g["g_mwh"] / g["fc_mwh"]).to_numpy(), "RTM %": (100 * g["r_mwh"] / g["fc_mwh"]).to_numpy(),
    })
