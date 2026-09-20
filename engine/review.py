"""
Numbers behind the Learn page: what happened on recent settled days, and what the price history says.

Everything is computed from the uploaded workbook; nothing is typed in. Settlement rules match
engine.dsm (bands as % of AvC, ACP-based rates) and the sheet's REC convention (Rs300/MWh on DAC, DAM and
RTM, none on G-DAM).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bundle import DataBundle
from .dsm import ENERGY_FACTOR, dsm_charge
from .features import CAP, Panel, holiday_name, is_holiday
from .learn_engine import cluster_of

EVENING = slice(72, 88)          # blocks 73-88, 18:00-22:00
CLUSTERS = ["C-1", "C-2", "C-3", "C-4"]
CLUSTER_SLOT = {"C-1": "06-10am", "C-2": "10am-6pm", "C-3": "6-10pm", "C-4": "10pm-6am"}
BLOCK_CLUSTER = np.array([cluster_of(b) for b in range(1, 97)])


def _label(d: pd.Timestamp, delivery: pd.Timestamp) -> str:
    k = (delivery - d).days
    name = "Yesterday" if k == 1 else f"D-{k}"
    return f"{name} · {d.day} {d:%b}"


def review_days(bundle: DataBundle, delivery, n: int = 5, rec: float = 300.0) -> dict:
    """The last `n` settled days before `delivery`, as they actually settled."""
    delivery = pd.Timestamp(delivery)
    df = bundle.workbook.df
    days = [pd.Timestamp(d) for d in bundle.workbook.settled_dates if pd.Timestamp(d) <= delivery - pd.Timedelta(days=1)][-n:]
    rows, eve_err, rest_err = [], [], []
    tot = {"da_val": 0.0, "da_vol": 0.0, "r_val": 0.0, "r_vol": 0.0}
    prom_mat, act_mat, avc_mat = [], [], []
    for d in days:
        g = df[df["date"] == d].sort_values("block")
        act, fin, avc, acp = (g[c].to_numpy(float) for c in ("actual_ll", "final_schedule", "avc", "acp"))
        dac, dam, gd, rtm = (g[c].fillna(0).to_numpy(float) for c in ("dac_sched", "dam_sched", "gdam_sched", "rtm_sched"))
        dac_px = g["dac_mcp"].fillna(g["gdam_mcp"]).to_numpy(float)
        da_val = float((dac * (dac_px + rec) + dam * (g["dam_mcp"].to_numpy(float) + rec) + gd * g["gdam_mcp"].to_numpy(float)).sum())
        da_vol = float((dac + dam + gd).sum())
        r_val = float((rtm * (g["rtm_mcp"].to_numpy(float) + rec)).sum())
        r_vol = float(rtm.sum())
        dsm = float(np.sum(dsm_charge(act, fin, avc, acp)["total_dsm"]))
        dev = act - fin
        eve_short = bool(dev[EVENING].mean() < 0)
        absdev = np.abs(dev) / avc * 100
        rest = np.ones(96, bool)
        rest[EVENING] = False
        eve_err.append(absdev[EVENING].mean())
        rest_err.append(absdev[rest].mean())
        for k, v in (("da_val", da_val), ("da_vol", da_vol), ("r_val", r_val), ("r_vol", r_vol)):
            tot[k] += v
        actual_mwh, promised_mwh = act.sum() * ENERGY_FACTOR, fin.sum() * ENERGY_FACTOR
        rows.append({
            "date": d, "label": _label(d, delivery), "actual_mwh": actual_mwh, "promised_mwh": promised_mwh,
            "gap_mwh": actual_mwh - promised_mwh,
            "da_price": da_val / da_vol / 1000 if da_vol > 0 else np.nan,       # Rs per unit (kWh)
            "rtm_price": r_val / r_vol / 1000 if r_vol > 0 else np.nan,
            "dev_cost_inr": dsm, "evening_short": eve_short,
        })
        prom_mat.append(fin)
        act_mat.append(act)
        avc_mat.append(avc)
    table = pd.DataFrame(rows)
    stats = {}
    if len(table):
        da_avg = tot["da_val"] / tot["da_vol"] / 1000 if tot["da_vol"] > 0 else np.nan
        r_avg = tot["r_val"] / tot["r_vol"] / 1000 if tot["r_vol"] > 0 else np.nan
        stats = {
            "n_days": len(table), "eve_short_days": int(table["evening_short"].sum()),
            "eve_err_pct": float(np.mean(eve_err)), "rest_err_pct": float(np.mean(rest_err)),
            "rtm_premium": float(r_avg - da_avg) if np.isfinite(da_avg) and np.isfinite(r_avg) else np.nan,
            "first": table["date"].iloc[0], "last": table["date"].iloc[-1],
        }
    blocks = {}
    if len(table):
        promised = np.mean(prom_mat, axis=0)
        delivered = np.mean(act_mat, axis=0)
        avc = np.mean(avc_mat, axis=0)
        gap = np.abs(delivered - promised)
        by_cluster = {c: float(gap[BLOCK_CLUSTER == c].mean()) for c in CLUSTERS}
        blocks = {"promised": promised, "delivered": delivered, "avc": avc, "band": 0.10 * avc,
                  "widest_cluster": max(by_cluster, key=by_cluster.get), "gap_by_cluster": by_cluster}
    return {"table": table, "stats": stats, "blocks": blocks}


# ------------------------------------------------------------------ price history panels
def price_matrix(panel: Panel, market: str, last_n: int = 30):
    """(dates, 96-col matrix) for the last `last_n` days that have any price for that market."""
    arr = {"gdam": panel.gdam, "dam": panel.dam, "rtm": panel.rtm}[market]
    have = np.isfinite(arr).any(axis=1)
    idx = np.where(have)[0][-last_n:]
    return panel.dates[idx], arr[idx]


def _weekday_block(arr: np.ndarray, dates) -> dict:
    have = np.isfinite(arr).all(axis=1)
    if not have.any():
        return {"table": pd.DataFrame({"day": WEEKDAYS, "mean": np.nan, "n": 0}), "n_days": 0, "weekday_mean": float("nan"), "holidays": []}
    lvl = pd.Series(np.nanmean(arr[have], axis=1), index=dates[have])
    grp = lvl.groupby(lvl.index.dayofweek)
    table = pd.DataFrame({"day": WEEKDAYS, "mean": [grp.mean().get(i, np.nan) for i in range(7)], "n": [int(grp.size().get(i, 0)) for i in range(7)]})
    hol = [{"date": d, "name": holiday_name(d), "price": float(lvl[d])} for d in lvl.index if is_holiday(d) and d.dayofweek < 5]
    return {"table": table, "n_days": int(have.sum()), "weekday_mean": float(lvl[lvl.index.dayofweek < 5].mean()), "holidays": hol}


WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def weekday_effect(panel: Panel) -> dict:
    """Mean daily price by weekday for the two markets we forecast, with the day count behind each."""
    return {"gdam": _weekday_block(panel.gdam, panel.dates), "rtm": _weekday_block(panel.rtm, panel.dates)}


def default_high_threshold(panel: Panel, market: str = "gdam") -> float:
    """What counts as a high price is set by the data: the market's average price plus half a standard deviation in this
    workbook, to the nearest Rs100 (a level well above normal, without being pinned to the price cap)."""
    arr = {"gdam": panel.gdam, "dam": panel.dam, "rtm": panel.rtm}[market]
    v = arr[np.isfinite(arr)]
    if not v.size:
        return CAP
    return float(min(CAP, round(float(v.mean() + 0.5 * v.std()), -2)))


def high_runs(panel: Panel, market: str = "gdam", threshold: float | None = None) -> dict:
    """How long the price stays at or above `threshold` once it gets there (runs of consecutive blocks)."""
    thr = default_high_threshold(panel, market) if threshold is None else float(threshold)
    arr = {"gdam": panel.gdam, "dam": panel.dam, "rtm": panel.rtm}[market]
    have = np.isfinite(arr).all(axis=1)
    runs = []
    for row in arr[have]:
        high = row >= thr
        i = 0
        while i < 96:
            if high[i]:
                j = i
                while j < 96 and high[j]:
                    j += 1
                runs.append(j - i)
                i = j
            else:
                i += 1
    share = float((arr[have] >= thr).mean()) if have.any() else float("nan")
    return {"runs": runs, "share": share, "threshold": thr, "n_days": int(have.sum()),
            "median": float(np.median(runs)) if runs else 0.0, "longest": int(max(runs)) if runs else 0}


def spread_by_cluster(panel: Panel, rec: float = 300.0) -> pd.DataFrame:
    """Where RTM (plus REC) beat G-DAM, by the desk's four slots, on days both prices exist."""
    have = np.isfinite(panel.gdam).all(axis=1) & np.isfinite(panel.rtm).all(axis=1)
    g, r = panel.gdam[have], panel.rtm[have]
    edge = r + rec - g
    rows = []
    for c in CLUSTERS:
        m = BLOCK_CLUSTER == c
        rows.append({"cluster": c, "slot": CLUSTER_SLOT[c], "mean_edge": float(edge[:, m].mean()),
                     "share_rtm_better": float((edge[:, m] > 0).mean()), "mean_g": float(g[:, m].mean()), "mean_r": float(r[:, m].mean())})
    out = pd.DataFrame(rows)
    out.attrs["n_days"] = int(have.sum())
    return out
