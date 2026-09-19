"""
Walk-forward test of the price model.

For every held-out day T the model is trained only on days whose prices the desk
already knows when it decides for T (G-DAM/DAM to T-1, RTM to T-2), then asked for
T's 96 blocks. Nothing from T itself is visible. Results are scored against the
naive predictors that a desk could use with no model at all, and against a perfect
foresight ceiling, on the same blocks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import ENERGY_FACTOR, WINDOW_LABELS, Panel, Policy, build_training_frame
from .price_model import ModelConfig, PriceModel, select_training

REC_DEFAULT = 300.0     # Rs/MWh REC uplift the sheet adds on RTM (not on G-DAM)


def run(panel: Panel, start_index: int = 7, policy: Policy = Policy(),
        cfg: ModelConfig = ModelConfig(), *, start_date=None, end_date=None,
        features: list[str] | None = None, extra_fn=None) -> pd.DataFrame:
    """Long frame: one row per held-out (date, block) with the model forecast, the naive
    forecasts, the realised prices and the wind forecast volume. Held-out days are those from
    `start_date` (or the `start_index`-th day) to `end_date` that have both prices."""
    frame = build_training_frame(panel, policy, extra_fn=extra_fn)
    out = []
    for i, T in enumerate(panel.dates):
        if (start_date is not None and T < pd.Timestamp(start_date)) or (start_date is None and i < start_index):
            continue
        if end_date is not None and T > pd.Timestamp(end_date):
            continue
        x = frame[frame["date"] == T]
        if x["y_gdam"].isna().all() or x["y_rtm"].isna().all():
            continue                                  # not settled: nothing to score against
        model = PriceModel(cfg, features).fit(select_training(frame, T, policy))
        p = model.predict(x)
        r = x[["date", "block", "window", "fc", "y_gdam", "y_rtm", "g_m3", "g_all", "r_m3", "r_all"]].copy()
        r["g_m3f"] = r["g_m3"].fillna(r["g_all"])
        r["r_m3f"] = r["r_m3"].fillna(r["r_all"])
        r["g_allf"], r["r_allf"] = r["g_all"], r["r_all"]
        r = pd.concat([r, p], axis=1)
        r["train_days_g"], r["train_days_r"] = model.n_days["g"], model.n_days["r"]
        r["fitted"] = bool(model.fitted["g"] and model.fitted["r"])
        out.append(r)
    return pd.concat(out, ignore_index=True)


METHODS = {
    "Model (P50)": ("g_p50", "r_p50"),
    "Persistence": ("g_anchor", "r_anchor"),
    "3-day mean": ("g_m3f", "r_m3f"),
    "All-history mean": ("g_allf", "r_allf"),
}


def score(pred: pd.DataFrame, rec: float = REC_DEFAULT, dates=None) -> dict:
    P = pred.dropna(subset=["y_gdam", "y_rtm", "g_anchor", "r_anchor", "g_allf", "r_allf"])
    if dates is not None:
        P = P[P["date"].isin(dates)]
    fc = np.nan_to_num(P["fc"].to_numpy(dtype=float))      # no plant forecast for the day -> no volume weight
    act_sp = P["y_rtm"].to_numpy() + rec - P["y_gdam"].to_numpy()
    price_g = P["y_gdam"].to_numpy()

    rows = []
    for name, (gc, rc) in METHODS.items():
        pg, pr = P[gc].to_numpy(), P[rc].to_numpy()
        sp = pr + rec - pg
        go_rtm = sp > 0
        chosen_minus_g = np.where(go_rtm, act_sp, 0.0)          # Rs/MWh gained vs selling everything G-DAM
        rows.append({
            "Method": name,
            "G-DAM MAE (Rs/MWh)": np.abs(pg - price_g).mean(),
            "RTM MAE (Rs/MWh)": np.abs(pr - P["y_rtm"].to_numpy()).mean(),
            "Spread direction correct": float((go_rtm == (act_sp > 0)).mean()),
            "Blocks sent to RTM": float(go_rtm.mean()),
            "Gain vs all-G-DAM (Rs lakh)": float((fc * ENERGY_FACTOR * chosen_minus_g).sum() / 1e5),
        })
    rows.append({"Method": "Always G-DAM", "Spread direction correct": float((act_sp <= 0).mean()),
                 "Blocks sent to RTM": 0.0, "Gain vs all-G-DAM (Rs lakh)": 0.0})
    rows.append({"Method": "Always RTM", "Spread direction correct": float((act_sp > 0).mean()),
                 "Blocks sent to RTM": 1.0, "Gain vs all-G-DAM (Rs lakh)": float((fc * ENERGY_FACTOR * act_sp).sum() / 1e5)})
    rows.append({"Method": "Perfect foresight", "Spread direction correct": 1.0,
                 "Blocks sent to RTM": float((act_sp > 0).mean()),
                 "Gain vs all-G-DAM (Rs lakh)": float((fc * ENERGY_FACTOR * np.maximum(act_sp, 0)).sum() / 1e5)})
    table = pd.DataFrame(rows).set_index("Method")

    cover = {m: float(((P[f"{m}_p10"] <= P[y]) & (P[y] <= P[f"{m}_p90"])).mean())
             for m, y in (("g", "y_gdam"), ("r", "y_rtm"))}

    win = P.assign(g_err_model=(P["g_p50"] - P["y_gdam"]).abs(), g_err_pers=(P["g_anchor"] - P["y_gdam"]).abs(),
                   r_err_model=(P["r_p50"] - P["y_rtm"]).abs(), r_err_pers=(P["r_anchor"] - P["y_rtm"]).abs())
    by_window = win.groupby("window")[["g_err_model", "g_err_pers", "r_err_model", "r_err_pers"]].mean()
    by_window.index = [WINDOW_LABELS[int(i)] for i in by_window.index]
    by_window.columns = ["G-DAM model", "G-DAM persistence", "RTM model", "RTM persistence"]

    by_day = win.groupby("date").agg(
        g_model=("g_err_model", "mean"), g_pers=("g_err_pers", "mean"),
        r_model=("r_err_model", "mean"), r_pers=("r_err_pers", "mean"),
    )
    return {"table": table, "coverage_p10_p90": cover, "by_window": by_window, "by_day": by_day,
            "n_days": int(P["date"].nunique()), "n_blocks": int(len(P)),
            "base_rate_g_better": float((act_sp <= 0).mean()), "rec": rec,
            "both_at_cap_share": float(((P["y_gdam"] >= 9999) & (P["y_rtm"] >= 9999)).mean()),
            "any_at_cap_share": float(((P["y_gdam"] >= 9999) | (P["y_rtm"] >= 9999)).mean())}
