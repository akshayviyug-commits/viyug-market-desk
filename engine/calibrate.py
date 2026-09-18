"""
Calibration mode - Market_Desk_Requirements.docx sections 6-8.

For each of the 96 blocks, backtests candidate DA shares against the
settled history and recommends three shares (Conservative / Balanced /
Aggressive), plus the reference numbers the daily-planning mode and the
Console consume.

-- "Forecast" for backtesting --
Section 7.1's formulas (DA_MW = s x Forecast, RTM_MW = Forecast - DA_MW)
need, for each *historical* block-day, the total volume that was actually
committed that day. That is `final_schedule` (DAC+DAM+GDAM+RTM), not the
early `day_ahead_schedule` figure - the DSM engine (dsm.py) was verified
against the sheet's own DSM columns using final_schedule as the scheduled
quantity, confirming deviation was historically measured against the final
total. Re-splitting that same total between DA/RTM keeps total deviation
risk identical to what actually happened (section 6's core point), which
is exactly the backtest we want.

-- Avoiding the section-6 "trap" --
Section 6 warns that a naive optimiser maximising (or minimising-worst-case)
average revenue drives every block to a 0%/100% split, because DSM cost is
provably independent of the split - whichever venue had the marginally
better price *every* day dominates on both the mean AND the worst day. On
this data that is not hypothetical: several blocks have one venue pricing
higher on 25+ of 28 days. A share search unconstrained by any prior belief
about next-day prices reproduces exactly the degenerate, in-sample-overfit
answer the spec calls "arithmetically correct and operationally useless".

The fix used here: each scenario searches a *bounded window* of candidate
shares around the desk's own historical commit ratio for that block
(`baseline_da_share`), rather than the unrestricted [0,1] grid:
  - Conservative : baseline +/- CONSERVATIVE_WINDOW  (stays close to current practice)
  - Balanced     : baseline +/- BALANCED_WINDOW      (moderate tilt toward the signal)
  - Aggressive   : the full [0,1] grid                (the pure backtest optimum)
Within its window, each scenario picks the share maximising average NetRev
over the settled days (both average and worst-day NetRev at that share are
reported, per docx 8.1). This mirrors CIP-023 in the platform requirements
doc ("Conservative reproduces the desk's commit ratios per cluster") and
matches the smoothly-varying, non-degenerate share curves shown in the
reference UI mockup.

All window widths and the share-grid step are configuration (docx sec. 9),
never hardcoded assumptions about the data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dsm import dsm_charge, ENERGY_FACTOR

SHARE_STEP = 0.01
CONSERVATIVE_WINDOW = 0.05
BALANCED_WINDOW = 0.20


def _share_grid(step: float) -> np.ndarray:
    n = int(round(1.0 / step))
    return np.round(np.linspace(0, 1, n + 1), 6)


def _net_revenue_curve(rows: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Returns a (len(grid), n_days) matrix of NetRev for this block."""
    forecast = rows["final_schedule"].to_numpy()
    actual = rows["actual_ll"].to_numpy()
    avc = rows["avc"].to_numpy()
    acp = rows["acp"].to_numpy()
    gdam_mcp = rows["gdam_mcp"].to_numpy()
    rtm_mcp = rows["rtm_mcp"].to_numpy()
    charges = rows["total_charges"].to_numpy()

    # DSM is split-invariant (section 6): computed once per block, not per candidate.
    dsm_total = dsm_charge(actual, forecast, avc, acp)["total_dsm"]

    da_mw = grid[:, None] * forecast[None, :]          # (n_candidates, n_days)
    rtm_mw = forecast[None, :] - da_mw
    revenue = (da_mw * gdam_mcp[None, :] + rtm_mw * rtm_mcp[None, :]) * ENERGY_FACTOR
    net = revenue - dsm_total[None, :] - charges[None, :]
    return net


def _pick_in_window(grid: np.ndarray, net: np.ndarray, center: float, window: float) -> dict:
    mask = np.abs(grid - center) <= window + 1e-9
    if not mask.any():
        mask = np.ones_like(grid, dtype=bool)
    sub_grid = grid[mask]
    sub_avg = net[mask].mean(axis=1)
    best_idx = int(np.argmax(sub_avg))
    share = float(sub_grid[best_idx])
    avg = float(sub_avg[best_idx])
    worst = float(net[mask][best_idx].min())
    return {"share": share, "avg_net_revenue": avg, "worst_day_net_revenue": worst}


def historical_baseline_share(rows: pd.DataFrame) -> float:
    da_committed = rows["dac_sched"] + rows["dam_sched"] + rows["gdam_sched"]
    ratio = da_committed / rows["final_schedule"]
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    return float(ratio.mean()) if len(ratio) else float("nan")


def calibrate_block(rows: pd.DataFrame, share_step: float = SHARE_STEP,
                     conservative_window: float = CONSERVATIVE_WINDOW,
                     balanced_window: float = BALANCED_WINDOW) -> dict:
    grid = _share_grid(share_step)
    net = _net_revenue_curve(rows, grid)
    baseline = historical_baseline_share(rows)
    center = baseline if np.isfinite(baseline) else 0.5

    conservative = _pick_in_window(grid, net, center, conservative_window)
    balanced = _pick_in_window(grid, net, center, balanced_window)
    aggressive = _pick_in_window(grid, net, center, window=1.01)  # unrestricted

    return {
        "baseline_da_share": None if not np.isfinite(baseline) else round(baseline, 4),
        "conservative": conservative,
        "balanced": balanced,
        "aggressive": aggressive,
    }


def _time_label(t) -> str:
    if hasattr(t, "strftime"):
        return t.strftime("%H:%M")
    return str(t)[:5]


def run_calibration(df: pd.DataFrame, settled_dates: list, **kwargs) -> pd.DataFrame:
    settled = df[df["date"].isin(settled_dates)]
    rows_out = []
    for block in range(1, 97):
        block_rows = settled[settled["block"] == block]
        if block_rows.empty:
            raise ValueError(f"Block {block} has no settled rows - cannot calibrate.")
        result = calibrate_block(block_rows, **kwargs)

        actuals = block_rows["actual_ll"]
        avg_actual = float(actuals.mean())
        cov = float(actuals.std(ddof=0) / avg_actual) if avg_actual else None

        rows_out.append({
            "block": block,
            "time": _time_label(block_rows["time"].iloc[0]),
            "da_share_conservative": result["conservative"]["share"],
            "da_share_balanced": result["balanced"]["share"],
            "da_share_aggressive": result["aggressive"]["share"],
            "avg_net_revenue_conservative": round(result["conservative"]["avg_net_revenue"]),
            "worst_day_net_revenue_conservative": round(result["conservative"]["worst_day_net_revenue"]),
            "avg_net_revenue_balanced": round(result["balanced"]["avg_net_revenue"]),
            "worst_day_net_revenue_balanced": round(result["balanced"]["worst_day_net_revenue"]),
            "avg_net_revenue_aggressive": round(result["aggressive"]["avg_net_revenue"]),
            "worst_day_net_revenue_aggressive": round(result["aggressive"]["worst_day_net_revenue"]),
            "baseline_da_share": result["baseline_da_share"],
            "avg_actual_generation_mw": round(avg_actual, 2),
            "coefficient_of_variation": round(cov, 4) if cov is not None else None,
        })
    return pd.DataFrame(rows_out)
