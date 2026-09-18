"""
Generates a synthetic sample workbook with the same shape and headers as the
real settlement sheet, but entirely fabricated numbers - safe to commit to a
public repo / deploy to Streamlit Community Cloud. Not derived from the real
client workbook in any way (no scaling, no perturbation of real values).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=7)

AVC = 100.0
N_SETTLED_DAYS = 21
BLOCKS = np.arange(1, 97)


def diurnal_wind_profile(blocks, day_seed):
    """A plausible-looking wind generation curve: higher overnight/evening,
    a midday dip, block-to-block noise, all synthetic."""
    hour = (blocks - 1) * 0.25
    base = 55 + 25 * np.sin((hour - 3) / 24 * 2 * np.pi) + 15 * np.cos((hour - 19) / 24 * 2 * np.pi)
    day_rng = np.random.default_rng(day_seed)
    noise = day_rng.normal(0, 6, size=len(blocks))
    profile = np.clip(base + noise, 5, AVC)
    return profile


def price_series(blocks, day_seed, base_level):
    day_rng = np.random.default_rng(day_seed + 1000)
    hour = (blocks - 1) * 0.25
    evening_premium = 1500 * np.clip(np.sin((hour - 15) / 12 * np.pi), 0, 1)
    noise = day_rng.normal(0, 400, size=len(blocks))
    return np.clip(base_level + evening_premium + noise, 1500, 10000)


rows = []
for day_idx in range(N_SETTLED_DAYS + 1):  # last day is forecast-only
    date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=day_idx)
    is_forecast_only = day_idx == N_SETTLED_DAYS
    seed = 42 + day_idx

    day_ahead_schedule = diurnal_wind_profile(BLOCKS, seed)
    final_schedule = np.clip(day_ahead_schedule * (1 + np.random.default_rng(seed + 2000).normal(0, 0.03, 96)), 0, AVC)

    final_schedule = np.round(final_schedule, 2)  # round once, before deriving legs, so the identity is exact
    baseline_share = np.clip(0.78 + 0.05 * np.sin((BLOCKS - 1) / 96 * 2 * np.pi)
                              + np.random.default_rng(seed + 3000).normal(0, 0.02, 96), 0.5, 0.98)
    gdam_sched = np.round(final_schedule * baseline_share, 2)
    rtm_sched = final_schedule - gdam_sched  # exact remainder, no independent rounding drift
    dac_sched = np.zeros(96)
    dam_sched = np.zeros(96)

    gdam_mcp = price_series(BLOCKS, seed, base_level=5500)
    rtm_mcp = price_series(BLOCKS, seed + 500, base_level=4800)
    dam_mcp = price_series(BLOCKS, seed + 700, base_level=5300)
    dac_mcp = np.where(np.random.default_rng(seed + 900).random(96) > 0.4,
                        price_series(BLOCKS, seed + 900, base_level=6000), np.nan)
    acp = (gdam_mcp + rtm_mcp + dam_mcp) / 3

    if is_forecast_only:
        actual_ll = np.zeros(96)
        gdam_mcp = dam_mcp = rtm_mcp = dac_mcp = acp = np.full(96, np.nan)
        total_charges = np.full(96, np.nan)
    else:
        dev_noise = np.random.default_rng(seed + 4000).normal(0, 0.08 * AVC, 96)
        actual_ll = np.clip(final_schedule + dev_noise, 0, AVC * 1.05)
        total_charges = 45 * final_schedule * 0.25 + np.random.default_rng(seed + 5000).normal(0, 50, 96)

    for i, block in enumerate(BLOCKS):
        total_minutes = (block - 1) * 15
        time_val = pd.Timestamp("1899-12-30") + pd.Timedelta(minutes=total_minutes)
        rows.append({
            "Time": time_val.time(),
            "Block": int(block),
            "Date": date,
            "Day Ahead Schedule": round(float(day_ahead_schedule[i]), 2),
            "DAC Schedule (Interface Point)": round(float(dac_sched[i]), 2),
            "DAM Schedule (Interface Point)": round(float(dam_sched[i]), 2),
            "GDAM Schedule (Interface Point)": round(float(gdam_sched[i]), 2),
            "RTM Schedule (Interface Point)": round(float(rtm_sched[i]), 2),
            "Final Schedule Power (MW)": round(float(final_schedule[i]), 2),
            "AvC(MW)": AVC,
            "Actual Injected Power after line loss(MW)": round(float(actual_ll[i]), 2),
            "ACP (wt. avg of MCP's)": None if is_forecast_only else round(float(acp[i]), 2),
            "DAC MCP": None if (is_forecast_only or np.isnan(dac_mcp[i])) else round(float(dac_mcp[i]), 2),
            "G-DAM MCP": None if is_forecast_only else round(float(gdam_mcp[i]), 2),
            "DAM MCP": None if is_forecast_only else round(float(dam_mcp[i]), 2),
            "RTM MCP": None if is_forecast_only else round(float(rtm_mcp[i]), 2),
            "Total charges": None if is_forecast_only else round(float(total_charges[i]), 2),
        })

df = pd.DataFrame(rows)

out_path = "sample_data/sample_workbook.xlsx"
import os
os.makedirs("sample_data", exist_ok=True)

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    # header lands on Excel row 4 (startrow=3, 0-indexed), matching the real sheet's layout
    df.to_excel(writer, sheet_name="5. Detailed sheet", index=False, header=True, startrow=3)

print(f"Wrote {out_path}: {len(df)} rows, {N_SETTLED_DAYS} settled days + 1 forecast-only day.")
