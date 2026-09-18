import sys
sys.path.insert(0, ".")
from engine.loader import load_workbook
from engine.calibrate import run_calibration
from engine.planning import build_daily_plan

PATH = r"D:\Viyug\00. HATAGERI PH-1 (102.3 MW) (2).xlsx"
result = load_workbook(PATH)
calibration = run_calibration(result.df, result.settled_dates)

# Aug 29 is the held-out forecast-only day - use its day_ahead_schedule as
# "tomorrow's forecast" for a smoke test of planning mode end to end.
tomorrow = result.df[result.df["date"].isin(result.forecast_only_dates)].copy()
forecast = tomorrow[["block", "day_ahead_schedule"]].rename(columns={"day_ahead_schedule": "forecast_mw"})
avc_by_block = dict(zip(tomorrow["block"], tomorrow["avc"]))

for strategy in ["conservative", "balanced", "aggressive"]:
    out = build_daily_plan(forecast, calibration, strategy=strategy, avc_by_block=avc_by_block)
    print(f"--- {strategy} ---")
    print(out["summary"])
    print("warnings:", out["warnings"][:3], "..." if len(out["warnings"]) > 3 else "")
    print()

# Acceptance test: override on a couple of blocks
out = build_daily_plan(forecast, calibration, strategy="balanced", overrides={1: 0.5, 50: 0.9}, avc_by_block=avc_by_block)
print("Override test: override_count =", out["summary"]["override_count"])
print(out["plan"][out["plan"]["block"].isin([1, 50])])
