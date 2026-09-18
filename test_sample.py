import sys
sys.path.insert(0, ".")
from engine.loader import load_workbook
from engine.calibrate import run_calibration
from engine.planning import build_daily_plan

PATH = "sample_data/sample_workbook.xlsx"
result = load_workbook(PATH)
print("Loaded OK:", result.n_settled_days, "settled days,", result.date_range)
print("Forecast-only:", result.forecast_only_dates)
print("Warnings:", result.warnings)

calibration = run_calibration(result.df, result.settled_dates)
print(calibration[["block", "time", "da_share_conservative", "da_share_balanced", "da_share_aggressive"]].head(5))
print("Aggressive extremes:", ((calibration["da_share_aggressive"] == 0) | (calibration["da_share_aggressive"] == 1)).sum(), "/ 96")

tomorrow = result.df[result.df["date"].isin(result.forecast_only_dates)]
forecast = tomorrow[["block", "day_ahead_schedule"]].rename(columns={"day_ahead_schedule": "forecast_mw"})
out = build_daily_plan(forecast, calibration, strategy="balanced")
print("Daily plan summary:", out["summary"])
print("\nAll good.")
