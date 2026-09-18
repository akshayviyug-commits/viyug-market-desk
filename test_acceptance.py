"""Automated version of Market_Desk_Requirements.docx section 10 acceptance tests."""
import sys
sys.path.insert(0, ".")
from engine.loader import load_workbook
from engine.calibrate import run_calibration
from engine.planning import build_daily_plan
from engine.dsm import ENERGY_FACTOR

PATH = r"D:\Viyug\00. HATAGERI PH-1 (102.3 MW) (2).xlsx"


def test_energy_conversion():
    assert 100 * ENERGY_FACTOR == 25, "100 MW for a 15-min block must be 25 MWh, not 100."
    print("PASS: energy conversion (100 MW block -> 25 MWh)")


def test_load_and_identities():
    result = load_workbook(PATH)  # raises DataQualityError internally if identities/96-block fail
    assert result.n_settled_days > 0
    print(f"PASS: load + identities + 96-block guardrail ({result.n_settled_days} settled days, "
          f"{result.date_range[0].date()} .. {result.date_range[1].date()})")
    return result


def test_thumb_rule_and_no_negatives(result):
    calibration = run_calibration(result.df, result.settled_dates)
    tomorrow = result.df[result.df["date"].isin(result.forecast_only_dates)]
    forecast = tomorrow[["block", "day_ahead_schedule"]].rename(columns={"day_ahead_schedule": "forecast_mw"})
    for strategy in ["conservative", "balanced", "aggressive"]:
        out = build_daily_plan(forecast, calibration, strategy=strategy)
        plan = out["plan"]
        err = (plan["da_mw"] + plan["rtm_mw"] - plan["forecast_mw"]).abs().max()
        assert err < 1e-9, f"{strategy}: thumb rule violated by {err}"
        assert (plan["da_mw"] >= 0).all() and (plan["rtm_mw"] >= 0).all(), f"{strategy}: negative volume"
    print("PASS: thumb rule (DA+RTM=Forecast exactly) and no negative volumes, all 3 strategies")
    return calibration


if __name__ == "__main__":
    test_energy_conversion()
    result = test_load_and_identities()
    test_thumb_rule_and_no_negatives(result)
    print("\nAll acceptance tests passed.")
