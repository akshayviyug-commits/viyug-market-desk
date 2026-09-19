"""
Walk-forward validation: calibrate on 1-20 Aug only, plan 21 Aug using ONLY
what would genuinely be known in advance (the forecast + AvC - no prices, no
actual generation, no settlement data), then reveal 21 Aug's real outcome and
settle each candidate plan against it. Compares our three strategies against
what the desk actually did on 21 Aug, on the same real settlement math.
"""
import sys
sys.path.insert(0, ".")
import pandas as pd

from engine.loader import load_workbook
from engine.calibrate import run_calibration
from engine.planning import build_daily_plan
from engine.dsm import dsm_charge, ENERGY_FACTOR

PATH = r"D:\Viyug\00. HATAGERI PH-1 (102.3 MW) (2).xlsx"
CALIBRATION_END = pd.Timestamp("2026-08-20")
TEST_DATE = pd.Timestamp("2026-08-21")


def settle(da_mw, rtm_mw, scheduled_total, actual, avc, acp, gdam_mcp, rtm_mcp, charges):
    dsm = dsm_charge(actual, scheduled_total, avc, acp)
    revenue = (da_mw * gdam_mcp + rtm_mw * rtm_mcp) * ENERGY_FACTOR
    net = revenue - dsm["total_dsm"] - charges
    return {
        "gross_revenue": round(revenue.sum()),
        "dsm_cost": round(dsm["total_dsm"].sum()),
        "charges": round(charges.sum()),
        "net_revenue": round(net.sum()),
        "da_mwh": round(da_mw.sum() * ENERGY_FACTOR, 1),
        "rtm_mwh": round(rtm_mw.sum() * ENERGY_FACTOR, 1),
    }


def main():
    result = load_workbook(PATH)
    df = result.df

    calib_dates = [d for d in result.settled_dates if d <= CALIBRATION_END]
    print(f"Calibration window: {len(calib_dates)} settled days, "
          f"{min(calib_dates).date()} .. {max(calib_dates).date()}")
    print(f"Test date (held out entirely from calibration): {TEST_DATE.date()}")
    print(f"({TEST_DATE.date()} onward was NEVER shown to the calibration step.)\n")

    calibration = run_calibration(df, calib_dates)

    test_rows = df[df["date"] == TEST_DATE].sort_values("block").reset_index(drop=True)
    assert len(test_rows) == 96, f"Expected 96 rows for {TEST_DATE.date()}, got {len(test_rows)}"

    # Planning input: ONLY what's knowable in advance. No actual_ll, no MCPs,
    # no total_charges - those get revealed only after plans are built.
    forecast = test_rows[["block", "day_ahead_schedule"]].rename(columns={"day_ahead_schedule": "forecast_mw"})
    avc_by_block = dict(zip(test_rows["block"], test_rows["avc"]))

    plans = {}
    for strategy in ["conservative", "balanced", "aggressive"]:
        plans[strategy] = build_daily_plan(forecast, calibration, strategy=strategy, avc_by_block=avc_by_block)

    # ---- reveal actuals for 21 Aug, settle each plan against them ----
    actual = test_rows.set_index("block")
    results = {}
    for strategy, out in plans.items():
        plan = out["plan"].set_index("block")
        results[strategy] = settle(
            da_mw=plan["da_mw"], rtm_mw=plan["rtm_mw"],
            scheduled_total=plan["forecast_mw"],  # thumb rule: DA+RTM = forecast, always
            actual=actual["actual_ll"], avc=actual["avc"], acp=actual["acp"],
            gdam_mcp=actual["gdam_mcp"], rtm_mcp=actual["rtm_mcp"], charges=actual["total_charges"],
        )

    # ---- what the desk actually did on 21 Aug, settled the same way ----
    desk_da_mw = actual["dac_sched"] + actual["dam_sched"] + actual["gdam_sched"]
    desk_rtm_mw = actual["rtm_sched"]
    results["desk_actual"] = settle(
        da_mw=desk_da_mw, rtm_mw=desk_rtm_mw, scheduled_total=actual["final_schedule"],
        actual=actual["actual_ll"], avc=actual["avc"], acp=actual["acp"],
        gdam_mcp=actual["gdam_mcp"], rtm_mcp=actual["rtm_mcp"], charges=actual["total_charges"],
    )

    total_forecast_mwh = round(forecast["forecast_mw"].sum() * ENERGY_FACTOR, 1)
    total_actual_mwh = round(actual["actual_ll"].sum() * ENERGY_FACTOR, 1)
    print(f"21 Aug forecast (known in advance): {total_forecast_mwh} MWh")
    print(f"21 Aug actual generation (revealed after): {total_actual_mwh} MWh  "
          f"(forecast error: {round(total_actual_mwh - total_forecast_mwh, 1):+} MWh, "
          f"{round((total_actual_mwh/total_forecast_mwh - 1)*100, 1):+}%)\n")

    order = ["desk_actual", "conservative", "balanced", "aggressive"]
    print(f"{'':14}{'DA MWh':>9}{'RTM MWh':>9}{'Gross Rev':>13}{'DSM Cost':>11}{'Charges':>10}{'Net Rev':>13}{'vs Desk':>12}")
    desk_net = results["desk_actual"]["net_revenue"]
    for k in order:
        r = results[k]
        delta = r["net_revenue"] - desk_net
        delta_str = "-" if k == "desk_actual" else f"{delta:+,}"
        print(f"{k:14}{r['da_mwh']:>9}{r['rtm_mwh']:>9}{r['gross_revenue']:>13,}{r['dsm_cost']:>11,}"
              f"{r['charges']:>10,}{r['net_revenue']:>13,}{delta_str:>12}")

    return results, total_forecast_mwh, total_actual_mwh


if __name__ == "__main__":
    main()
