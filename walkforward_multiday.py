"""
True walk-forward over 21-28 Aug: for each test date D, calibrate only on
settled days strictly before D (never on D or later), plan D from the
forecast alone, then reveal D's actuals and settle. No leakage, ever.
"""
import sys
sys.path.insert(0, ".")
import pandas as pd

from engine.loader import load_workbook
from engine.calibrate import run_calibration
from engine.planning import build_daily_plan
from engine.dsm import dsm_charge, ENERGY_FACTOR

PATH = r"D:\Viyug\00. HATAGERI PH-1 (102.3 MW) (2).xlsx"
TEST_DATES = pd.date_range("2026-08-21", "2026-08-28", freq="D")


def settle(da_mw, rtm_mw, scheduled_total, actual, avc, acp, gdam_mcp, rtm_mcp, charges):
    dsm = dsm_charge(actual, scheduled_total, avc, acp)
    revenue = (da_mw * gdam_mcp + rtm_mw * rtm_mcp) * ENERGY_FACTOR
    net = revenue - dsm["total_dsm"] - charges
    return {
        "gross_revenue": revenue.sum(), "dsm_cost": dsm["total_dsm"].sum(),
        "charges": charges.sum(), "net_revenue": net.sum(),
    }


def main():
    result = load_workbook(PATH)
    df = result.df

    totals = {k: {"gross_revenue": 0, "dsm_cost": 0, "charges": 0, "net_revenue": 0}
              for k in ["desk_actual", "conservative", "balanced", "aggressive"]}
    per_day = []

    for test_date in TEST_DATES:
        calib_dates = [d for d in result.settled_dates if d < test_date]
        if len(calib_dates) < 5:
            continue
        test_rows = df[df["date"] == test_date].sort_values("block").reset_index(drop=True)
        if len(test_rows) != 96 or test_date not in result.settled_dates:
            continue  # need real actuals to score against

        calibration = run_calibration(df, calib_dates)
        forecast = test_rows[["block", "day_ahead_schedule"]].rename(columns={"day_ahead_schedule": "forecast_mw"})
        avc_by_block = dict(zip(test_rows["block"], test_rows["avc"]))
        actual = test_rows.set_index("block")

        day_result = {"date": test_date.date(), "calib_days": len(calib_dates)}
        for strategy in ["conservative", "balanced", "aggressive"]:
            out = build_daily_plan(forecast, calibration, strategy=strategy, avc_by_block=avc_by_block)
            plan = out["plan"].set_index("block")
            r = settle(plan["da_mw"], plan["rtm_mw"], plan["forecast_mw"],
                       actual["actual_ll"], actual["avc"], actual["acp"],
                       actual["gdam_mcp"], actual["rtm_mcp"], actual["total_charges"])
            for k in totals[strategy]:
                totals[strategy][k] += r[k]
            day_result[f"{strategy}_net"] = round(r["net_revenue"])

        desk_da = actual["dac_sched"] + actual["dam_sched"] + actual["gdam_sched"]
        desk_rtm = actual["rtm_sched"]
        r = settle(desk_da, desk_rtm, actual["final_schedule"], actual["actual_ll"], actual["avc"],
                   actual["acp"], actual["gdam_mcp"], actual["rtm_mcp"], actual["total_charges"])
        for k in totals["desk_actual"]:
            totals["desk_actual"][k] += r[k]
        day_result["desk_net"] = round(r["net_revenue"])
        per_day.append(day_result)

    print(f"Walk-forward over {len(per_day)} held-out days: "
          f"{per_day[0]['date']} .. {per_day[-1]['date']} (each calibrated only on prior settled days)\n")

    print(f"{'date':12}{'calib_days':>11}{'desk_net':>12}{'conservative':>14}{'balanced':>11}{'aggressive':>12}")
    for d in per_day:
        print(f"{str(d['date']):12}{d['calib_days']:>11}{d['desk_net']:>12,}{d['conservative_net']:>14,}"
              f"{d['balanced_net']:>11,}{d['aggressive_net']:>12,}")

    print(f"\n{'TOTAL (' + str(len(per_day)) + ' days)':22}{'DSM cost':>12}{'Gross rev':>13}{'Net rev':>13}{'vs desk':>12}")
    desk_net_total = totals["desk_actual"]["net_revenue"]
    for k in ["desk_actual", "conservative", "balanced", "aggressive"]:
        t = totals[k]
        delta = t["net_revenue"] - desk_net_total
        delta_str = "-" if k == "desk_actual" else f"{delta:+,.0f}"
        print(f"{k:22}{t['dsm_cost']:>12,.0f}{t['gross_revenue']:>13,.0f}{t['net_revenue']:>13,.0f}{delta_str:>12}")


if __name__ == "__main__":
    main()
