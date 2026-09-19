"""
Read-only comparison: our grid-search calibration vs the Learn/Predict engine,
on the same real workbook,
walk-forward over 21-28 Aug, BOTH under the platform's D-2 information cutoff.
Scoring is identical for both engines (our v1 accounting: DA leg valued at
G-DAM MCP, RTM leg at RTM MCP, DSM vs forecast, historical charges carried).
Both engines commit total = forecast, so DSM is identical by construction.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from engine.loader import load_workbook
from engine.calibrate import run_calibration
from engine.planning import build_daily_plan
from engine.dsm import dsm_charge, ENERGY_FACTOR
from engine.learn_engine import Day, build_learn_profile, LINE_LOSS
from engine.predict_split import predict_all_dials

PATH = r"D:\Viyug\00. HATAGERI PH-1 (102.3 MW) (2).xlsx"
result = load_workbook(PATH)
df = result.df
settled = set(result.settled_dates)


def iso(ts):
    return ts.strftime("%Y-%m-%d")


all_days = {}
for d, g in df.groupby("date"):
    g = g.sort_values("block")
    is_settled = d in settled
    all_days[iso(d)] = Day(
        date=iso(d), blk=g["block"].tolist(), da_sch=g["day_ahead_schedule"].tolist(), avc=g["avc"].tolist(),
        act=(g["actual_ll"] / (1 - LINE_LOSS)).tolist() if is_settled else None,   # metered = act*(1-LINE_LOSS) = actual_ll
        dac=g["dac_sched"].fillna(0).tolist(), dam=g["dam_sched"].fillna(0).tolist(), gdam=g["gdam_sched"].fillna(0).tolist(),
        mcp_gdam=g["gdam_mcp"].tolist() if is_settled else None,
        mcp_dam=g["dam_mcp"].tolist() if is_settled else None,
        mcp_rtm=g["rtm_mcp"].tolist() if is_settled else None,
        has_split=is_settled, has_prices=is_settled,
    )


def net(da, rtm, total, g):
    dsm = dsm_charge(g["actual_ll"].to_numpy(), total, g["avc"].to_numpy(), g["acp"].to_numpy())["total_dsm"]
    rev = (da * g["gdam_mcp"].to_numpy() + rtm * g["rtm_mcp"].to_numpy()) * ENERGY_FACTOR
    return float((rev - dsm - g["total_charges"].to_numpy()).sum()), float(dsm.sum())


rows, dshare = [], {k: [] for k in ["o_conservative", "o_balanced", "o_aggressive", "t_conservative", "t_balanced", "t_aggressive"]}
for D in pd.date_range("2026-08-21", "2026-08-28"):
    g = df[df["date"] == D].sort_values("block")
    cutoff = D - pd.Timedelta(days=2)                       # platform rule: D-2
    calib = [x for x in result.settled_dates if x <= cutoff]
    r = {"date": iso(D), "calib_days": len(calib)}

    # desk baselines: (a) as in my earlier test, (b) desk's DAC volume valued at the DAC price
    desk_da = (g["dac_sched"] + g["dam_sched"] + g["gdam_sched"]).to_numpy()
    r["desk_a"], _ = net(desk_da, g["rtm_sched"].to_numpy(), g["final_schedule"].to_numpy(), g)
    dac_px = g["dac_mcp"].fillna(g["gdam_mcp"]).to_numpy()
    rev_b = (g["dac_sched"].to_numpy() * dac_px + (g["dam_sched"] + g["gdam_sched"]).to_numpy() * g["gdam_mcp"].to_numpy()
             + g["rtm_sched"].to_numpy() * g["rtm_mcp"].to_numpy()) * ENERGY_FACTOR
    dsm_d = dsm_charge(g["actual_ll"].to_numpy(), g["final_schedule"].to_numpy(), g["avc"].to_numpy(), g["acp"].to_numpy())["total_dsm"]
    r["desk_b"] = float((rev_b - dsm_d - g["total_charges"].to_numpy()).sum())

    # ours (grid-search calibration) at D-2
    cal = run_calibration(df, calib)
    fc = g[["block", "day_ahead_schedule"]].rename(columns={"day_ahead_schedule": "forecast_mw"})
    for s in ["conservative", "balanced", "aggressive"]:
        p = build_daily_plan(fc, cal, strategy=s)["plan"].sort_values("block")
        r["o_" + s], dsm_o = net(p["da_mw"].to_numpy(), p["rtm_mw"].to_numpy(), p["forecast_mw"].to_numpy(), g)
        dshare["o_" + s].append(p["da_mw"].sum() / p["forecast_mw"].sum())

    # theirs (Learn + Predict port) at D-2, target stripped of everything not known in advance
    tgt = all_days[iso(D)]
    target = Day(date=tgt.date, blk=tgt.blk, da_sch=tgt.da_sch, avc=tgt.avc)
    profile = build_learn_profile(all_days, target, k=3)
    bids = predict_all_dials(target, profile)
    if iso(D) == "2026-08-21":
        print("THEIR LEARN PROFILE for 21 Aug:", profile.error.source)
        print("  cutoff:", profile.cutoff, "| commit ratio per cluster:", {k: round(v, 3) for k, v in profile.commit.ratio.items()},
              "| mirror days:", profile.commit.days_used, "| venue days:", profile.segment.days_used)
        print("  clusters on built-in fallback (err_da):", profile.error.fallback_da)
    for s, sh in bids.items():
        da = np.array(sh.gdam) + np.array(sh.dam)
        rtm = np.array(sh.rtm)
        tot = np.array(sh.total)
        assert np.abs(da + rtm - tot).max() < 1e-9, "thumb rule broken"
        r["t_" + s], dsm_t = net(da, rtm, tot, g)
        assert abs(dsm_t - dsm_o) < 1e-6, "DSM should be identical across engines (same total)"
        dshare["t_" + s].append(da.sum() / tot.sum())
    rows.append(r)

t = pd.DataFrame(rows)
cols = ["desk_a", "desk_b", "o_conservative", "o_balanced", "o_aggressive", "t_conservative", "t_balanced", "t_aggressive"]
print("\nNet revenue, Rs lakh, per held-out day (D-2 cutoff for both engines):")
print((t.set_index("date")[cols] / 1e5).round(1).to_string())
tot = t[cols].sum()
print("\n8-day totals (Rs lakh) and gap vs desk baseline (a) / (b):")
for c in cols:
    print(f"  {c:16} {tot[c]/1e5:8.1f}   vs desk_a {(tot[c]-tot['desk_a'])/1e5:+7.1f}   vs desk_b {(tot[c]-tot['desk_b'])/1e5:+7.1f}")
print("\nAverage DA share of forecast (8 days):")
for k, v in dshare.items():
    print(f"  {k:16} {np.mean(v):.1%}")
