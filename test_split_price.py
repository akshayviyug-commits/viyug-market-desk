"""Acceptance tests for the price-driven split (thumb rule, bounds, guardrail, dials, overrides)."""
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from engine.bundle import build_bundle
from engine.price_forecast import Forecaster
from engine.split_price import DIAL_TILT, cluster_table, price_split

SAMPLE = Path("sample_data/sample_workbook.xlsx")


def prices_frame(g50, r50, width=600.0):
    b = np.arange(1, 97)
    f = pd.DataFrame({"block": b})
    for m, v in (("g", g50), ("r", r50)):
        f[f"{m}_p50"] = v
        f[f"{m}_p10"] = v - width
        f[f"{m}_p90"] = v + width
    return f


def test_identities(b, T, label):
    for dial in DIAL_TILT:
        pr = prices_frame(np.full(96, 5000.0), np.linspace(2000, 9000, 96))
        p = price_split(b, T, pr, dial)["plan"]
        assert len(p) == 96
        assert (p["gdam_mw"] + p["rtm_mw"] - p["forecast_mw"]).abs().max() < 1e-9
        assert ((p["gdam_mw"] >= 0) & (p["rtm_mw"] >= -1e-9) & (p["dam_mw"] == 0)).all()
        assert (p["gdam_mw"] <= p["avc_mw"] + 1e-9).all()
    print(f"PASS [{label}]: thumb rule exact, no negatives, day-ahead within capacity, 96 rows, all 3 dials")


def test_guardrail_and_dials(b, T, label):
    rtm_wins = prices_frame(np.full(96, 4000.0), np.full(96, 7000.0))       # RTM clearly better
    g_wins = prices_frame(np.full(96, 7000.0), np.full(96, 4000.0))         # G-DAM clearly better
    res = {}
    for dial in DIAL_TILT:
        a = price_split(b, T, rtm_wins, dial)
        c = price_split(b, T, g_wins, dial)
        res[dial] = (a["summary"]["gdam_share"], c["summary"]["gdam_share"], c["plan"])
        pl = c["plan"]
        # guardrail: never more day-ahead than the forecast-error volume, whatever the price says
        assert (pl["da_share_applied"] <= pl["ceiling_da_share"] + 1e-9).all()
        assert a["summary"]["gdam_share"] < c["summary"]["gdam_share"], "prices favouring RTM must lower G-DAM share"
    # a stronger dial reacts more to the same price signal
    swing = {d: res[d][1] - res[d][0] for d in res}
    assert swing["aggressive"] >= swing["balanced"] - 1e-9, swing
    # conservative never goes above the desk's own habit
    cons = res["conservative"][2]
    assert (cons["da_share_applied"] <= cons["habit_da_share"] + 1e-9).all()
    print(f"PASS [{label}]: RTM-favouring prices lower the G-DAM share, G-DAM-favouring raise it; guardrail respected; "
          f"Conservative never exceeds desk habit; swing {({d: round(v, 2) for d, v in swing.items()})}")


def test_no_edge_stays_near_habit(b, T, label):
    flat = prices_frame(np.full(96, 5000.0), np.full(96, 4700.0), width=3000.0)     # RTM+REC = G, huge uncertainty
    out = price_split(b, T, flat, "balanced")["plan"]
    assert (out["confidence_z"].abs() < 0.2).all()
    assert (out["da_share_applied"] - np.minimum(out["habit_da_share"], out["ceiling_da_share"])).abs().max() < 0.05
    print(f"PASS [{label}]: with no price edge the plan stays at the desk's habit (within the guardrail)")


def test_override(b, T, label):
    pr = prices_frame(np.full(96, 5000.0), np.linspace(2000, 9000, 96))
    out = price_split(b, T, pr, "balanced", overrides={10: 0.3, 50: 1.0})
    p = out["plan"].set_index("block")
    assert abs(p.loc[10, "da_share_applied"] - 0.3) < 1e-9 and p.loc[10, "override_flag"]
    assert p.loc[50, "gdam_mw"] <= p.loc[50, "avc_mw"] + 1e-9 and p.loc[50, "gdam_mw"] <= p.loc[50, "forecast_mw"] + 1e-9
    assert (p["gdam_mw"] + p["rtm_mw"] - p["forecast_mw"]).abs().max() < 1e-9
    assert out["summary"]["override_count"] == 2
    print(f"PASS [{label}]: overrides applied to the chosen blocks only, thumb rule still exact")


def test_cluster_table(b, T, label):
    pr = prices_frame(np.full(96, 5000.0), np.linspace(2000, 9000, 96))
    out = price_split(b, T, pr, "balanced")
    ct = cluster_table(out["plan"])
    assert list(ct["cluster"]) == ["C-1", "C-2", "C-3", "C-4"]
    assert abs(ct["Forecast (MWh)"].sum() - out["summary"]["total_forecast_mwh"]) < 0.05
    assert abs(ct["G-DAM (MWh)"].sum() + ct["RTM (MWh)"].sum() - ct["Forecast (MWh)"].sum()) < 1e-6
    assert ((ct["G-DAM %"] + ct["RTM %"]).sub(100).abs() < 1e-6).all()
    print(f"PASS [{label}]: c-1..c-4 table adds back to the forecast exactly")


def test_guardrail_switch(b, T, label):
    g_wins = prices_frame(np.full(96, 7000.0), np.full(96, 4000.0))
    on = price_split(b, T, g_wins, "balanced", guardrail=True)["summary"]["gdam_share"]
    off = price_split(b, T, g_wins, "balanced", guardrail=False)["summary"]["gdam_share"]
    assert off >= on - 1e-9
    habit = price_split(b, T, g_wins, "balanced", guardrail=False, tilt=0.0)["plan"]
    cap_share = np.where(habit["forecast_mw"] > 0, habit["avc_mw"] / habit["forecast_mw"].where(habit["forecast_mw"] > 0, 1), 1.0)
    expected = np.where(habit["forecast_mw"] > 0, np.minimum(habit["habit_da_share"], cap_share), 0.0)   # capped at capacity; no forecast -> no volume
    assert (habit["da_share_applied"] - expected).abs().max() < 1e-9
    print(f"PASS [{label}]: guardrail off never lowers day-ahead ({on:.0%} -> {off:.0%}); tilt 0 reproduces the desk's habit exactly")


def test_real_forecast_runs(b, T, label):
    f = Forecaster(b).forecast(T)
    assert len(f) == 96 and f[["g_p10", "g_p50", "g_p90", "r_p10", "r_p50", "r_p90"]].notna().all().all()
    assert (f["g_p10"] <= f["g_p50"] + 1e-9).all() and (f["g_p50"] <= f["g_p90"] + 1e-9).all()
    assert (f["r_p10"] <= f["r_p50"] + 1e-9).all() and (f["r_p50"] <= f["r_p90"] + 1e-9).all()
    out = price_split(b, T, f, "balanced")
    print(f"PASS [{label}]: end-to-end forecast -> split for {T:%d %b}: {out['summary']['gdam_share']:.0%} G-DAM / "
          f"{out['summary']['rtm_share']:.0%} RTM; model info {({k: v for k, v in f.attrs['info'].items() if not k.startswith('features')})}")


if __name__ == "__main__":
    runs = []
    bs = build_bundle(str(SAMPLE))
    runs.append(("sample workbook (workbook-only mode)", bs, pd.Timestamp(bs.workbook.forecast_only_dates[-1])))
    real = Path(os.environ.get("MARKET_DESK_REAL_WORKBOOK", "/nonexistent"))
    if real.exists():
        br = build_bundle(str(real), iex_dir=os.environ.get("MARKET_DESK_IEX_DIR") or None)
        runs.append((f"real workbook ({br.mode} mode)", br, pd.Timestamp(br.workbook.forecast_only_dates[-1])))
    for label, b, T in runs:
        test_identities(b, T, label)
        test_guardrail_and_dials(b, T, label)
        test_no_edge_stays_near_habit(b, T, label)
        test_override(b, T, label)
        test_cluster_table(b, T, label)
        test_guardrail_switch(b, T, label)
        test_real_forecast_runs(b, T, label)
    print("\nAll split tests passed.")
