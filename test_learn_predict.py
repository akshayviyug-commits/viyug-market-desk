"""Acceptance tests for the Learn & Predict integration (engine/learn_predict.py)."""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from engine.loader import load_workbook
from engine.learn_engine import CLUSTERS, cluster_of, as_of_cutoff
from engine.learn_predict import build_days, plan_day, learn_view
from engine.predict_split import DIALS

SOURCES = {"sample workbook": Path("sample_data/sample_workbook.xlsx")}
REAL = Path(r"D:\Viyug\00. HATAGERI PH-1 (102.3 MW) (2).xlsx")
if REAL.exists():
    SOURCES["real workbook"] = REAL


def delivery_date(result):
    return result.forecast_only_dates[-1] if result.forecast_only_dates else max(result.df["date"])


def test_clusters():
    sizes = {c: sum(cluster_of(b) == c for b in range(1, 97)) for c in CLUSTERS}
    assert sizes == {"C-1": 16, "C-2": 32, "C-3": 16, "C-4": 32}, sizes
    print("PASS: 96 blocks split into clusters 16/32/16/32")


def test_thumb_rule_and_bounds(label, days, D):
    for dial in DIALS:
        p = plan_day(days, D, dial)["plan"]
        assert (p["da_mw"] + p["rtm_mw"] - p["forecast_mw"]).abs().max() < 1e-9
        assert ((p["gdam_mw"] >= 0) & (p["dam_mw"] >= 0) & (p["rtm_mw"] >= -1e-12)).all()
        assert (p["da_mw"] <= p["avc_mw"] + 1e-9).all() and (p["da_mw"] <= p["forecast_mw"] + 1e-9).all()
        assert len(p) == 96
    print(f"PASS [{label}]: thumb rule exact, volumes in bounds, 96 rows, all 3 dials")


def test_no_leak_after_cutoff(label, result, D):
    """CIP-T03: changing anything dated after the D-2 cutoff must not change Learn or Predict."""
    cutoff = pd.Timestamp(as_of_cutoff(D))
    base = build_days(result.df, result.settled_dates)
    mutated = result.df.copy()
    late = mutated["date"] > cutoff
    cols_result = ["actual_ll", "gdam_mcp", "dam_mcp", "rtm_mcp", "acp", "dac_mcp", "dac_sched", "dam_sched",
                   "gdam_sched", "rtm_sched", "final_schedule", "total_charges"]
    mutated.loc[late, cols_result] = mutated.loc[late, cols_result] * 3
    # forecast/AvC of other late days are irrelevant too; the delivery day's own forecast is a legitimate input
    other_late = late & (mutated["date"] != pd.Timestamp(D))
    mutated.loc[other_late, ["day_ahead_schedule", "avc"]] = mutated.loc[other_late, ["day_ahead_schedule", "avc"]] * 3
    changed = build_days(mutated, result.settled_dates)
    for dial in DIALS:
        a = plan_day(base, D, dial)["plan"]
        b = plan_day(changed, D, dial)["plan"]
        pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(learn_view(base, D)["cluster_table"], learn_view(changed, D)["cluster_table"])
    print(f"PASS [{label}]: nothing after {cutoff.date()} (D-2 of {D}) can change the plan")


def test_fallback_on_thin_history(label, result):
    first = str(min(result.df["date"]).date())
    days = build_days(result.df, result.settled_dates)
    view = learn_view(days, first)
    assert view["provenance"]["fallback_error_clusters"] == CLUSTERS
    p = plan_day(days, first, "balanced")["plan"]
    assert (p["da_mw"] + p["rtm_mw"] - p["forecast_mw"]).abs().max() < 1e-9
    print(f"PASS [{label}]: no history before {first} -> built-in starter sample used, plan still valid")


def test_overrides(label, days, D):
    base = plan_day(days, D, "balanced")["plan"].set_index("block")
    out = plan_day(days, D, "balanced", overrides={10: 0.5, 60: 1.0})["plan"].set_index("block")
    assert abs(out.loc[10, "da_mw"] - 0.5 * out.loc[10, "forecast_mw"]) < 1e-9
    assert abs(out.loc[60, "da_mw"] - out.loc[60, "forecast_mw"]) < 1e-9
    assert out["override_flag"].sum() == 2
    untouched = out.index.difference([10, 60])
    assert np.allclose(out.loc[untouched, "da_mw"], base.loc[untouched, "da_mw"])
    assert (out["da_mw"] + out["rtm_mw"] - out["forecast_mw"]).abs().max() < 1e-9
    print(f"PASS [{label}]: overrides replace only the chosen blocks, thumb rule still exact")


if __name__ == "__main__":
    test_clusters()
    for label, path in SOURCES.items():
        result = load_workbook(str(path))
        days = build_days(result.df, result.settled_dates)
        D = pd.Timestamp(delivery_date(result)).strftime("%Y-%m-%d")
        test_thumb_rule_and_bounds(label, days, D)
        test_no_leak_after_cutoff(label, result, D)
        test_fallback_on_thin_history(label, result)
        test_overrides(label, days, D)
    print("\nAll Learn & Predict tests passed.")
