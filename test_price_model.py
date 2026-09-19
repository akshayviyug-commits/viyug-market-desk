"""Acceptance tests for the price features and model (leak-freedom, bounds, determinism)."""
import copy
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from engine.features import CAP, FEATURES, Policy, build_panel, build_training_frame, features_for_date, is_holiday
from engine.loader import load_workbook
from engine.price_model import PriceModel, select_training

SOURCES = {"sample workbook": Path("sample_data/sample_workbook.xlsx")}
REAL = Path(os.environ.get("MARKET_DESK_REAL_WORKBOOK", "/nonexistent"))
if REAL.exists():
    SOURCES["real workbook"] = REAL


def perturbed(panel, target, policy):
    """Copy with every value the desk could NOT know at 09:00 on D-1 replaced by junk."""
    p = copy.deepcopy(panel)
    rng = np.random.default_rng(1)
    late_g = p.dates > target - pd.Timedelta(days=policy.lag_gdam)
    late_r = p.dates > target - pd.Timedelta(days=policy.lag_rtm)
    for arr in (p.gdam, p.dam):
        arr[late_g] = rng.uniform(0, CAP, arr[late_g].shape)
    for arr in (p.rtm, p.acp, p.actual, p.final):
        arr[late_r] = rng.uniform(0, CAP, arr[late_r].shape)
    return p


def frames_equal(a, b):
    return np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)


def test_no_leak(label, panel):
    for policy in (Policy(), Policy(lag_gdam=2, lag_rtm=2)):
        for i in (5, len(panel.dates) // 2, len(panel.dates) - 1):
            T = panel.dates[i]
            f1 = features_for_date(panel, T, policy=policy)
            f2 = features_for_date(perturbed(panel, T, policy), T, policy=policy)
            assert frames_equal(f1, f2), f"features changed when post-cutoff data was altered ({T:%d %b}, {policy})"
    # power check: altering data that IS inside the window must change the features
    T = panel.dates[len(panel.dates) // 2]
    p = copy.deepcopy(panel)
    p.gdam[p.dates == T - pd.Timedelta(days=1)] += 500
    assert not frames_equal(features_for_date(panel, T), features_for_date(p, T)), "test has no power"
    print(f"PASS [{label}]: features ignore everything after the D-1 / D-2 cutoffs (both policies), and react to data inside it")


def test_predictions_ignore_future(label, panel):
    T = panel.dates[len(panel.dates) // 2 + 3]
    policy = Policy()
    frame = build_training_frame(panel, policy)
    m1 = PriceModel().fit(select_training(frame, T, policy))
    p1 = m1.predict(frame[frame["date"] == T])
    pp = perturbed(panel, T, policy)
    frame2 = build_training_frame(pp, policy)
    m2 = PriceModel().fit(select_training(frame2, T, policy))
    p2 = m2.predict(frame2[frame2["date"] == T])
    assert frames_equal(p1, p2), "forecast for T changed when data after its cutoff changed"
    print(f"PASS [{label}]: forecast for {T:%d %b} is unchanged when later data is scrambled")


def test_bounds_and_determinism(label, panel):
    policy = Policy()
    frame = build_training_frame(panel, policy)
    T = panel.dates[-1]
    x = frame[frame["date"] == T]
    tr = select_training(frame, T, policy)
    a = PriceModel().fit(tr).predict(x)
    b = PriceModel().fit(tr).predict(x)
    assert frames_equal(a, b), "not deterministic"
    for m in "gr":
        assert (a[f"{m}_p50"].between(0, CAP)).all() and a[f"{m}_p10"].between(0, CAP).all() and a[f"{m}_p90"].between(0, CAP).all()
        assert (a[f"{m}_p10"] <= a[f"{m}_p50"] + 1e-9).all() and (a[f"{m}_p50"] <= a[f"{m}_p90"] + 1e-9).all()
    assert len(a) == 96 and a.notna().all().all()
    print(f"PASS [{label}]: 96 rows, no NaN, P10<=P50<=P90, all within [0, {CAP:.0f}], deterministic")


def test_thin_history_falls_back(label, panel):
    frame = build_training_frame(panel)
    T = panel.dates[2]                                    # < 3 days of history
    model = PriceModel().fit(select_training(frame, T))
    out = model.predict(frame[frame["date"] == T])
    assert not model.fitted["g"] and not model.fitted["r"]
    assert len(out) == 96 and out.notna().all().all()
    print(f"PASS [{label}]: with under 3 days of history the model falls back to the anchor (no crash, no NaN)")


def test_calendar():
    assert is_holiday("2026-08-15") and not is_holiday("2026-08-14")
    print("PASS: 15 Aug 2026 is an India national holiday; 14 Aug is not")


def test_calendar_flags(label, panel):
    T = pd.Timestamp("2026-08-14")
    if T in set(panel.dates):
        f = features_for_date(panel, T)
        assert f["hol_next"].iloc[0] == 1.0 and f["is_holiday"].iloc[0] == 0.0
        s = features_for_date(panel, pd.Timestamp("2026-08-16"))
        assert s["is_weekend"].iloc[0] == 1.0 and s["hol_prev"].iloc[0] == 1.0
        print(f"PASS [{label}]: 14 Aug flags 'holiday tomorrow'; Sunday 16 Aug is weekend and follows a holiday")


if __name__ == "__main__":
    test_calendar()
    for label, path in SOURCES.items():
        r = load_workbook(path)
        panel = build_panel(r.df, r.settled_dates)
        test_no_leak(label, panel)
        test_predictions_ignore_future(label, panel)
        test_bounds_and_determinism(label, panel)
        test_thin_history_falls_back(label, panel)
        test_calendar_flags(label, panel)
    print("\nAll price-model tests passed.")
