"""
Viyug.AI Market Desk - Streamlit prototype.
Backend: pandas/numpy engine in engine/. Frontend: this file.
"""
from __future__ import annotations

import base64
import datetime
import io
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from engine.loader import load_workbook, SchemaError, DataQualityError
from engine.calibrate import run_calibration
from engine.planning import build_daily_plan, validate_forecast, ForecastValidationError
from engine import learn_predict as lp

ASSETS_DIR = Path(__file__).parent / "assets"
NAVY, SKY = "#1A3E8C", "#4A8FE8"

st.set_page_config(
    page_title="Viyug.AI - Market Desk",
    page_icon=str(ASSETS_DIR / "favicon.png"),
    layout="wide",
)


def inject_brand_css():
    css_path = ASSETS_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def sidebar_brand():
    icon_path = ASSETS_DIR / "logo_icon.png"
    if not icon_path.exists():
        st.markdown("## Viyug.AI")
        st.caption("Market Desk")
        return
    icon_b64 = base64.b64encode(icon_path.read_bytes()).decode()
    st.markdown(
        f"""
        <div class="viyug-sidebar-brand">
          <img src="data:image/png;base64,{icon_b64}" />
          <div class="brand-text">
            <div class="name">Viyug.AI</div>
            <div class="tagline">VISION TO VALUE</div>
          </div>
        </div>
        <div class="viyug-sidebar-app">Market Desk</div>
        """,
        unsafe_allow_html=True,
    )


inject_brand_css()

STRATEGY_LABELS = {"conservative": "Conservative", "balanced": "Balanced", "aggressive": "Aggressive"}
DIAL_KEYS = ["conservative", "balanced", "aggressive"]


def init_state():
    st.session_state.setdefault("load_result", None)
    st.session_state.setdefault("calibration", None)
    st.session_state.setdefault("overrides_by_key", {})   # {(method, date, dial): {block: da_share}}
    st.session_state.setdefault("run_log", [])            # session-scoped run history (see History page)
    st.session_state.setdefault("_logged_plans", set())


def log_run(kind: str, detail: str):
    st.session_state["run_log"].insert(0, {
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "type": kind,
        "detail": detail,
    })


def log_plan_once(key: tuple, detail: str):
    if key not in st.session_state["_logged_plans"]:
        st.session_state["_logged_plans"].add(key)
        log_run("Daily Plan", detail)


def get_overrides(key: tuple) -> dict:
    return st.session_state["overrides_by_key"].get(key, {})


def set_overrides(key: tuple, value: dict):
    st.session_state["overrides_by_key"][key] = value


def get_days(result):
    """Learn/Predict inputs for the loaded workbook, built once per upload."""
    cache = st.session_state.get("_days_cache")
    if cache is None or cache[0] is not result:
        cache = (result, lp.build_days(result.df, result.settled_dates))
        st.session_state["_days_cache"] = cache
    return cache[1]


def delivery_date_picker(result, key: str, forecast_only: bool = False):
    """Delivery date choice, remembered across pages. Learn & Predict can also replay a
    settled day - it still only sees what was knowable two days before."""
    forecast_dates = set(result.forecast_only_dates)
    all_dates = sorted(pd.Timestamp(d) for d in result.df["date"].unique())
    options = [d for d in all_dates if d in forecast_dates] if forecast_only else all_dates
    if not options:
        return None
    saved = st.session_state.get("delivery_date")
    index = options.index(saved) if saved in options else len(options) - 1
    choice = st.selectbox(
        "Delivery date", options, index=index, key=key,
        format_func=lambda d: f"{d:%d %b %Y}" + ("  -  forecast" if d in forecast_dates else "  -  replay of a settled day"),
    )
    st.session_state["delivery_date"] = choice
    return choice


def days_span(prov: dict, short: bool = False) -> str:
    used = prov["days_used"]
    if not used:
        return "built-in sample"
    first, last = pd.Timestamp(used[0]), pd.Timestamp(used[-1])
    return f"{first:%d %b} - {last:%d %b}" if short else f"{first:%d %b} - {last:%d %b %Y}"


def provenance_notes(prov: dict):
    used = prov["days_used"]
    if used:
        st.caption(
            f"Learned from {len(used)} settled day(s), {days_span(prov)}. Walk-forward: only days up to "
            f"{pd.Timestamp(prov['cutoff']):%d %b %Y} (two days before delivery) are ever read - the day before "
            f"is still running at bid time."
        )
    else:
        st.caption("No settled history before this date - Learn is using its built-in starter sample.")
    if used and prov["fallback_error_clusters"]:
        st.warning(
            f"Fewer than 20 observations for {', '.join(prov['fallback_error_clusters'])} - the built-in starter "
            f"sample is filling in for those windows. Real history takes over automatically as days accumulate."
        )
    if used and prov["commit_fallback"]:
        st.warning("No recent day carries the desk's own split - typical commit ratios are being used.")
    if used and prov["venue_fallback"]:
        st.warning("No recent priced day on file - a seasonal default venue is being used per window.")


def show_diagram(filename: str):
    path = ASSETS_DIR / filename
    if path.exists():
        st.image(str(path), use_container_width=True)


def forecast_frame(result, date):
    day = result.df[result.df["date"] == date]
    forecast = day[["block", "day_ahead_schedule"]].rename(columns={"day_ahead_schedule": "forecast_mw"})
    return forecast, dict(zip(day["block"], day["avc"]))


# ---------- page renderers ----------
def render_overview(result):
    st.title("Market Desk")
    st.caption(f"DA / RTM planning · {result.n_settled_days} settled days on file "
               f"({result.date_range[0].date()} - {result.date_range[1].date()})")

    days = get_days(result)
    latest = max(pd.Timestamp(d) for d in result.df["date"].unique())
    view = lp.learn_view(days, latest.strftime("%Y-%m-%d"))
    prov = view["provenance"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Settled days on file", result.n_settled_days)
    c2.metric("Next delivery date", f"{latest:%d %b %Y}")
    c3.metric("Learned from", f"{len(prov['days_used'])} days")

    fig = go.Figure(go.Bar(x=list(view["commit_ratio"].keys()),
                           y=list(view["commit_ratio"].values()), marker_color=NAVY))
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], height=280,
                      margin=dict(t=40, l=10, r=10, b=10),
                      title="How much of the forecast the desk usually commits a day ahead, by time of day")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Learn shows what history says. Daily Plan turns it into tomorrow's DA / RTM volumes for all 96 blocks.")

    calibration = st.session_state["calibration"]
    if calibration is not None:
        st.subheader("Backtest calibration")
        st.plotly_chart(
            go.Figure(go.Scatter(x=calibration["time"], y=calibration["da_share_balanced"], mode="lines",
                                 line=dict(color=NAVY)))
            .update_layout(yaxis_tickformat=".0%", height=280, margin=dict(t=10, l=10, r=10, b=10),
                           title="Balanced DA share by block"),
            use_container_width=True,
        )


def render_learn(result):
    st.title("Learn")
    st.caption("What the desk's own history says: how wrong the forecast usually is, how much the desk usually "
               "commits a day ahead, and which day-ahead market has been paying more.")
    days = get_days(result)
    date = delivery_date_picker(result, "dd_learn")
    try:
        view = lp.learn_view(days, date.strftime("%Y-%m-%d"))
    except ForecastValidationError as e:
        st.error(str(e))
        st.stop()
    prov = view["provenance"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Days learned from", len(prov["days_used"]))
    c2.metric("History window", days_span(prov, short=True))
    c3.metric("Cutoff (D-2)", f"{pd.Timestamp(prov['cutoff']):%d %b %Y}")
    provenance_notes(prov)

    st.subheader("Profile by time of day")
    table = view["cluster_table"].copy()
    table["Desk day-ahead commit"] = (table["Desk day-ahead commit"] * 100).round(1)
    for col in ["Forecast error P25 (MW)", "Forecast error P50 (MW)", "Forecast error P75 (MW)"]:
        table[col] = table[col].round(1)
    st.dataframe(
        table, use_container_width=True, hide_index=True,
        column_config={"Desk day-ahead commit": st.column_config.NumberColumn("Desk day-ahead commit (%)", format="%.1f")},
    )
    st.caption("Forecast error = metered output minus the day-ahead forecast, pooled across every prior block in the window. "
               "A negative median means the forecast usually over-promises.")

    box = go.Figure()
    for cluster, errors in view["error_by_cluster"].items():
        box.add_trace(go.Box(y=errors, name=cluster, marker_color=NAVY, boxpoints=False))
    box.add_hline(y=0, line_dash="dot", line_color="gray")
    box.update_layout(height=320, margin=dict(t=10, l=10, r=10, b=10), showlegend=False,
                      yaxis_title="MW (actual - forecast)")
    st.subheader("Forecast error by time of day")
    st.plotly_chart(box, use_container_width=True)

    st.subheader("Which day-ahead market has paid more, block by block")
    spread = pd.Series(view["spread"], dtype=float)
    if spread.isna().all():
        st.info("No priced history yet - each window uses its seasonal default venue.")
    else:
        colors = [NAVY if (s == s and s >= 0) else SKY for s in spread]
        bars = go.Figure(go.Bar(x=view["times"], y=spread, marker_color=colors))
        bars.add_hline(y=0, line_color="gray")
        bars.update_layout(height=300, margin=dict(t=10, l=10, r=10, b=10),
                           yaxis_title="G-DAM minus (DAM + REC), Rs/MWh")
        st.plotly_chart(bars, use_container_width=True)
        st.caption("Navy: G-DAM has paid more over the last few days for that block, so it is preferred. "
                   "Light blue: DAM (plus the REC uplift) has paid more.")

    with st.expander("How Learn works"):
        show_diagram("learn_pipeline.png")


def bid_export(plan: pd.DataFrame) -> bytes:
    cols = ["block", "time", "forecast_mw", "gdam_mw", "dam_mw", "rtm_mw", "da_share_applied", "override_flag"]
    return plan[cols].round(4).to_csv(index=False).encode()


def render_plan_learn_predict(result):
    days = get_days(result)
    date = delivery_date_picker(result, "dd_plan")
    dial = st.radio("Strategy", DIAL_KEYS, format_func=lambda s: STRATEGY_LABELS[s],
                    horizontal=True, key="plan_strategy")
    iso = date.strftime("%Y-%m-%d")
    ov_key = ("lp", iso, dial)

    try:
        out = lp.plan_day(days, iso, dial, overrides=get_overrides(ov_key))
    except ForecastValidationError as e:
        st.error(str(e))
        st.stop()

    log_plan_once(ov_key, f"{date:%d %b %Y} - {STRATEGY_LABELS[dial]} (Learn & Predict)")
    for w in out["warnings"]:
        st.warning(w)
    if date not in set(result.forecast_only_dates):
        st.info("Replay: this date is already settled in the file. The plan below uses only what was knowable "
                "two days before delivery - it never sees that day's actuals.")
    provenance_notes(out["provenance"])

    s, plan = out["summary"], out["plan"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total forecast", f"{s['total_forecast_mwh']:.0f} MWh")
    c2.metric("Day-ahead", f"{s['overall_da_share']:.0%}", f"{s['total_da_mwh']:.0f} MWh")
    c3.metric("RTM", f"{1 - s['overall_da_share']:.0%}", f"{s['total_rtm_mwh']:.0f} MWh")
    c4.metric("Overrides", s["override_count"])
    m1, m2, m3 = st.columns(3)
    m1.metric("G-DAM", f"{s['total_gdam_mwh']:.0f} MWh")
    m2.metric("DAM", f"{s['total_dam_mwh']:.0f} MWh")
    if "indicative_revenue_inr" in s:
        m3.metric("Indicative revenue (not a forecast)", f"Rs {s['indicative_revenue_inr']:,.0f}")
        st.caption(s["indicative_revenue_note"])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=plan["time"], y=plan["gdam_mw"], name="G-DAM", marker_color=NAVY))
    fig.add_trace(go.Bar(x=plan["time"], y=plan["dam_mw"], name="DAM",
                         marker=dict(color="#DCE6F8", pattern=dict(shape="/", fgcolor=NAVY, size=6))))
    fig.add_trace(go.Bar(x=plan["time"], y=plan["rtm_mw"], name="RTM", marker_color=SKY))
    fig.update_layout(barmode="stack", height=350, margin=dict(t=10, l=10, r=10, b=10),
                      legend=dict(orientation="h", y=1.1), yaxis_title="MW")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Day-ahead + RTM equals the forecast in every block, exactly - nothing is held back.")

    with st.expander("Edit block overrides"):
        st.caption("Each block defaults to the strategy's own day-ahead share. An override replaces it for that block; "
                   "RTM takes the rest, so day-ahead + RTM still equals the forecast.")
        edited = st.data_editor(
            plan[["block", "time", "forecast_mw", "da_share_applied"]],
            disabled=["block", "time", "forecast_mw"], hide_index=True, use_container_width=True,
            key="override_editor_" + "_".join(map(str, ov_key)),
            column_config={"da_share_applied": st.column_config.NumberColumn(
                "DA share", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")},
        )
        if st.button("Apply overrides"):
            base = plan.set_index("block")["base_da_share"]
            set_overrides(ov_key, {int(r.block): float(r.da_share_applied) for r in edited.itertuples()
                                   if abs(r.da_share_applied - base.loc[int(r.block)]) > 1e-9})
            st.rerun()

    with st.expander("Show block details"):
        st.dataframe(plan.drop(columns=["base_da_share"]).round(3), use_container_width=True, hide_index=True)

    st.download_button("Export bid file (CSV)", bid_export(plan), f"bid_{date:%Y%m%d}_{dial}.csv", "text/csv")

    with st.expander("How Predict works"):
        show_diagram("predict_pipeline.png")


def render_plan_backtest(result):
    calibration = st.session_state["calibration"]
    if calibration is None:
        st.warning("Run calibration first (Calibration page) - this method applies the calibrated per-block shares.")
        st.stop()
    if not result.forecast_only_dates:
        st.info("No forecast-only date found in the uploaded workbook. Upload a forecast to plan a new day.")
        st.stop()

    delivery_date = delivery_date_picker(result, "dd_plan_bt", forecast_only=True)
    forecast, avc_by_block = forecast_frame(result, delivery_date)
    strategy = st.radio("Strategy", DIAL_KEYS, format_func=lambda s: STRATEGY_LABELS[s],
                        horizontal=True, key="plan_strategy_bt")
    ov_key = ("bt", delivery_date.strftime("%Y-%m-%d"), strategy)

    try:
        out = build_daily_plan(forecast, calibration, strategy=strategy,
                               overrides=get_overrides(ov_key), avc_by_block=avc_by_block)
    except ForecastValidationError as e:
        st.error(str(e))
        st.stop()

    log_plan_once(ov_key, f"{delivery_date:%d %b %Y} - {STRATEGY_LABELS[strategy]} (Backtest calibration)")
    for w in out["warnings"]:
        st.warning(w)

    s = out["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total forecast", f"{s['total_forecast_mwh']:.0f} MWh")
    c2.metric("DA", f"{s['overall_da_share']:.0%}", f"{s['total_da_mwh']:.0f} MWh")
    c3.metric("RTM", f"{1 - s['overall_da_share']:.0%}", f"{s['total_rtm_mwh']:.0f} MWh")
    c4.metric("Overrides", s["override_count"])
    if "indicative_revenue_inr" in s:
        st.metric("Indicative revenue (not a forecast)", f"Rs {s['indicative_revenue_inr']:,.0f}")
        st.caption(s["indicative_revenue_note"])

    plan = out["plan"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=plan["time"], y=plan["da_mw"], name="DA", marker_color=NAVY))
    fig.add_trace(go.Bar(x=plan["time"], y=plan["rtm_mw"], name="RTM", marker_color=SKY))
    fig.update_layout(barmode="stack", height=350, margin=dict(t=10, l=10, r=10, b=10),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Edit block overrides"):
        st.caption("Every block defaults to the calibrated share; overrides log the applied value.")
        edited = st.data_editor(
            plan[["block", "time", "forecast_mw", "da_share_applied"]],
            disabled=["block", "time", "forecast_mw"], hide_index=True, use_container_width=True,
            key="override_editor_" + "_".join(map(str, ov_key)),
        )
        if st.button("Apply overrides"):
            base_share = calibration.set_index("block")[f"da_share_{strategy}"]
            set_overrides(ov_key, {int(r.block): float(r.da_share_applied) for r in edited.itertuples()
                                   if abs(r.da_share_applied - base_share.loc[int(r.block)]) > 1e-9})
            st.rerun()

    with st.expander("Show block details"):
        st.dataframe(plan, use_container_width=True, hide_index=True)

    st.download_button("Export bid file (CSV)", plan.to_csv(index=False).encode(),
                       f"bid_{delivery_date:%Y%m%d}_{strategy}.csv", "text/csv")


init_state()

with st.sidebar:
    sidebar_brand()
    page = st.radio("Navigate", ["Overview", "Learn", "Daily Plan", "Calibration", "History"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("#### Data")
    uploaded = st.file_uploader("Upload settlement workbook (.xlsx)", type=["xlsx"])
    if uploaded is not None:
        # st.file_uploader keeps returning the same UploadedFile on every
        # rerun (not just at upload time) - e.g. clicking a strategy radio
        # or switching pages also re-enters this block. Only reprocess (and
        # reset calibration) when the uploaded file has actually changed.
        file_id = getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)
        if st.session_state.get("_uploaded_file_id") != file_id:
            try:
                # Read straight from the in-memory upload - never written to disk,
                # never leaves this session's server-side memory.
                st.session_state["load_result"] = load_workbook(io.BytesIO(uploaded.getvalue()))
                st.session_state["calibration"] = None
                st.session_state["_uploaded_file_id"] = file_id
                st.success(f"Loaded {uploaded.name}")
            except (SchemaError, DataQualityError) as e:
                st.error(str(e))

    default_path = Path(__file__).parent / "sample_data" / "sample_workbook.xlsx"
    if st.session_state["load_result"] is None and default_path.exists():
        if st.button("Load sample workbook"):
            try:
                st.session_state["load_result"] = load_workbook(str(default_path))
            except (SchemaError, DataQualityError) as e:
                st.error(str(e))

result = st.session_state["load_result"]

if result is None:
    st.title("Market Desk")
    st.info("Upload a settlement workbook, or load the sample workbook, from the sidebar to begin.")
    st.stop()

if page == "Overview":
    render_overview(result)

elif page == "Learn":
    render_learn(result)

elif page == "Daily Plan":
    st.title("Daily Plan")
    method = st.radio("Method", ["Learn & Predict", "Backtest calibration"], horizontal=True, key="plan_method",
                      help="Learn & Predict sizes each commitment from history's forecast-error pattern. "
                           "Backtest calibration applies the per-block shares found on the Calibration page.")
    if method == "Learn & Predict":
        render_plan_learn_predict(result)
    else:
        render_plan_backtest(result)

elif page == "Calibration":
    st.title("Calibration")
    st.caption("Historical performance translated into a block-wise DA allocation strategy.")
    st.caption("Backtest calibration is the alternative to Learn & Predict: it tries every candidate day-ahead share "
               "for each of the 96 blocks against settled history and keeps the best one per risk posture.")

    settled_start, settled_end = min(result.settled_dates), max(result.settled_dates)
    c1, c2, c3 = st.columns(3)
    c1.metric("Period", f"{result.n_settled_days} days")
    c2.metric("Blocks", f"{result.n_settled_days * 96:,}")
    c3.metric("Date range", f"{settled_start.strftime('%d %b')} - {settled_end.strftime('%d %b %Y')}")
    if result.warnings:
        for w in result.warnings:
            st.warning(w)
    else:
        st.success("Validation passed")

    if st.button("Run calibration", type="primary"):
        with st.spinner("Backtesting 96 blocks against settled history..."):
            st.session_state["calibration"] = run_calibration(result.df, result.settled_dates)
        log_run("Calibration", f"{result.n_settled_days} settled days ({settled_start:%d %b} - {settled_end:%d %b %Y})")
        st.success("Calibration complete.")

    calibration = st.session_state["calibration"]
    if calibration is not None:
        strategy = st.radio("Strategy", DIAL_KEYS, format_func=lambda s: STRATEGY_LABELS[s],
                            horizontal=True, key="cal_strategy")
        share_col = f"da_share_{strategy}"
        avg_col = f"avg_net_revenue_{strategy}"
        worst_col = f"worst_day_net_revenue_{strategy}"

        st.subheader("Recommended DA share by block")
        st.caption(
            f"Not tied to a single delivery date - this is the reference table backtested against "
            f"all {result.n_settled_days} settled days on file "
            f"({min(result.settled_dates):%d %b} - {max(result.settled_dates):%d %b %Y}), one recommended "
            f"share per time-of-day block. Daily Plan (Backtest calibration method) applies this same table to "
            f"whichever date you're planning."
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=calibration["time"], y=calibration["baseline_da_share"],
                                 mode="lines", name="Baseline (current desk)", line=dict(dash="dot", color="gray")))
        fig.add_trace(go.Scatter(x=calibration["time"], y=calibration[share_col],
                                 mode="lines", name=STRATEGY_LABELS[strategy], line=dict(color=NAVY)))
        fig.update_layout(yaxis_tickformat=".0%", height=350, margin=dict(t=10, l=10, r=10, b=10),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Average DA share", f"{calibration[share_col].mean():.0%}")
        m2.metric("Worst-day net revenue (avg across blocks)", f"Rs {calibration[worst_col].sum():,.0f}")
        m3.metric("Average net revenue (sum across blocks)", f"Rs {calibration[avg_col].sum():,.0f}")

        with st.expander("View block-level calibration"):
            show_cols = ["block", "time", share_col, "baseline_da_share", avg_col, worst_col,
                         "avg_actual_generation_mw", "coefficient_of_variation"]
            st.dataframe(calibration[show_cols], use_container_width=True, hide_index=True)

        csv = calibration.to_csv(index=False).encode()
        st.download_button("Download full calibration table (CSV)", csv, "calibration_output.csv", "text/csv")

elif page == "History":
    st.title("History")
    st.caption("Runs in this session.")
    log = st.session_state["run_log"]
    if not log:
        st.info("Nothing run yet this session - generate a daily plan or run a calibration to see it logged here.")
    else:
        st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)
        st.caption(
            "This log lives only in your current browser session - it resets on page reload. "
            "Durable, cross-session history is a platform feature (the `Run` entity in the "
            "requirements spec) that needs a real backend, not yet wired up in this prototype."
        )
