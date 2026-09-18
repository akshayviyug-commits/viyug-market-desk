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

ASSETS_DIR = Path(__file__).parent / "assets"

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


def init_state():
    st.session_state.setdefault("load_result", None)
    st.session_state.setdefault("calibration", None)
    st.session_state.setdefault("strategy", "balanced")
    st.session_state.setdefault("overrides", {})
    st.session_state.setdefault("run_log", [])       # session-scoped run history (see History page)
    st.session_state.setdefault("_logged_plans", set())


def log_run(kind: str, detail: str):
    st.session_state["run_log"].insert(0, {
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "type": kind,
        "detail": detail,
    })


init_state()

with st.sidebar:
    sidebar_brand()
    page = st.radio("Navigate", ["Overview", "Calibration", "Daily Plan", "History"], label_visibility="collapsed")
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
    st.info("Upload a settlement workbook, or load the demo workbook, from the sidebar to begin.")
    st.stop()

# ---------- Overview ----------
if page == "Overview":
    st.title("Market Desk")
    st.caption(f"DA / RTM planning · {result.n_settled_days} settled days on file "
               f"({result.date_range[0].date()} - {result.date_range[1].date()})")

    if st.session_state["calibration"] is None:
        st.warning("Run calibration first (Calibration page) before planning tomorrow's split.")
    else:
        calibration = st.session_state["calibration"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Settled days used", result.n_settled_days)
        with col2:
            st.metric("Average DA share (Balanced)", f"{calibration['da_share_balanced'].mean():.0%}")
        st.plotly_chart(
            go.Figure(go.Scatter(x=calibration["time"], y=calibration["da_share_balanced"], mode="lines"))
            .update_layout(yaxis_tickformat=".0%", height=300, margin=dict(t=10, l=10, r=10, b=10),
                            title="Balanced DA share by block"),
            use_container_width=True,
        )
    if result.forecast_only_dates:
        st.caption(f"Forecast-only day(s) available for planning: "
                   + ", ".join(d.strftime("%Y-%m-%d") for d in result.forecast_only_dates))

# ---------- Calibration ----------
elif page == "Calibration":
    st.title("Calibration")
    st.caption("Historical performance translated into a block-wise DA allocation strategy.")

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
        strategy = st.radio("Strategy", ["conservative", "balanced", "aggressive"],
                             format_func=lambda s: STRATEGY_LABELS[s], horizontal=True, key="cal_strategy")
        share_col = f"da_share_{strategy}"
        avg_col = f"avg_net_revenue_{strategy}"
        worst_col = f"worst_day_net_revenue_{strategy}"

        st.subheader("Recommended DA share by block")
        st.caption(
            f"Not tied to a single delivery date - this is the reference table backtested against "
            f"all {result.n_settled_days} settled days on file "
            f"({min(result.settled_dates):%d %b} - {max(result.settled_dates):%d %b %Y}), one recommended "
            f"share per time-of-day block. Daily Plan applies this same table to whichever date you're planning."
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=calibration["time"], y=calibration["baseline_da_share"],
                                  mode="lines", name="Baseline (current desk)", line=dict(dash="dot", color="gray")))
        fig.add_trace(go.Scatter(x=calibration["time"], y=calibration[share_col],
                                  mode="lines", name=STRATEGY_LABELS[strategy], line=dict(color="#1A3E8C")))
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

# ---------- Daily Plan ----------
elif page == "Daily Plan":
    st.title("Daily Plan")
    calibration = st.session_state["calibration"]
    if calibration is None:
        st.warning("Run calibration first.")
        st.stop()

    if not result.forecast_only_dates:
        st.info("No forecast-only date found in the uploaded workbook. Upload a forecast to plan a new day.")
        st.stop()

    delivery_date = st.selectbox("Delivery date", result.forecast_only_dates,
                                  format_func=lambda d: d.strftime("%d %b %Y"))
    tomorrow = result.df[result.df["date"] == delivery_date]
    forecast = tomorrow[["block", "day_ahead_schedule"]].rename(columns={"day_ahead_schedule": "forecast_mw"})
    avc_by_block = dict(zip(tomorrow["block"], tomorrow["avc"]))

    strategy = st.radio("Strategy", ["conservative", "balanced", "aggressive"],
                         format_func=lambda s: STRATEGY_LABELS[s], horizontal=True, key="plan_strategy")

    try:
        out = build_daily_plan(forecast, calibration, strategy=strategy,
                                overrides=st.session_state["overrides"], avc_by_block=avc_by_block)
    except ForecastValidationError as e:
        st.error(str(e))
        st.stop()

    plan_key = (delivery_date, strategy)
    if plan_key not in st.session_state["_logged_plans"]:
        st.session_state["_logged_plans"].add(plan_key)
        log_run("Daily Plan", f"{delivery_date:%d %b %Y} - {STRATEGY_LABELS[strategy]}")

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
    fig.add_trace(go.Bar(x=plan["time"], y=plan["da_mw"], name="DA", marker_color="#1A3E8C"))
    fig.add_trace(go.Bar(x=plan["time"], y=plan["rtm_mw"], name="RTM", marker_color="#4A8FE8"))
    fig.update_layout(barmode="stack", height=350, margin=dict(t=10, l=10, r=10, b=10),
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Edit block overrides"):
        st.caption("Every block defaults to the calibrated share; overrides log the applied value.")
        edited = st.data_editor(
            plan[["block", "time", "forecast_mw", "da_share_applied"]],
            disabled=["block", "time", "forecast_mw"],
            hide_index=True, use_container_width=True, key="override_editor",
        )
        if st.button("Apply overrides"):
            base_share = calibration.set_index("block")[f"da_share_{strategy}"]
            new_overrides = {}
            for _, row in edited.iterrows():
                blk = int(row["block"])
                if abs(row["da_share_applied"] - base_share.loc[blk]) > 1e-9:
                    new_overrides[blk] = float(row["da_share_applied"])
            st.session_state["overrides"] = new_overrides
            st.rerun()

    with st.expander("Show block details"):
        st.dataframe(plan, use_container_width=True, hide_index=True)

    csv = plan.to_csv(index=False).encode()
    st.download_button("Export bid file (CSV)", csv,
                        f"bid_{delivery_date.strftime('%Y%m%d')}_{strategy}.csv", "text/csv")

# ---------- History ----------
elif page == "History":
    st.title("History")
    st.caption("Runs in this session.")
    log = st.session_state["run_log"]
    if not log:
        st.info("Nothing run yet this session - calibrate or generate a daily plan to see it logged here.")
    else:
        st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)
        st.caption(
            "This log lives only in your current browser session - it resets on page reload. "
            "Durable, cross-session history is a platform feature (the `Run` entity in the "
            "requirements spec) that needs a real backend, not yet wired up in this prototype."
        )
