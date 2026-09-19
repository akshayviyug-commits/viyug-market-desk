"""
Viyug.AI Market Desk - Learn and Forecast & Split.

    venv\\Scripts\\python.exe -m streamlit run app.py -- [--workbook PATH] [--iex-dir DIR] [--plant NAME]

Upload the settlement workbook in the sidebar. Nothing is written to disk.
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import ui
from engine import learn_predict as lp
from engine.bundle import DataBundle, build_bundle
from engine.iex_import import IexFileError
from engine.loader import DataQualityError, SchemaError
from engine.planning import ForecastValidationError
from engine.price_forecast import Forecaster, coverage
from engine.review import CLUSTER_SLOT, cap_runs, price_matrix, review_days, spread_by_cluster, weekday_effect
from engine.split_backtest import ALL_G, HABIT, replay, summarise
from engine.split_price import CLUSTER_NAME, DIAL_LABEL, DIAL_TILT, cluster_table, price_split

ASSETS = Path(__file__).parent / "assets"
BLUE, NAVY, RED, AMBER, GREY = "#5B8BDF", "#0A2273", "#C0503A", "#B45309", "#8B95A7"
DIALS = list(DIAL_TILT)
FONT = dict(family="Inter, Segoe UI, sans-serif", size=12, color="#161B2E")

st.set_page_config(page_title="Viyug.AI - Market Desk", page_icon=str(ASSETS / "favicon.png"), layout="wide")


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", default="")
    ap.add_argument("--iex-dir", default="")
    ap.add_argument("--plant", default="")
    return ap.parse_known_args(sys.argv[1:])[0]


ARGS = cli()

for css in ("style.css", "desk.css"):
    p = ASSETS / css
    if p.exists():
        st.markdown(f"<style>{p.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def sidebar_brand():
    icon = ASSETS / "logo_icon.png"
    if not icon.exists():
        st.markdown("## Viyug.AI")
        return
    b64 = base64.b64encode(icon.read_bytes()).decode()
    st.markdown(
        f'<div class="viyug-sidebar-brand"><img src="data:image/png;base64,{b64}"/><div class="brand-text">'
        f'<div class="name">Viyug.AI</div><div class="tagline">VISION TO VALUE</div></div></div>'
        f'<div class="viyug-sidebar-app">Market Desk</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- context
class Ctx:
    """Everything derived from one data load, computed once and reused by both pages."""

    def __init__(self, bundle: DataBundle, plant: str = ""):
        self.b, self.plant = bundle, plant
        self.fc = Forecaster(bundle)
        self._replay, self._imp, self._walk = {}, {}, None

    @property
    def replay_window(self):
        s = [pd.Timestamp(d) for d in self.b.workbook.settled_dates]
        return s[min(7, len(s) - 1)], s[-1]

    def replay(self, rec: float, guard: bool) -> pd.DataFrame:
        if (rec, guard) not in self._replay:
            self._replay[(rec, guard)] = replay(self.b, self.fc, *self.replay_window, rec=rec, guardrail=guard)
        return self._replay[(rec, guard)]

    def walk(self) -> pd.DataFrame:
        if self._walk is None:
            self._walk = self.fc.walk(*self.replay_window)
        return self._walk

    def importance(self, T, market="g") -> pd.Series:
        key = (pd.Timestamp(T), market)
        if key not in self._imp:
            self._imp[key] = self.fc.importance(T, market)
        return self._imp[key]


@st.cache_resource(show_spinner="Preparing the price model on the days on file (a minute or two the first time)...", max_entries=3)
def make_ctx(_key: str, _bundle: DataBundle, plant: str, warm_date) -> Ctx:
    """Built and warmed once per data load. Streamlit holds a lock per key, so a second session opening the same
    data waits for this instead of repeating the work."""
    ctx = Ctx(_bundle, plant)
    ctx.fc.forecast(warm_date)
    ctx.walk()
    ctx.replay(300.0, True)
    return ctx


@st.cache_resource(show_spinner=False, max_entries=2)
def cached_bundle(wb_path: str, iex_dir) -> DataBundle:
    return build_bundle(wb_path, iex_dir=iex_dir)


def get_bundle(wb_up, use_sample: bool):
    """Build (or reuse) the data bundle for whatever is currently supplied. The optional exchange-price history
    comes only from a server-side folder (--iex-dir); users are never asked to upload it."""
    ss = st.session_state
    if wb_up is not None:
        wb_src, wb_id = io.BytesIO(wb_up.getvalue()), ("upload", wb_up.name, wb_up.size)
    elif use_sample:
        wb_src, wb_id = str(Path(__file__).parent / "sample_data" / "sample_workbook.xlsx"), ("sample",)
    elif ARGS.workbook and Path(ARGS.workbook).exists():
        wb_src, wb_id = ARGS.workbook, ("path", ARGS.workbook)
    else:
        return None, None
    iex_dir = ARGS.iex_dir if ARGS.iex_dir and Path(ARGS.iex_dir).is_dir() else None
    key = (wb_id, iex_dir)
    if ss.get("_data_key") != key:
        ss["_data_key"], ss["_data_err"], ss["bundle"] = key, None, None
        try:
            with st.spinner("Reading and checking the files..."):
                ss["bundle"] = build_bundle(wb_src, iex_dir=iex_dir) if wb_id[0] == "upload" else cached_bundle(wb_src, iex_dir)
        except (SchemaError, DataQualityError, IexFileError) as e:
            ss["_data_err"] = str(e)
    return ss.get("bundle"), key


# ----------------------------------------------------------------------------- chart helpers
def base_fig(h=320, **kw):
    fig = go.Figure()
    fig.update_layout(height=h, margin=dict(t=10, l=10, r=10, b=10), font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.12, x=0), **kw)
    fig.update_xaxes(showgrid=False, linecolor="#E3E8F0")
    fig.update_yaxes(gridcolor="#EEF1F6", zeroline=False)
    return fig


def times():
    return [f"{(b - 1) * 15 // 60:02d}:{(b - 1) * 15 % 60:02d}" for b in range(1, 97)]


def tick_kw():
    t = times()
    return dict(tickmode="array", tickvals=[t[i] for i in (0, 24, 40, 72, 88)], ticktext=["00:00", "06:00", "10:00", "18:00", "22:00"])


def card():
    return st.container(border=True)


NICE = {"g_l1": "G-DAM, same block yesterday", "g_nb": "G-DAM, neighbouring blocks yesterday", "g_m3": "G-DAM, 3-day average", "g_m7": "G-DAM, 7-day average",
        "g_all": "G-DAM, all-history average", "dm_l1": "DAM, same block yesterday", "dm_m3": "DAM, 3-day average", "r_l2": "RTM, same block 2 days ago",
        "r_nb": "RTM, neighbouring blocks", "r_m3": "RTM, 3-day average", "r_all": "RTM, all-history average", "sp_m3": "RTM minus G-DAM, 3-day average",
        "block": "Time of day (block)", "blk_sin": "Time of day (sine)", "blk_cos": "Time of day (cosine)", "window": "Time-of-day window", "dow": "Day of week",
        "is_weekend": "Weekend", "is_holiday": "Public holiday", "hol_prev": "Day after a holiday", "hol_next": "Day before a holiday",
        "dow_level_ratio": "Weekday price level", "g_day_l1": "G-DAM, yesterday's daily level", "g_trend": "G-DAM, 1-day vs 3-day trend",
        "g_cap_l1": "G-DAM at the cap yesterday", "g_cap_rate7": "G-DAM cap frequency, 7 days", "g_run_len": "High-price run length (G-DAM)",
        "g_run_pos": "Position within a high-price run", "g_maxrun": "Longest high-price run yesterday", "g_cap_run_len": "Cap run length yesterday",
        "r_day_l2": "RTM, daily level 2 days ago", "r_cap_l2": "RTM at the cap 2 days ago", "r_cap_rate7": "RTM cap frequency, 7 days",
        "r_run_len": "High-price run length (RTM)", "g_minus_dm_l1": "G-DAM minus DAM yesterday"}


# ----------------------------------------------------------------------------- header
def header(ctx: Ctx, T: pd.Timestamp):
    b = ctx.b
    wb = b.workbook
    avc = float(wb.df["avc"].max())
    plant = f"{ctx.plant} · " if ctx.plant else ""
    if b.mode == "extended":
        hist = " · ".join(f"{m.replace('GDAM', 'G-DAM')} {n}d" for m, n in sorted(b.history_days.items()))
        price_src = f"Exchange prices: {hist}"
    else:
        price_src = f"Workbook prices · {wb.n_settled_days} days"
    st.markdown('<div class="mk-eyebrow">Market desk · DAM / RTM day</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mk-title">MKT-{T:%Y}-{T:%m%d} — DAM/RTM optimisation, {plant}{avc:,.1f} MW wind</div>', unsafe_allow_html=True)
    st.markdown('<div class="mk-lede">A market pack built evidence first, then a split reasoned block by block. Every figure is computed from '
                'the desk\'s own workbook and the exchange\'s published prices; the price forecast is a small statistical model, and its track '
                'record is shown next to it.</div>', unsafe_allow_html=True)
    st.markdown(ui.meta_strip([
        ("Portfolio", f"{plant}{avc:,.1f} MW wind · merchant"),
        ("Objective", "max E[revenue] − E[DSM] · per 15-min block"),
        ("Data on file", f"{wb.n_settled_days} settled days · {wb.date_range[0]:%d %b} – {wb.date_range[1]:%d %b %Y}"),
        ("Price history", price_src),
        ("Status", "Morning cycle · model ready"),
    ]), unsafe_allow_html=True)


# ----------------------------------------------------------------------------- LEARN
def render_learn(ctx: Ctx, T: pd.Timestamp):
    b = ctx.b
    rv = review_days(b, T, n=5)
    tbl, stats, blocks = rv["table"], rv["stats"], rv["blocks"]
    if tbl.empty:
        st.warning("No settled days before this delivery date - nothing to learn from yet.")
        return

    with card():
        st.markdown(ui.card_title(f"a · What happened in the last {stats['n_days']} settled days?",
                                  f"{stats['first'].day}–{stats['last'].day} {stats['last']:%b} · read straight out of your settlement file"),
                    unsafe_allow_html=True)
        st.markdown(ui.review_table(tbl), unsafe_allow_html=True)
        st.markdown(ui.pattern_text(stats), unsafe_allow_html=True)

    with card():
        st.markdown(ui.card_title("b · Block-level evidence", f"promised against delivered, mean of the same {stats['n_days']} days · hover for detail"),
                    unsafe_allow_html=True)
        st.markdown('<div class="mk-sub">The same days at 15-minute resolution. Where the actual line sits below the scheduled one, the wind '
                    'under-delivered against the schedule; the shaded band is the free ±10% of AvC, inside which deviation costs nothing.</div>',
                    unsafe_allow_html=True)
        t = times()
        fig = base_fig(320)
        upper, lower = blocks["promised"] + blocks["band"], blocks["promised"] - blocks["band"]
        fig.add_trace(go.Scatter(x=t, y=upper, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=t, y=lower, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(244,228,200,.55)",
                                 name="Free band (±10% of AvC)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=t, y=blocks["promised"], mode="lines", name="Promised (final schedule, mean)", line=dict(color=AMBER, width=2)))
        fig.add_trace(go.Scatter(x=t, y=blocks["delivered"], mode="lines", name="Delivered (metered, mean)", line=dict(color=BLUE, width=2)))
        fig.update_yaxes(title="MW", rangemode="tozero")
        fig.update_xaxes(**tick_kw())
        st.plotly_chart(fig, width="stretch")
        wc = blocks["widest_cluster"]
        gaps = blocks["gap_by_cluster"]
        st.markdown(f'<div class="mk-note"><b>Reading:</b> the {CLUSTER_SLOT[wc]} blocks ({wc.lower()}) are where the two lines separate most '
                    f'(mean gap {gaps[wc]:.1f} MW, against {min(gaps.values()):.1f} MW in the quietest slot) — which is why the split is decided '
                    f'block by block rather than as one daily number.</div>', unsafe_allow_html=True)

        st.markdown('<div class="mk-h">Forecast error by time of day</div>', unsafe_allow_html=True)
        try:
            view = lp.learn_view(b.days, T.strftime("%Y-%m-%d"))
        except ForecastValidationError as e:
            st.error(str(e))
            view = None
        if view:
            prov = view["provenance"]
            st.markdown(f'<div class="mk-sub">What the desk\'s own history says: how wrong the day-ahead forecast usually is in each of the four '
                        f'slots. Learned from {len(prov["days_used"])} settled day(s) up to {pd.Timestamp(prov["cutoff"]):%d %b} '
                        f'(two days before delivery; the day before is still running at bid time). A negative median means the forecast usually over-promises.</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns([1.5, 1])
            ct = view["cluster_table"].copy()
            ct["Time window"] = [CLUSTER_NAME[c] for c in ("C-1", "C-2", "C-3", "C-4")]
            ct["Desk day-ahead commit"] = (ct["Desk day-ahead commit"] * 100).round(1)
            with c1:
                st.dataframe(ct[["Time window", "Desk day-ahead commit", "Forecast error P25 (MW)", "Forecast error P50 (MW)", "Forecast error P75 (MW)", "Samples", "Source"]]
                             .round(1), hide_index=True, width="stretch",
                             column_config={"Desk day-ahead commit": st.column_config.NumberColumn("Desk day-ahead commit (%)", format="%.1f")})
                if prov["fallback_error_clusters"]:
                    st.warning("Fewer than 20 observations for " + ", ".join(c.lower() for c in prov["fallback_error_clusters"]) +
                               " - a built-in starter sample fills in for those slots until real history accumulates.")
            with c2:
                box = base_fig(280)
                for c, errs in view["error_by_cluster"].items():
                    box.add_trace(go.Box(y=errs, name=CLUSTER_NAME[c], marker_color=BLUE, boxpoints=False, line=dict(color=NAVY)))
                box.add_hline(y=0, line_dash="dot", line_color="gray")
                box.update_layout(showlegend=False)
                box.update_yaxes(title="MW (actual − forecast)")
                st.plotly_chart(box, width="stretch")

    render_price_history(ctx, T)
    with st.expander("Data and method notes"):
        st.markdown(
            "- **Source:** the desk's own workbook (sheet *5. Detailed sheet*): generated = actual injected power after line loss; promised = final "
            "schedule; deviation cost = DSM settled from those two against AvC with the sheet's bands.\n"
            "- **Prices** in section a are volume-weighted averages actually achieved, with REC ₹300/MWh on DAC, DAM and RTM but not G-DAM.\n"
            "- **Deviation cost** here is the DSM bands only; the 5% SLDC surcharge the sheet reports separately is not added.\n"
            "- **Walk-forward:** everything used to decide a day stops at the day the desk could actually see: G-DAM/DAM prices to D-1, RTM prices, "
            "actuals and Learn's error profile to D-2.")


def render_price_history(ctx: Ctx, T: pd.Timestamp):
    b = ctx.b
    panel = b.panel
    with card():
        st.markdown(ui.card_title("c · What the price history says", "read from the exchange prices on file"), unsafe_allow_html=True)
        st.markdown('<div class="mk-sub">The price model learns from these patterns. Dark navy cells are blocks that cleared at the ₹10,000 exchange cap.</div>',
                    unsafe_allow_html=True)
        for market, label in (("gdam", "G-DAM"), ("rtm", "RTM")):
            d, m = price_matrix(panel, market, 30)
            fig = base_fig(230)
            fig.add_trace(go.Heatmap(z=m, x=times(), y=[f"{x.day} {x:%b}" for x in d], zmin=0, zmax=10000, xgap=0, ygap=1,
                                     colorscale=[[0, "#EAF2FE"], [0.55, BLUE], [0.99, "#1A3E8C"], [1, "#0A1E52"]],
                                     colorbar=dict(title="₹/MWh", thickness=10, len=.9),
                                     hovertemplate="%{y} %{x}<br>₹%{z:,.0f}/MWh<extra>" + label + "</extra>"))
            fig.update_xaxes(**tick_kw())
            fig.update_yaxes(autorange="reversed", type="category", tickmode="auto", nticks=6)
            st.markdown(f'<div class="mk-h">{label}, last {len(d)} days with prices</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        wk = weekday_effect(panel)
        with c1:
            st.markdown(f'<div class="mk-h">Day of week — {wk["market"]} daily average, {wk["n_days"]} days</div>', unsafe_allow_html=True)
            t = wk["table"]
            fig = base_fig(250)
            fig.add_trace(go.Bar(x=t["day"], y=t["mean"], marker_color=[BLUE] * 5 + ["#9DB9EC", NAVY], text=[f"n={n}" for n in t["n"]],
                                 textposition="outside", hovertemplate="%{x}: ₹%{y:,.0f}/MWh<extra></extra>"))
            fig.update_yaxes(title="₹/MWh", rangemode="tozero")
            st.plotly_chart(fig, width="stretch")
            sun, wkd = float(t.loc[6, "mean"]), wk["weekday_mean"]
            if np.isfinite(sun) and wkd > 0:
                st.markdown(f'<div class="mk-note">Sundays average {abs(sun / wkd - 1):.0%} {"below" if sun < wkd else "above"} weekdays. '
                            f'Each weekday has only {int(t["n"].min())}–{int(t["n"].max())} observations on file.</div>', unsafe_allow_html=True)
            if wk["holidays"]:
                items = "; ".join(f'{h["date"].day} {h["date"]:%b} {h["name"]} (₹{h["price"]:,.0f})' for h in wk["holidays"][:4])
                st.markdown(f'<div class="mk-note">Weekday holidays in the window vs the weekday average of ₹{wkd:,.0f}: {items}. '
                            f'Too few to call a holiday effect yet.</div>', unsafe_allow_html=True)
        with c2:
            cr = cap_runs(panel, "gdam")
            st.markdown(f'<div class="mk-h">Continuous high-price stretches — G-DAM at the cap</div>', unsafe_allow_html=True)
            fig = base_fig(250)
            fig.add_trace(go.Histogram(x=cr["runs"], xbins=dict(start=0.5, end=max(cr["longest"], 8) + 0.5, size=2), marker_color=NAVY,
                                       hovertemplate="%{x} blocks in a row: %{y} stretches<extra></extra>"))
            fig.update_xaxes(title="blocks in a row at the cap (4 blocks = 1 hour)")
            fig.update_yaxes(title="stretches")
            st.plotly_chart(fig, width="stretch")
            st.markdown(f'<div class="mk-note">G-DAM sat at the cap in {cr["share"]:.0%} of blocks over {cr["n_days"]} days. Once at the cap it stayed there '
                        f'a median of {cr["median"]:.0f} blocks ({cr["median"] / 4:.1f} h), the longest {cr["longest"]} blocks ({cr["longest"] / 4:.1f} h).</div>',
                        unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            sp = spread_by_cluster(panel, 300.0)
            st.markdown(f'<div class="mk-h">Where RTM beat G-DAM — by slot, {sp.attrs["n_days"]} days</div>', unsafe_allow_html=True)
            fig = base_fig(250)
            fig.add_trace(go.Bar(x=[f'{r.cluster.lower()} · {r.slot}' for r in sp.itertuples()], y=sp["mean_edge"],
                                 marker_color=[AMBER if v > 0 else BLUE for v in sp["mean_edge"]],
                                 text=[f"RTM ahead in {s:.0%}" for s in sp["share_rtm_better"]], textposition="outside",
                                 hovertemplate="%{x}<br>RTM + REC minus G-DAM: ₹%{y:,.0f}/MWh<extra></extra>"))
            fig.add_hline(y=0, line_color="gray")
            fig.update_yaxes(title="RTM + ₹300 REC minus G-DAM, ₹/MWh")
            st.plotly_chart(fig, width="stretch")
            st.markdown('<div class="mk-note">Blue: G-DAM paid more on average in that slot. Amber: RTM (with its REC) paid more. Where both markets sit at the '
                            'cap, the REC alone makes RTM count as ahead, so the share can exceed 50% even when the average favours G-DAM.</div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="mk-h">What the G-DAM price model leans on</div>', unsafe_allow_html=True)
            imp = ctx.importance(T, "g")
            if len(imp):
                top = imp.head(8)[::-1]
                fig = base_fig(250)
                fig.add_trace(go.Bar(x=top.values, y=[NICE.get(k, k) for k in top.index], orientation="h", marker_color=NAVY,
                                     hovertemplate="%{y}<extra></extra>"))
                fig.update_xaxes(title="accuracy lost if scrambled (₹/MWh)")
                fig.update_layout(margin=dict(t=10, l=200, r=10, b=10))
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Not enough history to fit the model yet - it is using yesterday's prices.")

        w = ctx.walk()
        st.markdown('<div class="mk-h">Track record — each day forecast using only what was known then</div>', unsafe_allow_html=True)
        if len(w):
            c = coverage(w)
            mg, pg = (w.g_p50 - w.y_gdam).abs().mean(), (w.g_anchor - w.y_gdam).abs().mean()
            mr, pr = (w.r_p50 - w.y_rtm).abs().mean(), (w.r_anchor - w.y_rtm).abs().mean()
            rows = [("Average error, model", f"₹{mg:,.0f}/MWh", f"₹{mr:,.0f}/MWh"),
                    ("Average error, yesterday's price", f"₹{pg:,.0f}/MWh", f"₹{pr:,.0f}/MWh"),
                    ("Model vs yesterday's price", f"{(1 - mg / pg):+.0%}", f"{(1 - mr / pr):+.0%}"),
                    ("Actual price inside the P10–P90 band (target 80%)", f"{c['g']:.0%}", f"{c['r']:.0%}")]
            body = "".join(f"<tr><td class='t'>{a}</td><td class='num'>{x}</td><td class='num'>{y}</td></tr>" for a, x, y in rows)
            st.markdown(f"<table class='mk'><tr><th></th><th class='num'>G-DAM</th><th class='num'>RTM</th></tr>{body}</table>", unsafe_allow_html=True)
            n_days = w["date"].nunique()
            st.markdown(f'<div class="mk-note">{n_days} forecast days ({w["date"].min():%d %b} – {w["date"].max():%d %b}). A short test: differences of a few percent '
                        f'are within noise, and the price forecast is best read as an edge on the desk\'s judgement, not a substitute for it.</div>',
                        unsafe_allow_html=True)
        months = len({(d.year, d.month) for d in panel.dates})
        if months < 2:
            st.markdown('<div class="mk-note"><b>Season:</b> the data covers a single month, so no seasonal effect can be learned yet.</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- FORECAST & SPLIT
def get_overrides(key):
    return st.session_state.setdefault("overrides", {}).get(key, {})


def set_overrides(key, value):
    st.session_state.setdefault("overrides", {})[key] = value


def render_split(ctx: Ctx, T: pd.Timestamp):
    b = ctx.b
    forecast_dates = {pd.Timestamp(d) for d in b.workbook.forecast_only_dates}
    with card():
        c1, c3, c4 = st.columns([2.6, 1.7, 1.9])
        with c1:
            dial = st.radio("Strategy", DIALS, index=1, format_func=lambda k: DIAL_LABEL[k], horizontal=True, key="dial")
        with c3:
            rec_on = st.checkbox("Count \u20b9300 REC on RTM", value=True, help="DAM, DAC and RTM earn \u20b9300/MWh REC; G-DAM does not.")
        with c4:
            guard = st.checkbox("Forecast-error guardrail", value=True,
                                help="Caps day-ahead volume at the level history's forecast error supports. Off: day-ahead is limited only by capacity.")
        st.markdown('<div class="mk-note">Conservative moves the share least on a price signal; Aggressive the most: each unit of price confidence '
                    'moves the day-ahead share by 10 / 20 / 30 points from the desk\'s usual commit ratio.</div>', unsafe_allow_html=True)
    rec = 300.0 if rec_on else 0.0
    ov_key = (T.strftime("%Y-%m-%d"), dial, rec, guard)

    with st.spinner("Forecasting the 96 blocks..."):
        prices = ctx.fc.forecast(T)
    try:
        out = price_split(b, T, prices, dial, rec, overrides=get_overrides(ov_key), guardrail=guard)
    except ForecastValidationError as e:
        st.error(str(e))
        return
    plan, s = out["plan"], out["summary"]
    for w in out["warnings"]:
        st.warning(w)
    info = prices.attrs["info"]

    # ---------------------------------------------------------------- a
    with card():
        st.markdown(ui.card_title("a · Tomorrow forecast and split, by time of day", "96 blocks grouped into 4 slots"), unsafe_allow_html=True)
        tag = "forecast" if T in forecast_dates else "replay of a settled day"
        st.markdown(
            f'<div class="mk-sub"><b>Forecast source:</b> the desk\'s own day-ahead forecast for {T.day} {T:%b} ({tag}) — a single point forecast, not a P10/P90 band. '
            f'<b>Total: {s["total_forecast_mwh"]:,.0f} MWh</b>, split <b>{s["gdam_share"]:.0%} G-DAM / {s["rtm_share"]:.0%} RTM</b> '
            f'(price forecast for both markets, {DIAL_LABEL[dial]} dial{"" if guard else ", guardrail off"}) — the two always add back to the forecast exactly, block by block.</div>',
            unsafe_allow_html=True)
        ct = cluster_table(plan)
        st.markdown(ui.cluster_split_table(ct, "C-3"), unsafe_allow_html=True)
        lo, hi = plan["da_share_applied"].min(), plan["da_share_applied"].max()
        st.markdown(f'<div class="mk-note">Shaded row: the evening slot, where the wind forecast misses most. Volume the plan holds for RTM: '
                    f'{s["blocks_price_rtm"]} blocks because the price favours RTM, {s["blocks_held_back"]} held back by the forecast-error guardrail.</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="mk-h">The split, block by block</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mk-sub">The table averages four slots; at block resolution the G-DAM share swings from {lo:.0%} to {hi:.0%}. '
                    f'The strip below each bar shows the price edge the model saw in that block.</div>', unsafe_allow_html=True)
        st.markdown('<div class="mk-legend"><span><i style="background:#5B8BDF"></i>G-DAM — filed day-ahead</span>'
                    '<span><i style="background:#0A2273"></i>RTM — held for real-time</span>'
                    '<span><i style="background:#C0803A"></i>Price edge: amber = RTM (with REC) expected to pay more</span></div>', unsafe_allow_html=True)
        t = times()
        fig = base_fig(330, barmode="stack")
        fig.add_trace(go.Bar(x=t, y=plan["gdam_mw"], name="G-DAM", marker_color=BLUE, hovertemplate="%{x}: %{y:.1f} MW G-DAM<extra></extra>"))
        fig.add_trace(go.Bar(x=t, y=plan["rtm_mw"], name="RTM", marker_color=NAVY, hovertemplate="%{x}: %{y:.1f} MW RTM<extra></extra>"))
        fig.update_layout(showlegend=False, bargap=0.05)
        fig.update_yaxes(title="MW", domain=[0.22, 1])
        fig.update_xaxes(**tick_kw())
        edge = plan["rtm_edge"].to_numpy()
        lim = max(1500.0, float(np.nanmax(np.abs(edge))))
        fig.add_trace(go.Heatmap(z=[edge], x=t, y=["price edge"], zmid=0, zmin=-lim, zmax=lim, showscale=False, yaxis="y2",
                                 colorscale=[[0, BLUE], [0.5, "#F6F3EC"], [1, "#C0803A"]],
                                 hovertemplate="%{x}: RTM + REC minus G-DAM = ₹%{z:,.0f}/MWh<extra></extra>"))
        fig.update_layout(yaxis2=dict(domain=[0, 0.13], showticklabels=False, anchor="x"))
        st.plotly_chart(fig, width="stretch")
        zero = plan.loc[plan["forecast_mw"] <= 0, "time"].tolist()
        if zero:
            st.markdown(f'<div class="mk-note">The workbook\'s forecast is zero for {", ".join(zero)} (likely a planned curtailment or outage), '
                        f'so nothing is scheduled in those blocks and the chart shows a gap.</div>', unsafe_allow_html=True)

        st.markdown('<div class="mk-h">What the price model expects</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mk-sub">Median forecast with a calibrated P10–P90 band for each market; RTM includes the REC so the two lines compare like for like. '
                    f'Trained on {info["train_days_g"]} days of G-DAM and {info["train_days_r"]} days of RTM prices known at bid time.</div>', unsafe_allow_html=True)
        fig = base_fig(300)
        for m, name, colour, fill, add in (("g", "G-DAM", BLUE, "rgba(91,139,223,.18)", 0.0), ("r", "RTM + REC", NAVY, "rgba(10,34,115,.12)", rec)):
            fig.add_trace(go.Scatter(x=t, y=plan[f"{m}_p90"] + add, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=t, y=plan[f"{m}_p10"] + add, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=fill, hoverinfo="skip",
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=t, y=plan[f"{m}_p50"] + add, mode="lines", name=f"{name} forecast", line=dict(color=colour, width=2)))
        actual = prices.set_index("block")
        if actual["y_gdam"].notna().any():
            fig.add_trace(go.Scatter(x=t, y=actual["y_gdam"].to_numpy(), mode="lines", name="G-DAM actual (published later)",
                                     line=dict(color=BLUE, width=1, dash="dot")))
            if actual["y_rtm"].notna().any():
                fig.add_trace(go.Scatter(x=t, y=actual["y_rtm"].to_numpy() + rec, mode="lines", name="RTM actual + REC",
                                         line=dict(color=NAVY, width=1, dash="dot")))
        fig.update_yaxes(title="₹/MWh", range=[0, 10600])
        fig.update_xaxes(**tick_kw())
        st.plotly_chart(fig, width="stretch")

    # ---------------------------------------------------------------- b
    rep = ctx.replay(rec, guard)
    summ = summarise(rep)
    label = DIAL_LABEL[dial]
    n_rep = int(summ["Days"].iloc[0])
    with card():
        st.markdown(ui.card_title("b · What the split is worth", f"replayed against the last {n_rep} settled days"), unsafe_allow_html=True)
        st.markdown(f'<div class="mk-sub">The {label} dial was run for each of the last {n_rep} settled days as it would have been that morning — the price model '
                    f'and Learn using only what was known then — and scored against the wind and prices that actually followed. '
                    f'Gain is measured against selling the whole forecast on G-DAM.</div>', unsafe_allow_html=True)
        piv = rep.pivot(index="date", columns="strategy", values="net_inr")
        gain = (piv[label] - piv[ALL_G]) / 1e5
        habit_gain = (piv[HABIT] - piv[ALL_G]) / 1e5
        fig = base_fig(300)
        x = [f"{d.day} {d:%b}" for d in piv.index]
        fig.add_trace(go.Bar(x=x, y=gain, name=f"{label}: gain vs all G-DAM", marker_color=[BLUE if v >= 0 else RED for v in gain],
                             hovertemplate="%{x}: %{y:+.2f} lakh<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=habit_gain, mode="lines+markers", name="Desk habit, no price view", line=dict(color=AMBER, width=1.5, dash="dot"),
                                 marker=dict(size=5)))
        fig.update_yaxes(title="₹ lakh per day")
        fig.add_hline(y=0, line_color="gray")
        st.plotly_chart(fig, width="stretch")

        net = piv[label] / 1e5
        dsm = rep[rep["strategy"] == label]["dsm_inr"].mean()
        q = lambda s_, p: float(np.quantile(s_, p))
        head = f"<tr><th>If {T.day} {T:%b} pays like…</th><th class='num'>P10</th><th class='num'>P50</th><th class='num'>P90</th></tr>"
        r1 = f"<tr><td class='t'>Net revenue, per day</td>" + "".join(f"<td class='num'>{ui.lakh_l(q(net, p))}</td>" for p in (.1, .5, .9)) + "</tr>"
        r2 = f"<tr><td class='t'>Gain vs all G-DAM, per day</td>" + "".join(
            f"<td class='num'>{ui.lakh_l(q(gain, p), True)}</td>" for p in (.1, .5, .9)) + "</tr>"
        st.markdown(f"<table class='mk'>{head}{r1}{r2}</table>", unsafe_allow_html=True)
        tot = float(gain.sum())
        st.markdown(ui.callout(
            f"<b>Expected deviation cost {ui.lakh(dsm)} per day</b> — measured against a free ±10% band around AvC. It is the same for every split, "
            f"because both legs are sized from the same forecast. Over {n_rep} days the {label} dial "
            f"{'earned' if tot >= 0 else 'gave up'} {ui.lakh_l(abs(tot))} {'more' if tot >= 0 else 'less'} than selling everything on G-DAM, and better than the desk's habit by "
            f"{ui.lakh_l(float((gain - habit_gain).sum()), True)}. With this few days a gap of that size is within noise.", "blue"), unsafe_allow_html=True)
        if guard:
            st.markdown('<div class="mk-note">The replay cannot credit the forecast-error guardrail for any deviation it might save through RTM revisions, because '
                        'revisions are not modelled - it shows only the guardrail\'s cost in price terms. Switch it off above to see the difference.</div>',
                        unsafe_allow_html=True)

    # ---------------------------------------------------------------- c
    with card():
        st.markdown(ui.card_title("c · Three strategies compared", "which split earns most after penalties?"), unsafe_allow_html=True)
        st.markdown(f'<div class="mk-sub">Same forecast, same prices, same replay days; only the day-ahead / RTM boundary moves. Day-ahead plus RTM adds back to the '
                    f'forecast in all three. Bars are the average per day over {n_rep} replays.</div>', unsafe_allow_html=True)
        names = [DIAL_LABEL[d] for d in DIALS]
        fig = base_fig(300, barmode="stack")
        show = names + [HABIT, ALL_G]
        fig.add_trace(go.Bar(x=show, y=[summ.loc[n, "G-DAM leg (Rs lakh/day)"] for n in show], name="G-DAM leg revenue", marker_color=BLUE))
        fig.add_trace(go.Bar(x=show, y=[summ.loc[n, "RTM leg (Rs lakh/day)"] for n in show], name="RTM leg revenue (incl. REC)", marker_color=NAVY))
        fig.add_trace(go.Bar(x=show, y=[summ.loc[n, "Avg DSM (Rs lakh/day)"] + summ.loc[n, "Charges (Rs lakh/day)"] for n in show],
                             name="Deviation cost + charges", marker_color=RED))
        fig.update_yaxes(title="₹ lakh per day")
        st.plotly_chart(fig, width="stretch")
        best = max(names, key=lambda n: summ.loc[n, "Net (Rs lakh/day)"])
        st.markdown(ui.strategy_table(summ, names + [HABIT, ALL_G], label, best, HABIT, ALL_G), unsafe_allow_html=True)
        d_h = summ.loc[best, "Net (Rs lakh/day)"] - summ.loc[HABIT, "Net (Rs lakh/day)"]
        d_a = summ.loc[best, "Net (Rs lakh/day)"] - summ.loc[ALL_G, "Net (Rs lakh/day)"]
        st.markdown(ui.callout(
            f"<b>Why {best} leads:</b> it moves furthest on the price signal. It earned {ui.lakh_l(d_h, True)} per day against the desk's habit of "
            f"about {summ.loc[HABIT, 'Avg G-DAM share']:.0%} day-ahead with no price view, and {ui.lakh_l(d_a, True)} per day against selling everything on G-DAM. "
            f"<b>None of the three is proven better than simply staying on G-DAM</b> — in this data G-DAM out-paid RTM in most blocks, and {n_rep} days is a short "
            f"test. What the price view does show is a consistent improvement over a fixed habit.", "amber"), unsafe_allow_html=True)

    # ---------------------------------------------------------------- d
    with card():
        st.markdown(ui.card_title(f"d · Final split: {s['gdam_share']:.0%} G-DAM / {s['rtm_share']:.0%} RTM", "the recommendation"), unsafe_allow_html=True)
        st.markdown(ui.callout(
            f"<b>Expected net (P50, replay of the last {n_rep} days): {ui.lakh_l(float(np.median(net)))} per day.</b> Expected deviation cost {ui.lakh(dsm)}. "
            f"Indicative value of this plan at the model's own median prices: {ui.lakh(s['indicative_revenue_inr'])} "
            f"({ui.lakh(s['indicative_gain_vs_all_gdam_inr'], True)} vs all G-DAM). {s['indicative_note']}", "blue"), unsafe_allow_html=True)
        with st.expander("Edit block overrides"):
            st.caption("Each block defaults to the plan's own G-DAM share. An override replaces it for that block; RTM takes the rest, so day-ahead + RTM still equals the forecast.")
            edited = st.data_editor(plan[["block", "time", "forecast_mw", "da_share_applied"]].round(3), disabled=["block", "time", "forecast_mw"],
                                    hide_index=True, width="stretch", key="ovr_" + "_".join(map(str, ov_key)),
                                    column_config={"da_share_applied": st.column_config.NumberColumn("G-DAM share", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")})
            if st.button("Apply overrides"):
                base = price_split(b, T, prices, dial, rec, guardrail=guard)["plan"].set_index("block")["da_share_applied"]
                set_overrides(ov_key, {int(r.block): float(r.da_share_applied) for r in edited.itertuples() if abs(r.da_share_applied - base.loc[int(r.block)]) > 1e-3})
                st.rerun()
        with st.expander("Show block details"):
            cols = ["block", "time", "cluster", "forecast_mw", "g_p50", "r_p50", "rtm_edge", "confidence_z", "habit_da_share", "da_share_applied", "gdam_mw", "rtm_mw", "reason"]
            st.dataframe(plan[cols].round(3), hide_index=True, width="stretch")
        bid = plan[["block", "time", "forecast_mw", "gdam_mw", "rtm_mw", "da_share_applied", "override_flag"]].round(4).to_csv(index=False).encode()
        st.download_button("Export bid file (CSV)", bid, f"bid_{T:%Y%m%d}_{dial}.csv", "text/csv")

    with st.expander("How this works, and what it cannot tell you"):
        st.markdown(
            f"- **Price view:** for each block, expected RTM price (plus REC) minus expected G-DAM price, divided by the uncertainty of both. Each unit of "
            f"confidence moves the day-ahead share {DIAL_TILT[dial] * 100:.0f} points from the desk's usual commit ratio.\n"
            "- **Guardrail:** day-ahead volume never exceeds what history's forecast error supports (Learn's profile), so an unreliable forecast is not over-committed.\n"
            "- **DSM is not a lever:** both legs are sized from the same forecast, so deviation cost is the same whatever the split. It is a cost in the scoring.\n"
            "- **Not modelled:** RTM revisions, the bilateral DAC leg the desk arranges after 13:00 on D-1, and volume rationing when prices sit at the cap.\n"
            f"- **Data:** price model trained on {info['train_days_g']} days of G-DAM and {info['train_days_r']} days of RTM; bands calibrated on the last "
            f"{info['calib_days_used']} days of its own errors.")


# ----------------------------------------------------------------------------- main
with st.sidebar:
    sidebar_brand()
    st.markdown("#### Data")
    wb_up = st.file_uploader("Settlement workbook (.xlsx)", type=["xlsx"], key="wb_up")
    use_sample = st.session_state.get("use_sample", False)
    sample_path = Path(__file__).parent / "sample_data" / "sample_workbook.xlsx"
    if wb_up is None and not ARGS.workbook and sample_path.exists():
        if st.button("Load sample workbook"):
            st.session_state["use_sample"] = use_sample = True

bundle, key = get_bundle(wb_up, use_sample)
if st.session_state.get("_data_err"):
    st.error(st.session_state["_data_err"])
if bundle is None:
    st.markdown('<div class="mk-eyebrow">Market desk · DAM / RTM day</div><div class="mk-title">Market Desk</div>', unsafe_allow_html=True)
    st.info("Upload the settlement workbook in the sidebar, or load the sample workbook, to begin.")
    st.stop()

all_dates = sorted(pd.Timestamp(d) for d in bundle.workbook.df["date"].unique())
fdates = {pd.Timestamp(d) for d in bundle.workbook.forecast_only_dates}
default_T = max(fdates) if fdates else all_dates[-1]

ctx = make_ctx(repr(key), bundle, ARGS.plant, default_T)
with st.sidebar:
    for n in bundle.notes:
        st.warning(n)
    if bundle.mode == "extended":
        st.success("Price history: " + ", ".join(f"{m.replace('GDAM', 'G-DAM')} {n} days" for m, n in sorted(bundle.history_days.items())))
        st.caption("Exchange prices match the workbook's own price columns exactly on the days both cover.")
    else:
        st.info("The price model learns from the workbook's own days.")


with st.container(key="rail"):
    page = st.radio("Stage", ["1 LEARN  ·  reviews settled days", "2 FORECAST & SPLIT  ·  solves 96 blocks"], horizontal=True,
                    label_visibility="collapsed", key="stage")

col_date, _ = st.columns([1.2, 3])
with col_date:
    saved = st.session_state.get("delivery_date")
    idx = all_dates.index(saved) if saved in all_dates else all_dates.index(default_T)
    T = st.selectbox("Delivery date", all_dates, index=idx, key="dd",
                     format_func=lambda d: f"{d:%d %b %Y}" + ("  -  forecast" if d in fdates else "  -  replay of a settled day"))
    st.session_state["delivery_date"] = T

header(ctx, T)
if page.startswith("1"):
    render_learn(ctx, T)
else:
    render_split(ctx, T)
