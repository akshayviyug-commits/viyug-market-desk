"""HTML fragments for the Market Desk pages (kept apart from app.py so they can be tested)."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd


def lakh(x: float, sign: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    s = f"{abs(x) / 1e5:,.2f}"
    pre = ("+" if x >= 0 else "−") if sign else ("−" if x < 0 else "")
    return f"{pre}₹{s} L"


def lakh_l(x: float, sign: bool = False) -> str:
    """x already in lakh."""
    return lakh(x * 1e5, sign)


def card_title(left: str, right: str = "") -> str:
    return f'<div class="mk-card-title"><span>{left}</span><span class="r">{right}</span></div>'


def callout(body: str, kind: str = "") -> str:
    return f'<div class="mk-callout {kind}">{body}</div>'


def meta_strip(items: list[tuple[str, str]]) -> str:
    cells = "".join(f'<div><div class="k">{html.escape(k)}</div><div class="v">{v}</div></div>' for k, v in items)
    return f'<div class="mk-meta">{cells}</div>'


def review_table(tbl: pd.DataFrame) -> str:
    head = ("<tr><th>Day</th><th class='num'>Actually generated (MWh)</th><th class='num'>Promised (MWh)</th><th class='num'>Gap</th>"
            "<th class='num'>Day-ahead price<small>achieved, ₹/unit</small></th>"
            "<th class='num'>Real-time price<small>achieved, ₹/unit</small></th>"
            "<th class='num'>Deviation cost<small>DSM, ₹ lakh</small></th></tr>")
    rows = []
    for r in tbl.itertuples():
        g = r.gap_mwh
        gap = (f"<span class='pos'>+{g:,.0f} more</span>" if g >= 0.5 else f"<span class='neg'>−{abs(g):,.0f} short</span>" if g <= -0.5
               else "on schedule")
        rows.append(
            f"<tr class='{'hit' if r.evening_short else ''}'><td class='t'>{html.escape(r.label)}</td>"
            f"<td class='num'>{r.actual_mwh:,.0f}</td><td class='num'>{r.promised_mwh:,.0f}</td><td class='num'>{gap}</td>"
            f"<td class='num'>₹{r.da_price:.2f}</td><td class='num'>₹{r.rtm_price:.2f}</td>"
            f"<td class='num'>₹{r.dev_cost_inr / 1e5:.2f} L</td></tr>")
    return f"<table class='mk'>{head}{''.join(rows)}</table>"


def pattern_text(s: dict) -> str:
    n, k = s["n_days"], s["eve_short_days"]
    eve, rest = s["eve_err_pct"], s["rest_err_pct"]
    p = s["rtm_premium"]
    text = (f"<b>The pattern, measured not asserted.</b> On {k} of {n} evenings (18:00–22:00) the wind came in under what was promised. ")
    if eve >= rest:
        text += (f"Mean absolute deviation ran {eve:.1f}% of AvC in the evening blocks against {rest:.1f}% across the rest of the day. ")
    else:
        text += (f"Mean absolute deviation ran {rest:.1f}% of AvC outside the evening against {eve:.1f}% in it — "
                 f"on these days the misses were not an evening problem. ")
    if np.isfinite(p):
        if p >= 0:
            text += (f"Real-time cleared ₹{p:.2f}/unit above the day-ahead price actually achieved — so holding some volume back "
                     f"earns more and is penalised less.")
        else:
            text += (f"Real-time cleared ₹{abs(p):.2f}/unit below the day-ahead price actually achieved — so on these days "
                     f"real-time was the weaker market, and the case for holding volume back rests on deviation cost alone.")
    return callout(text)


def cluster_split_table(ct: pd.DataFrame, shade: str = "C-3") -> str:
    head = ("<tr><th>Time of day</th><th class='num'>Forecast</th><th class='num'>G-DAM filed</th><th class='num'>RTM planned</th></tr>")
    rows = []
    for _, r in ct.iterrows():
        rows.append(
            f"<tr class='{'hit' if r['cluster'] == shade else ''}'><td class='t'>{html.escape(str(r['Time of day']))}</td>"
            f"<td class='num'>{r['Forecast (MWh)']:,.0f} MWh</td>"
            f"<td class='num'>{r['G-DAM (MWh)']:,.0f} MWh &middot; {r['G-DAM %']:.0f}% G-DAM</td>"
            f"<td class='num'>{r['RTM (MWh)']:,.0f} MWh &middot; {r['RTM %']:.0f}% RTM</td></tr>")
    return f"<table class='mk'>{head}{''.join(rows)}</table>"


def strategy_table(summary: pd.DataFrame, order: list[str], selected: str, best: str, habit: str, all_g: str) -> str:
    head = ("<tr><th>Strategy</th><th class='num'>G-DAM<small>share</small></th><th class='num'>G-DAM leg<small>₹ L / day</small></th>"
            "<th class='num'>RTM leg<small>₹ L / day</small></th><th class='num'>Deviation cost<small>₹ L / day</small></th>"
            "<th class='num'>Charges<small>₹ L / day</small></th><th class='num'>Net<small>₹ L / day</small></th>"
            "<th class='num'>vs desk habit<small>₹ L / day</small></th><th class='num'>vs all G-DAM<small>₹ L / day</small></th>"
            "<th>Verdict</th></tr>")
    rows = []
    days = summary["Days"].iloc[0]
    for name in order:
        r = summary.loc[name]
        d_habit = r["Net (Rs lakh/day)"] - summary.loc[habit, "Net (Rs lakh/day)"]
        d_all = r["Net (Rs lakh/day)"] - summary.loc[all_g, "Net (Rs lakh/day)"]
        verdict = "<span class='mk-tag good'>highest net of the three</span>" if name == best else ""
        if name == selected:
            verdict = (verdict + " " if verdict else "") + "<span class='mk-tag'>selected</span>"
        cls = "sel" if name == selected else ""
        fmt = lambda v: f"<span class='{'pos' if v >= 0 else 'neg'}'>{'+' if v >= 0 else '−'}{abs(v):.2f}</span>"
        rows.append(
            f"<tr class='{cls}'><td class='t'>{html.escape(name)}</td><td class='num'>{r['Avg G-DAM share']:.0%}</td>"
            f"<td class='num'>{r['G-DAM leg (Rs lakh/day)']:.2f}</td><td class='num'>{r['RTM leg (Rs lakh/day)']:.2f}</td>"
            f"<td class='num'>{r['Avg DSM (Rs lakh/day)']:.2f}</td><td class='num'>{r['Charges (Rs lakh/day)']:.2f}</td>"
            f"<td class='num'>{r['Net (Rs lakh/day)']:.2f}</td><td class='num'>{fmt(d_habit)}</td><td class='num'>{fmt(d_all)}</td>"
            f"<td>{verdict}</td></tr>")
    return f"<table class='mk'>{head}{''.join(rows)}</table><div class='mk-note'>Averages over {days} replayed days.</div>"
