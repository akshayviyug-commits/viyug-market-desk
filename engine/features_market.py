"""
Market panel built from IEX snapshots, plus the bid-stack features.

The IEX exports carry, for every market and block, how much was bid to buy and
sell as well as the clearing price. The workbook carries the plant's own forecast,
actuals and settlement, but only for its own days. This module lines them up on one
calendar: exchange data wherever the exchange published it, plant data wherever the
workbook has it, NaN elsewhere.

Bid-stack features follow the same information rule as prices: G-DAM and DAM
bid stacks are known up to D-1 (cleared on D-2); RTM up to D-2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import N_BLOCKS, Panel, Policy, _day_row, _nanmean, _window
from .iex_import import IexLoad

BID_FEATURES = ["gd_bs_l1", "gd_bs_m3", "dam_bs_l1", "dam_bs_m3", "rtm_bs_l2", "gd_wind_share_l1"]


def _mat(df: pd.DataFrame, col: str, dates: pd.DatetimeIndex) -> np.ndarray:
    p = df.pivot(index="date", columns="block", values=col)
    return p.reindex(index=dates, columns=range(1, N_BLOCKS + 1)).to_numpy(dtype=float, copy=True)


def build_market_panel(iex: dict[str, IexLoad], workbook: Panel) -> Panel:
    """Daily range covering every IEX file and the workbook; prices and bid stacks from the
    exchange, plant fields from the workbook."""
    starts = [l.df["date"].min() for l in iex.values()] + [workbook.dates.min()]
    ends = [l.df["date"].max() for l in iex.values()] + [workbook.dates.max()]
    dates = pd.date_range(min(starts), max(ends))
    pos = {d: i for i, d in enumerate(dates)}
    n = len(dates)

    def from_workbook(arr):
        out = np.full((n, N_BLOCKS), np.nan)
        for i, d in enumerate(workbook.dates):
            out[pos[d]] = arr[i]
        return out

    def price(market, wb_arr):
        out = _mat(iex[market].df, "mcp", dates) if market in iex else np.full((n, N_BLOCKS), np.nan)
        fill = from_workbook(wb_arr)
        return np.where(np.isnan(out), fill, out)          # workbook only fills what IEX did not cover

    settled = np.zeros(n, dtype=bool)
    for i, d in enumerate(workbook.dates):
        settled[pos[d]] = workbook.settled[i]

    extra = {}
    for market, key in (("DAM", "dam"), ("GDAM", "gd"), ("RTM", "rtm")):
        if market in iex:
            extra[f"{key}_buy"] = _mat(iex[market].df, "purchase_mw", dates)
            extra[f"{key}_sell"] = _mat(iex[market].df, "sell_mw", dates)
    if "GDAM" in iex:
        extra["gd_wind_sell"] = _mat(iex["GDAM"].df, "wind_sell_mw", dates)

    return Panel(
        dates=dates, gdam=price("GDAM", workbook.gdam), dam=price("DAM", workbook.dam), rtm=price("RTM", workbook.rtm),
        acp=from_workbook(workbook.acp), actual=from_workbook(workbook.actual), final=from_workbook(workbook.final),
        fc=from_workbook(workbook.fc), avc=from_workbook(workbook.avc), settled=settled, extra=extra,
    )


def _ratio(panel: Panel, buy: str, sell: str, day) -> np.ndarray:
    if buy not in panel.extra or sell not in panel.extra:
        return np.full(N_BLOCKS, np.nan)
    b, s = _day_row(panel, panel.extra[buy], day), _day_row(panel, panel.extra[sell], day)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = b / s
    return np.where(np.isfinite(r), r, np.nan)


def _ratio_mean(panel: Panel, buy: str, sell: str, cutoff, n_days: int) -> np.ndarray:
    if buy not in panel.extra or sell not in panel.extra:
        return np.full(N_BLOCKS, np.nan)
    b, s = _window(panel, panel.extra[buy], cutoff, n_days), _window(panel, panel.extra[sell], cutoff, n_days)
    if not len(b):
        return np.full(N_BLOCKS, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = b / s
    return _nanmean(np.where(np.isfinite(r), r, np.nan))


def bidstack_features(panel: Panel, target, policy: Policy = Policy()) -> pd.DataFrame:
    """Buyers-to-sellers ratios (how oversubscribed each market was) at the same block on the
    last day the desk can see, under the same cutoffs as the price features."""
    target = pd.Timestamp(target).normalize()
    cg = target - pd.Timedelta(days=policy.lag_gdam)
    cr = target - pd.Timedelta(days=policy.lag_rtm)
    f = pd.DataFrame(index=range(N_BLOCKS))
    f["gd_bs_l1"] = _ratio(panel, "gd_buy", "gd_sell", cg)
    f["gd_bs_m3"] = _ratio_mean(panel, "gd_buy", "gd_sell", cg, 3)
    f["dam_bs_l1"] = _ratio(panel, "dam_buy", "dam_sell", cg)
    f["dam_bs_m3"] = _ratio_mean(panel, "dam_buy", "dam_sell", cg, 3)
    f["rtm_bs_l2"] = _ratio(panel, "rtm_buy", "rtm_sell", cr)
    if "gd_wind_sell" in panel.extra and "gd_sell" in panel.extra:
        w, s = _day_row(panel, panel.extra["gd_wind_sell"], cg), _day_row(panel, panel.extra["gd_sell"], cg)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = w / s
        f["gd_wind_share_l1"] = np.where(np.isfinite(share), share, np.nan)
    else:
        f["gd_wind_share_l1"] = np.nan
    return f[BID_FEATURES]
