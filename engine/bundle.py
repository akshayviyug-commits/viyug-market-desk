"""
Everything the app needs from the uploaded data, built once.

  * the settlement workbook (validated by engine.loader),
  * optionally the IEX market snapshots (DAM / G-DAM / RTM), each checked against the
    workbook wherever they overlap,
  * the calendar panel the price model reads: exchange + plant data when IEX history is
    present ("extended"), or the workbook alone ("workbook_only").

The price model always uses the market-level feature set, so the same code runs in both
modes and simply learns from more days when more history is supplied.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .features import MARKET_FEATURES, Panel, Policy, build_panel
from .features_market import build_market_panel
from .iex_import import IexFileError, IexLoad, compare_with_workbook, load_folder, load_sources, market_from_name
from .learn_engine import Day
from .learn_predict import build_days
from .loader import LoadResult, load_workbook

MARKETS = ("DAM", "GDAM", "RTM")
WORKBOOK_COL = {"DAM": "dam_mcp", "GDAM": "gdam_mcp", "RTM": "rtm_mcp"}


@dataclass
class DataBundle:
    workbook: LoadResult
    panel_workbook: Panel
    iex: dict[str, IexLoad]
    panel: Panel                        # what the price model reads
    days: dict[str, Day]                # inputs for the Learn / guardrail engine
    mode: str                           # "extended" | "workbook_only"
    features: list[str] = field(default_factory=lambda: list(MARKET_FEATURES))
    policy: Policy = field(default_factory=Policy)
    notes: list[str] = field(default_factory=list)
    checks: dict[str, dict] = field(default_factory=dict)      # IEX vs workbook, per market
    signature: str = ""

    @property
    def history_days(self) -> dict[str, int]:
        """Days of price history each market contributes to the model."""
        return {m: int(l.df["date"].nunique()) for m, l in self.iex.items()}


def build_bundle(workbook_source, *, iex_dir=None, iex_uploads: list | None = None) -> DataBundle:
    """`workbook_source`: path or in-memory file. IEX files come from a folder and/or from uploads
    (list of (file name, in-memory file)); the market is read from the start of each file name."""
    wb = load_workbook(workbook_source)
    pw = build_panel(wb.df, wb.settled_dates)
    days = build_days(wb.df, wb.settled_dates)

    iex: dict[str, IexLoad] = {}
    notes: list[str] = []
    if iex_dir and Path(iex_dir).is_dir():
        for m in MARKETS:
            try:
                iex[m] = load_folder(iex_dir, m)
            except IexFileError as e:
                if not str(e).startswith("No "):
                    raise
    if iex_uploads:
        by_market: dict[str, list] = {}
        for name, src in iex_uploads:
            m = market_from_name(name)
            if m is None:
                raise IexFileError(f"{name}: cannot tell the market. Start the file name with DAM_, GDAM_ or RTM_.")
            by_market.setdefault(m, []).append((name, src))
        for m, srcs in by_market.items():
            iex[m] = load_sources(srcs, m)

    checks = {}
    for m, l in iex.items():
        c = compare_with_workbook(l.df, wb.df, WORKBOOK_COL[m])
        checks[m] = c
        if c.get("overlap_block_days") and c["exact_match_share"] < 0.999:
            raise IexFileError(
                f"IEX {m} prices differ from the workbook's own {m} prices on {c['overlap_block_days']} overlapping "
                f"block-days (exact match {c['exact_match_share']:.1%}, largest gap Rs {c['max_abs_diff']:,.0f}). "
                "Check that the right market and dates were exported.")
        notes.extend(l.warnings)

    extended = "GDAM" in iex and "RTM" in iex
    if iex and not extended:
        notes.append("IEX files were found for " + ", ".join(iex) + " but the model needs both G-DAM and RTM "
                     "history; it is using the workbook alone.")
    panel = build_market_panel(iex, pw) if extended else pw
    sig = "|".join([f"wb:{pw.dates[0]:%Y%m%d}-{pw.dates[-1]:%Y%m%d}-{len(pw.dates)}"]
                   + [f"{m}:{l.df['date'].min():%Y%m%d}-{l.df['date'].max():%Y%m%d}-{len(l.df)}" for m, l in sorted(iex.items())])
    return DataBundle(workbook=wb, panel_workbook=pw, iex=iex, panel=panel, days=days,
                      mode="extended" if extended else "workbook_only", notes=notes, checks=checks, signature=sig)
