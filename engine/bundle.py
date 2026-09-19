"""
Everything the app needs from the uploaded workbook, built once:

  * the settlement workbook, validated by engine.loader (96 blocks a day, the three settlement identities),
  * the day x block panel the price model reads (prices, actual output, schedules, forecast, capacity),
  * the inputs for the Learn / guardrail engine.

The price model reads only this workbook: the days on file are the only history it learns from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .features import MARKET_FEATURES, Panel, Policy, build_panel
from .learn_engine import Day
from .learn_predict import build_days
from .loader import LoadResult, load_workbook


@dataclass
class DataBundle:
    workbook: LoadResult
    panel: Panel                        # what the price model reads
    days: dict[str, Day]                # inputs for the Learn / guardrail engine
    features: list[str] = field(default_factory=lambda: list(MARKET_FEATURES))
    policy: Policy = field(default_factory=Policy)
    notes: list[str] = field(default_factory=list)
    signature: str = ""


def build_bundle(workbook_source) -> DataBundle:
    """`workbook_source`: a path or an in-memory file (an upload is never written to disk)."""
    wb = load_workbook(workbook_source)
    panel = build_panel(wb.df, wb.settled_dates)
    days = build_days(wb.df, wb.settled_dates)
    sig = f"wb:{panel.dates[0]:%Y%m%d}-{panel.dates[-1]:%Y%m%d}-{len(panel.dates)}"
    return DataBundle(workbook=wb, panel=panel, days=days, signature=sig)
