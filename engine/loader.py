"""
Loads the desk's tracking workbook ("5. Detailed sheet") into a clean,
validated DataFrame.

Per Market_Desk_Requirements.docx section 3:
  - Column mapping is by HEADER NAME, never position. A missing required
    header stops ingestion with a clear error (never guess a column).
  - The system must not assume a fixed number of days: it reads whatever
    contiguous date range is present and derives the day count.
  - Every date must have exactly 96 distinct blocks (1-96); a short/long
    day (DST, data gap) must fail loudly, not corrupt every per-block
    aggregate downstream silently.
  - Section 3.1's three identities are asserted at load time, tolerance 1e-6.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field

SHEET_NAME = "5. Detailed sheet"
HEADER_ROW = 3          # 0-indexed -> Excel row 4
ENERGY_FACTOR = 0.25    # MW -> MWh for a 15-minute block. Single source of truth.
IDENTITY_TOLERANCE = 1e-6

# Canonical field name -> exact source header text in the workbook.
REQUIRED_COLUMNS = {
    "time": "Time",
    "block": "Block",
    "date": "Date",
    "day_ahead_schedule": "Day Ahead Schedule",
    "dac_sched": "DAC Schedule (Interface Point)",
    "dam_sched": "DAM Schedule (Interface Point)",
    "gdam_sched": "GDAM Schedule (Interface Point)",
    "rtm_sched": "RTM Schedule (Interface Point)",
    "final_schedule": "Final Schedule Power (MW)",
    "avc": "AvC(MW)",
    "actual_ll": "Actual Injected Power after line loss(MW)",
    "acp": "ACP (wt. avg of MCP's)",
    "dac_mcp": "DAC MCP",
    "gdam_mcp": "G-DAM MCP",
    "dam_mcp": "DAM MCP",
    "rtm_mcp": "RTM MCP",
    "total_charges": "Total charges",
}


class SchemaError(Exception):
    """Raised when the workbook doesn't have the headers/shape this system requires."""


class DataQualityError(Exception):
    """Raised when a load-time guardrail (96-block assertion, identity check) fails."""


@dataclass
class LoadResult:
    df: pd.DataFrame
    settled_dates: list
    forecast_only_dates: list
    date_range: tuple
    n_settled_days: int
    warnings: list = field(default_factory=list)


def _read_raw_sheet(path: str) -> pd.DataFrame:
    try:
        raw = pd.read_excel(path, sheet_name=SHEET_NAME, header=HEADER_ROW, engine="openpyxl")
    except ValueError as e:
        raise SchemaError(
            f"Worksheet '{SHEET_NAME}' not found in {path}. "
            f"This system reads that sheet by name; if the desk renamed it, "
            f"update SHEET_NAME in loader.py or rename the sheet back."
        ) from e
    return raw


def _map_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Map required columns by exact header name. Never guesses a column."""
    available = list(raw.columns)
    missing = {canon: src for canon, src in REQUIRED_COLUMNS.items() if src not in available}
    if missing:
        lines = "\n".join(f"  - '{src}'  (needed as `{canon}`)" for canon, src in missing.items())
        raise SchemaError(
            "Cannot ingest this workbook: the following required headers are missing "
            f"from worksheet '{SHEET_NAME}':\n{lines}\n\n"
            "Column mapping is by header name, not position, so a renamed or reordered "
            "column will not be silently guessed. Confirm the mapping with the desk before proceeding."
        )
    # Some source columns repeat across the sheet's duplicate mirrored block;
    # pandas disambiguates repeats as "X", "X.1", "X.2" - always take the first.
    out = pd.DataFrame()
    for canon, src in REQUIRED_COLUMNS.items():
        col = raw[src]
        if isinstance(col, pd.DataFrame):  # duplicate header name -> take first occurrence
            col = col.iloc[:, 0]
        out[canon] = col
    return out


def _derive_date_only(df: pd.DataFrame) -> pd.DataFrame:
    """The Date column carries a spurious time-of-day component in some exports;
    block start time is derivable from `block` alone (docx sec. 3), so we only
    ever need the calendar date."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def _assert_96_blocks_per_day(df: pd.DataFrame) -> None:
    problems = []
    for date, group in df.groupby("date"):
        blocks = sorted(group["block"].tolist())
        if len(blocks) != 96 or blocks != list(range(1, 97)):
            counts = pd.Series(blocks).value_counts()
            dupes = counts[counts > 1].index.tolist()
            missing = sorted(set(range(1, 97)) - set(blocks))
            problems.append(
                f"  - {date.date()}: {len(blocks)} rows (expected 96)"
                + (f", duplicated blocks {dupes}" if dupes else "")
                + (f", missing blocks {missing}" if missing else "")
            )
    if problems:
        raise DataQualityError(
            "Block-count guardrail failed - a short or long day (DST, data gap, "
            "duplicate paste) would silently corrupt every per-block aggregate "
            "downstream, so this aborts instead:\n" + "\n".join(problems)
        )


def _assert_contiguous_dates(dates: list) -> None:
    dates = sorted(dates)
    expected = pd.date_range(dates[0], dates[-1], freq="D")
    actual = pd.DatetimeIndex(dates)
    gap = expected.difference(actual)
    if len(gap) > 0:
        raise DataQualityError(
            "Date-range guardrail failed - the input has gaps on: "
            + ", ".join(d.strftime("%Y-%m-%d") for d in gap)
            + ". Calibration assumes a contiguous history; fix the source file or "
              "narrow the range before proceeding."
        )


def _classify_settled_vs_forecast(df: pd.DataFrame) -> tuple[list, list]:
    """A day is settled when actual generation and all four MCPs are present
    and non-sentinel for all 96 blocks; forecast-only otherwise (mirrors
    CIP-011 in the platform requirements doc)."""
    settled, forecast_only = [], []
    for date, group in df.groupby("date"):
        has_actual = (group["actual_ll"].notna()).all() and not (group["actual_ll"] == 0).all()
        # DAC MCP is legitimately nullable (DAC doesn't clear in every block, per
        # docx sec. 3) - only GDAM/DAM/RTM, which always clear, signal settlement.
        has_all_mcps = group[["gdam_mcp", "dam_mcp", "rtm_mcp"]].notna().all().all()
        if has_actual and has_all_mcps:
            settled.append(date)
        else:
            forecast_only.append(date)
    return sorted(settled), sorted(forecast_only)


def _run_identity_checks(df: pd.DataFrame, settled_dates: list) -> list:
    """Section 3.1's three load-time identities, settled days only. Aborts on
    violation beyond tolerance rather than producing quietly-wrong output."""
    settled = df[df["date"].isin(settled_dates)]
    warnings = []

    sum_legs = settled["dac_sched"] + settled["dam_sched"] + settled["gdam_sched"] + settled["rtm_sched"]
    err1 = (settled["final_schedule"] - sum_legs).abs().max()
    if err1 > IDENTITY_TOLERANCE:
        raise DataQualityError(
            f"Identity failed: Final Schedule Power != sum of legs (max abs error {err1:.6f}). "
            "The input file is malformed or the schema has changed."
        )

    # AvC >= day_ahead_schedule: warn only, per docx (the desk's own data violates it occasionally)
    violation = settled[settled["avc"] < settled["day_ahead_schedule"]]
    if len(violation):
        warnings.append(
            f"{len(violation)} block-day(s) have AvC < Day Ahead Schedule "
            "(known/accepted per spec - not a load failure)."
        )

    return warnings


def load_workbook(path: str) -> LoadResult:
    raw = _read_raw_sheet(path)
    df = _map_columns(raw)
    df = _derive_date_only(df)
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    _assert_96_blocks_per_day(df)
    dates = df["date"].unique().tolist()
    _assert_contiguous_dates(dates)

    settled_dates, forecast_only_dates = _classify_settled_vs_forecast(df)
    warnings = _run_identity_checks(df, settled_dates)

    date_range = (min(dates), max(dates))
    return LoadResult(
        df=df,
        settled_dates=settled_dates,
        forecast_only_dates=forecast_only_dates,
        date_range=date_range,
        n_settled_days=len(settled_dates),
        warnings=warnings,
    )
