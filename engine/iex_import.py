"""
Reads IEX "Market Snapshot" exports (15-minute blocks) into one clean frame per market.

Each export is one market (DAM, G-DAM or RTM) and at most about 30 days, so a long
history is a folder of files. The market is taken from the file name prefix
(DAM_..., GDAM_... / G-DAM_... / GREEN..., RTM_...). This module checks what it
reads instead of trusting it:

  * the date range printed in each file's header must match the dates actually inside;
  * every date must have exactly 96 blocks;
  * the same (market, date, block) seen in two files must carry identical values,
    otherwise the load stops. Identical repeats are dropped and reported, so a
    month exported twice under different names cannot double-count;
  * missing days inside the covered range are reported, never filled.

IEX data is licensed for personal, non-commercial use: keep the files local and
out of any public repository.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

NUM_COLUMNS = ["purchase_mw", "sell_mw", "mcv_mw", "fsv_mw", "mcp", "wind_sell_mw", "wind_mcv_mw"]
COLUMNS = ["date", "block"] + NUM_COLUMNS

# Each market's export has its own layout (G-DAM adds sell bids and volumes by source, RTM adds a
# session column), so columns are found by header text, never by position. First alias that
# matches wins. The last two are optional (wind bids exist only in the G-DAM export).
HEADER_ALIASES = {
    "purchase_mw": ["Purchase Bid (MW)"],
    "sell_mw": ["Sell Bid (MW)", "Total Sell Bid (MW)"],
    "mcv_mw": ["MCV (MW)", "Total MCV (MW)"],
    "fsv_mw": ["Final Scheduled Volume (MW)", "Total FSV (MW)"],
    "mcp": ["MCP (Rs/MWh)", "MCP (Rs/MWh) *"],
    "wind_sell_mw": ["Wind Sell Bid (MW)"],
    "wind_mcv_mw": ["Wind MCV (MW)"],
}
REQUIRED = ["purchase_mw", "sell_mw", "mcv_mw", "fsv_mw", "mcp"]
_BLOCK_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")
_RANGE_RE = re.compile(r"(\d{2}-\d{2}-\d{4})\s*to\s*(\d{2}-\d{2}-\d{4})", re.I)
_SINGLE_RE = re.compile(r"Date:\s*(\d{2}-\d{2}-\d{4})\s*$", re.I)


class IexFileError(Exception):
    """A snapshot file is malformed, or two files disagree."""


@dataclass
class IexLoad:
    market: str
    df: pd.DataFrame                      # one row per (date, block)
    files: list[str]
    duplicates: list[str] = field(default_factory=list)   # files that added nothing new
    missing_days: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def date_range(self):
        return (self.df["date"].min(), self.df["date"].max()) if len(self.df) else (None, None)


def market_from_name(path) -> str | None:
    n = Path(path).name.upper().replace("-", "").replace(" ", "")
    if n.startswith("GDAM") or n.startswith("GREEN"):
        return "GDAM"
    if n.startswith("RTM"):
        return "RTM"
    if n.startswith("DAM") or n.startswith("IDAM"):
        return "DAM"
    return None


def read_snapshot(path) -> tuple[pd.DataFrame, tuple | None]:
    """One export -> (frame, header date range or None)."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(wb.active.iter_rows(values_only=True))
    finally:
        wb.close()      # read-only workbooks keep the file locked on Windows until closed
    header_range = None
    for r in rows[:6]:
        text = " ".join(str(c) for c in r if c is not None)
        m = _RANGE_RE.search(text)
        if m:
            header_range = tuple(pd.to_datetime(m.group(i), format="%d-%m-%Y") for i in (1, 2))
            break
        m = _SINGLE_RE.search(text)
        if m:
            d = pd.to_datetime(m.group(1), format="%d-%m-%Y")
            header_range = (d, d)
            break
    hdr = next((i for i, r in enumerate(rows) if r and r[0] == "Date" and "Time Block" in r), None)
    if hdr is None:
        raise IexFileError(f"{Path(path).name}: no header row with 'Date' and 'Time Block' found.")
    head = [None if c is None else str(c).strip() for c in rows[hdr]]
    idx = {}
    for canon, aliases in HEADER_ALIASES.items():
        hit = next((head.index(a) for a in aliases if a in head), None)
        if hit is None and canon in REQUIRED:
            raise IexFileError(f"{Path(path).name}: required column missing (looked for {aliases}). "
                               f"Columns present: {[h for h in head if h]}")
        idx[canon] = hit
    i_block = head.index("Time Block")
    recs = []
    for r in rows[hdr + 1:]:
        m = _BLOCK_RE.match(str(r[i_block])) if len(r) > i_block and r[i_block] is not None else None
        if not m or r[0] is None:                       # summary rows carry no time block
            continue
        block = int(m.group(1)) * 4 + int(m.group(2)) // 15 + 1
        recs.append((pd.to_datetime(str(r[0]), format="%d-%m-%Y"), block,
                     *[(r[idx[c]] if idx[c] is not None else None) for c in NUM_COLUMNS]))
    df = pd.DataFrame(recs, columns=COLUMNS)
    for c in NUM_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, header_range


def _check_file(name: str, df: pd.DataFrame, header_range) -> list[str]:
    warn = []
    if df.empty:
        raise IexFileError(f"{name}: no data rows.")
    counts = df.groupby("date").size()
    bad = counts[counts != 96]
    if len(bad):
        raise IexFileError(f"{name}: dates without exactly 96 blocks: "
                           + ", ".join(f"{d:%d-%m-%Y} ({n})" for d, n in bad.items()))
    if df["mcp"].isna().any():
        raise IexFileError(f"{name}: {int(df['mcp'].isna().sum())} blocks have no MCP.")
    if header_range and (df["date"].min() != header_range[0] or df["date"].max() != header_range[1]):
        raise IexFileError(
            f"{name}: header says {header_range[0]:%d-%m-%Y} to {header_range[1]:%d-%m-%Y} but the data runs "
            f"{df['date'].min():%d-%m-%Y} to {df['date'].max():%d-%m-%Y}.")
    return warn


def load_folder(folder, market: str) -> IexLoad:
    """All snapshot files for one market in `folder` (file name must start with the market)."""
    market = market.upper().replace("-", "")
    paths = sorted(p for p in Path(folder).glob("*.xlsx")
                   if "SNAPSHOT" in p.name.upper() and not p.name.startswith("~$") and market_from_name(p) == market)
    if not paths:
        raise IexFileError(f"No {market} snapshot files found in {folder}.")
    return load_sources([(p.name, p) for p in paths], market)


def load_sources(sources, market: str) -> IexLoad:
    """`sources`: list of (file name, path or in-memory file object), all for one market."""
    market = market.upper().replace("-", "")
    seen: dict[tuple, tuple] = {}
    keep, dupes, files = [], [], []
    for name, src in sorted(sources, key=lambda s: s[0]):
        df, hdr = read_snapshot(src)
        _check_file(name, df, hdr)
        files.append(name)
        new_rows = []
        for row in df.itertuples(index=False):
            key = (row.date, row.block)
            vals = tuple(getattr(row, c) for c in NUM_COLUMNS)
            if key in seen:
                if not np.allclose(seen[key], vals, equal_nan=True, rtol=0, atol=1e-6):
                    raise IexFileError(f"{name}: {row.date:%d-%m-%Y} block {row.block} conflicts with an earlier file.")
                continue
            seen[key] = vals
            new_rows.append(row)
        if not new_rows:
            dupes.append(name)
        else:
            keep.append(pd.DataFrame(new_rows, columns=COLUMNS))
    out = pd.concat(keep, ignore_index=True).sort_values(["date", "block"]).reset_index(drop=True)
    days = pd.DatetimeIndex(sorted(out["date"].unique()))
    missing = list(pd.date_range(days.min(), days.max()).difference(days))
    warnings = []
    if dupes:
        warnings.append("Adds nothing new (same dates and values as another file): " + ", ".join(dupes))
    if missing:
        warnings.append(f"{len(missing)} day(s) missing inside {days.min():%d %b %Y} - {days.max():%d %b %Y}.")
    return IexLoad(market=market, df=out, files=files, duplicates=dupes, missing_days=missing, warnings=warnings)


def compare_with_workbook(iex: pd.DataFrame, workbook_df: pd.DataFrame, workbook_col: str) -> dict:
    """How closely do IEX prices match the workbook's own price column on the days both cover?
    `workbook_df` is the loader's frame; `workbook_col` is dam_mcp / gdam_mcp / rtm_mcp."""
    w = workbook_df[["date", "block", workbook_col]].dropna()
    m = w.merge(iex[["date", "block", "mcp"]], on=["date", "block"], how="inner")
    if m.empty:
        return {"overlap_block_days": 0}
    d = (m[workbook_col] - m["mcp"]).abs()
    return {"overlap_block_days": int(len(m)), "overlap_days": int(m["date"].nunique()),
            "exact_match_share": float((d < 0.5).mean()), "max_abs_diff": float(d.max())}
