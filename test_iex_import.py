"""Tests for the IEX snapshot importer, using small synthetic files (no real IEX data needed)."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
import openpyxl
import pandas as pd

from engine.iex_import import IexFileError, load_folder, market_from_name, read_snapshot


def write_snapshot(path, start, days, header_range=None, price=5000.0, drop_block=None):
    wb = openpyxl.Workbook()
    ws = wb.active
    end = pd.Timestamp(start) + pd.Timedelta(days=days - 1)
    hr = header_range or (pd.Timestamp(start), end)
    ws.append([])
    ws.append(["Market Snapshot"])
    ws.append([f"Date: {hr[0]:%d-%m-%Y} to {hr[1]:%d-%m-%Y}"])
    ws.append([])
    ws.append(["Date", "Hour", "Time Block", "Purchase Bid (MW)", "Sell Bid (MW)", "MCV (MW)",
               "Final Scheduled Volume (MW)", "MCP (Rs/MWh) *"])
    for d in range(days):
        date = pd.Timestamp(start) + pd.Timedelta(days=d)
        for b in range(96):
            if drop_block == (d, b):
                continue
            h, m = divmod(b * 15, 60)
            e = (b + 1) * 15
            ws.append([f"{date:%d-%m-%Y}", h + 1, f"{h:02d}:{m:02d} - {e // 60:02d}:{e % 60:02d}",
                       20000.0, 5000.0, 4000.0, 4000.0, price + b])
    ws.append(["Date", "Summary", "Purchase Bid", "Sell Bid", "MCV", "Final Scheduled Volume", "MCP"])
    ws.append([f"{pd.Timestamp(start):%d-%m-%Y}", "Total (MWh)", 1, 2, 3, 4, None])
    wb.save(path)


def test_market_names():
    assert market_from_name("DAM_Market Snapshot-May.xlsx") == "DAM"
    assert market_from_name("GDAM_Market Snapshot.xlsx") == "GDAM"
    assert market_from_name("G-DAM_Market Snapshot.xlsx") == "GDAM"
    assert market_from_name("RTM_Market Snapshot.xlsx") == "RTM"
    assert market_from_name("something.xlsx") is None
    print("PASS: market is read from the file-name prefix")


def test_reads_blocks_and_ignores_summary():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "DAM_Market Snapshot-A.xlsx"
        write_snapshot(p, "2026-05-01", 3)
        df, hdr = read_snapshot(p)
        assert len(df) == 288 and df["block"].min() == 1 and df["block"].max() == 96
        assert hdr == (pd.Timestamp("2026-05-01"), pd.Timestamp("2026-05-03"))
    print("PASS: 3 days -> 288 rows, blocks 1-96, summary rows ignored, header range read")


def test_duplicate_file_is_dropped_and_reported():
    with tempfile.TemporaryDirectory() as d:
        write_snapshot(Path(d) / "DAM_Market Snapshot-Aug.xlsx", "2026-08-01", 30)
        write_snapshot(Path(d) / "DAM_Market Snapshot-Jul.xlsx", "2026-08-01", 30)   # the same month saved twice
        write_snapshot(Path(d) / "DAM_Market Snapshot-May.xlsx", "2026-05-01", 31)
        r = load_folder(d, "DAM")
        assert r.df["date"].nunique() == 61 and len(r.df) == 61 * 96
        assert r.duplicates == ["DAM_Market Snapshot-Jul.xlsx"], r.duplicates
        assert r.missing_days and any("missing" in w for w in r.warnings)
    print("PASS: a month exported twice is not double-counted and is reported; gap between months is reported")


def test_conflicting_values_stop_the_load():
    with tempfile.TemporaryDirectory() as d:
        write_snapshot(Path(d) / "DAM_Market Snapshot-A.xlsx", "2026-08-01", 5, price=5000.0)
        write_snapshot(Path(d) / "DAM_Market Snapshot-B.xlsx", "2026-08-03", 5, price=6000.0)
        try:
            load_folder(d, "DAM")
        except IexFileError as e:
            assert "conflicts" in str(e)
        else:
            raise AssertionError("conflicting overlap was accepted")
    print("PASS: two files that disagree on the same block stop the load")


def test_header_mismatch_and_short_day_fail():
    with tempfile.TemporaryDirectory() as d:
        write_snapshot(Path(d) / "DAM_Market Snapshot-A.xlsx", "2026-08-01", 5,
                       header_range=(pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-05")))
        try:
            load_folder(d, "DAM")
        except IexFileError as e:
            assert "header says" in str(e)
        else:
            raise AssertionError("header/data mismatch accepted")
    with tempfile.TemporaryDirectory() as d:
        write_snapshot(Path(d) / "RTM_Market Snapshot-A.xlsx", "2026-08-01", 2, drop_block=(1, 10))
        try:
            load_folder(d, "RTM")
        except IexFileError as e:
            assert "96 blocks" in str(e)
        else:
            raise AssertionError("short day accepted")
    print("PASS: header/data mismatch and a day with 95 blocks both stop the load")


def test_market_filter():
    with tempfile.TemporaryDirectory() as d:
        write_snapshot(Path(d) / "DAM_Market Snapshot-A.xlsx", "2026-08-01", 2)
        try:
            load_folder(d, "GDAM")
        except IexFileError as e:
            assert "No GDAM" in str(e)
        else:
            raise AssertionError("loaded the wrong market")
    print("PASS: only files for the requested market are read")


def write_layout(path, layout, start="2026-08-01", days=2):
    """Same data as write_snapshot but in the G-DAM or RTM export layout, with distinctive numbers in every
    column so a wrong mapping is obvious: MCP is always 4321.5 + block."""
    wb = openpyxl.Workbook()
    ws = wb.active
    end = pd.Timestamp(start) + pd.Timedelta(days=days - 1)
    ws.append([])
    ws.append(["Market Snapshot"])
    ws.append([f"Date: {pd.Timestamp(start):%d-%m-%Y} to {end:%d-%m-%Y}"])
    ws.append([])
    if layout == "gdam":
        ws.append(["Date", "Hour", "Time Block", "Purchase Bid (MW)", "Total Sell Bid (MW)", "Hydro Sell Bid (MW)",
                   "Wind Sell Bid (MW)", "Other RE Sell Bid (MW)", "DRE Sell Bid (MW)", "Total MCV (MW)",
                   "Hydro MCV (MW)", "Wind MCV (MW)", "Other RE MCV (MW)", "DRE MCV (MW)", "Total FSV (MW)",
                   "Hydro FSV (MW)", "Wind FSV (MW)", "Other RE FSV (MW)", "DRE FSV (MW)", "MCP (Rs/MWh)"])
    else:
        ws.append(["Date", "Hour", "Session ID", "Time Block", "Purchase Bid (MW)", "Sell Bid (MW)", "MCV (MW)",
                   "Final Scheduled Volume (MW)", "MCP (Rs/MWh) *"])
    for d in range(days):
        date = pd.Timestamp(start) + pd.Timedelta(days=d)
        for b in range(96):
            h, m = divmod(b * 15, 60)
            e = (b + 1) * 15
            if layout == "gdam":
                tb = f"{h:02d}:{m:02d} - {e // 60:02d}:{e % 60:02d}"
                ws.append([f"{date:%d-%m-%Y}", h + 1, tb, 3000.0, 900.0, 11.0, 222.0, 333.0, 44.0, 800.0,
                           12.0, 555.0, 13.0, 14.0, 799.0, 15.0, 16.0, 17.0, 18.0, 4321.5 + b])
            else:
                tb = f"{h:02d}:{m:02d}-{e // 60:02d}:{e % 60:02d}"
                ws.append([f"{date:%d-%m-%Y}", h + 1, b // 2 + 1, tb, "20000.00", "7000.00", "6000.00", "5999.00",
                           str(4321.5 + b)])
    ws.append(["Date", "Summary", "x"])
    wb.save(path)


def test_column_layouts_are_mapped_by_header():
    with tempfile.TemporaryDirectory() as d:
        write_layout(Path(d) / "GDAM_Market Snapshot-x.xlsx", "gdam")
        write_layout(Path(d) / "RTM_Market Snapshot-x.xlsx", "rtm")
        g = load_folder(d, "GDAM").df
        r = load_folder(d, "RTM").df
        for df in (g, r):
            assert len(df) == 192 and (df["mcp"] == 4321.5 + df["block"] - 1).all(), "price taken from the wrong column"
        assert (g["sell_mw"] == 900.0).all() and (g["mcv_mw"] == 800.0).all() and (g["fsv_mw"] == 799.0).all()
        assert (g["wind_sell_mw"] == 222.0).all() and (g["wind_mcv_mw"] == 555.0).all()
        assert (r["sell_mw"] == 7000.0).all() and (r["purchase_mw"] == 20000.0).all() and r["wind_sell_mw"].isna().all()
    print("PASS: G-DAM layout (20 columns) and RTM layout (extra Session ID) both map price, bids and volumes by header")


def test_missing_required_column_fails():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "RTM_Market Snapshot-x.xlsx"
        write_layout(p, "rtm")
        wb = openpyxl.load_workbook(p)
        wb.active.cell(row=5, column=9).value = "Price"          # rename the MCP header
        wb.save(p)
        try:
            load_folder(d, "RTM")
        except IexFileError as e:
            assert "required column missing" in str(e)
        else:
            raise AssertionError("a missing price column was not caught")
    print("PASS: a renamed / missing price column stops the load instead of guessing")


if __name__ == "__main__":
    test_column_layouts_are_mapped_by_header()
    test_missing_required_column_fails()
    test_market_names()
    test_reads_blocks_and_ignores_summary()
    test_duplicate_file_is_dropped_and_reported()
    test_conflicting_values_stop_the_load()
    test_header_mismatch_and_short_day_fail()
    test_market_filter()
    print("\nAll IEX importer tests passed.")
