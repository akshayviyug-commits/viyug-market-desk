"""Regression test: a workbook a user uploads must replace whatever was loaded earlier in the same server process.

The bug this guards against: a cache keyed on the wrong thing served the first workbook ever loaded (here the sample)
to every later upload, so an upload dated August showed January's numbers.
"""
import io
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = ["app.py"]

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

SAMPLE = HERE / "sample_data" / "sample_workbook.xlsx"


def shifted_workbook(days: int) -> bytes:
    """The sample workbook with every date moved by `days`: same layout, different data on different dates."""
    df = pd.read_excel(SAMPLE, sheet_name="5. Detailed sheet", header=3)
    df["Date"] = pd.to_datetime(df["Date"]) + pd.Timedelta(days=days)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="5. Detailed sheet", index=False, header=True, startrow=3)
    return buf.getvalue()


class Upload:
    def __init__(self, data: bytes, name: str = "workbook.xlsx"):
        self._data, self.name, self.size, self.file_id = data, name, len(data), name

    def getvalue(self):
        return self._data


def run_app(upload: Upload | None):
    original = st.file_uploader
    st.file_uploader = lambda *a, **k: upload if (upload is not None and k.get("key") == "wb_up") else original(*a, **k)
    try:
        at = AppTest.from_file(str(HERE / "app.py"), default_timeout=600)
        if upload is None:
            at.session_state["use_sample"] = True
        at.run()
    finally:
        st.file_uploader = original
    assert not at.exception, [e.value[:300] for e in at.exception]
    return at, " ".join(m.value for m in at.markdown)


def review_labels(text: str) -> list[str]:
    return re.findall(r"(?:D-\d+|Yesterday) · \d+ \w{3}", text)


if __name__ == "__main__":
    at, txt = run_app(None)                                     # 1. the sample, as the first workbook on the server
    first = review_labels(txt)
    assert first and all("Jan" in l for l in first), first
    print(f"PASS: sample workbook shows its own dates {first[0]} ... {first[-1]}")

    up = Upload(shifted_workbook(60))                           # 2. a different workbook uploaded afterwards
    at, txt = run_app(up)
    second = review_labels(txt)
    assert second and all("Mar" in l for l in second), f"upload showed stale data: {second}"
    assert "Jan" not in " ".join(second)
    print(f"PASS: an upload after the sample shows the upload's dates {second[0]} ... {second[-1]}, not the sample's")

    up2 = Upload(shifted_workbook(120), name="workbook.xlsx")   # 3. same file name, different content
    at, txt = run_app(up2)
    third = review_labels(txt)
    assert third and all("May" in l for l in third), f"same name, different content showed stale data: {third}"
    print(f"PASS: a second upload with the same file name is read afresh {third[0]} ... {third[-1]}")

    at, txt = run_app(None)                                     # 4. and the sample is unaffected by the uploads
    assert all("Jan" in l for l in review_labels(txt))
    print("PASS: the sample is unaffected by uploads")
    print("\nAll data-switch tests passed.")
