import sys
sys.path.insert(0, ".")
import pandas as pd
from engine.loader import load_workbook, _read_raw_sheet, _map_columns, _derive_date_only
from engine.dsm import dsm_charge

PATH = r"D:\Viyug\00. HATAGERI PH-1 (102.3 MW) (2).xlsx"
result = load_workbook(PATH)
df = result.df[result.df["date"].isin(result.settled_dates)].copy()

# also pull the sheet's own DSM columns directly for comparison
raw = _read_raw_sheet(PATH)
extra_map = {
    "ui_band1_sheet": "UI DSM between 10% to 15% (INR)",
    "ui_band2_sheet": "DSM due to Deviation >15 % (INR)",
    "oi_band1_sheet": "OI DSM between 10% to 15% (INR)",
    "oi_band2_sheet": "OI Loss for >15% (INR)",
}
extra = pd.DataFrame()
for canon, src in extra_map.items():
    col = raw[src]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    extra[canon] = col
extra = _derive_date_only(pd.concat([raw[["Date"]].rename(columns={"Date": "date"}), extra], axis=1))
extra = extra.dropna(subset=["date"]).reset_index(drop=True)

merged = df.reset_index(drop=True).join(extra.drop(columns=["date"]))

computed = dsm_charge(
    actual=merged["actual_ll"], scheduled=merged["final_schedule"],
    avc=merged["avc"], acp=merged["acp"],
)

for key, sheet_col in [("ui_band1", "ui_band1_sheet"), ("ui_band2", "ui_band2_sheet"),
                        ("oi_band1", "oi_band1_sheet"), ("oi_band2", "oi_band2_sheet")]:
    err = (computed[key] - merged[sheet_col]).abs().max()
    print(f"Max abs error {key}: {err:.2e}")
