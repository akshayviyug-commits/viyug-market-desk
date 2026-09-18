import sys, io
sys.path.insert(0, ".")
from engine.loader import load_workbook

PATH = "sample_data/sample_workbook.xlsx"
with open(PATH, "rb") as f:
    buf = io.BytesIO(f.read())

result = load_workbook(buf)
print("Loaded from BytesIO OK:", result.n_settled_days, "settled days,", result.date_range)
