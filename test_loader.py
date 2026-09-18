import sys
sys.path.insert(0, ".")
from engine.loader import load_workbook, SchemaError, DataQualityError

PATH = r"D:\Viyug\00. HATAGERI PH-1 (102.3 MW) (2).xlsx"

try:
    result = load_workbook(PATH)
except (SchemaError, DataQualityError) as e:
    print("LOAD FAILED:")
    print(e)
    sys.exit(1)

print("Loaded OK")
print("Rows:", len(result.df))
print("Date range:", result.date_range)
print("Settled days:", result.n_settled_days, result.settled_dates[:3], "...", result.settled_dates[-3:])
print("Forecast-only days:", result.forecast_only_dates)
print("Warnings:", result.warnings)
print()
print(result.df.head(3).to_string())
print()
print("dtypes:")
print(result.df.dtypes)
