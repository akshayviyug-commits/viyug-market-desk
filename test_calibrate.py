import sys
sys.path.insert(0, ".")
from engine.loader import load_workbook
from engine.calibrate import run_calibration

PATH = r"D:\Viyug\00. HATAGERI PH-1 (102.3 MW) (2).xlsx"
result = load_workbook(PATH)
out = run_calibration(result.df, result.settled_dates)

pd_opts = ["da_share_conservative", "da_share_balanced", "da_share_aggressive", "baseline_da_share"]
print(out[["block", "time"] + pd_opts].to_string(index=False))
print()
print("Share ranges:")
for c in pd_opts:
    print(f"  {c}: min={out[c].min():.2f} max={out[c].max():.2f} mean={out[c].mean():.3f}")
print()
degenerate = out[(out["da_share_aggressive"] == 0) | (out["da_share_aggressive"] == 1)]
print(f"Blocks where Aggressive hits the 0%/100% extreme: {len(degenerate)} / 96")

out.to_csv("calibration_output.csv", index=False)
print("\nWrote calibration_output.csv")
