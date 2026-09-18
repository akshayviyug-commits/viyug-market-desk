"""
Daily planning mode - Market_Desk_Requirements.docx section 8.2 and section 4.

Consumes tomorrow's 96-block generation forecast plus the calibration
table, applies the chosen strategy's per-block DA share (or a manual
override), and emits the DA/RTM MW split the trader files.

Guardrails (docx sec. 4): exactly 96 rows; blocks 1-96 present exactly
once; no negative values; flag (never silently clamp) any forecast value
exceeding AvC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_SHARE_COLUMN = {
    "conservative": "da_share_conservative",
    "balanced": "da_share_balanced",
    "aggressive": "da_share_aggressive",
}


class ForecastValidationError(Exception):
    pass


def validate_forecast(forecast: pd.DataFrame, avc_by_block: dict | None = None) -> list:
    """forecast: DataFrame with columns ['block', 'forecast_mw']. Returns a list
    of warning strings for AvC exceedance (flagged, not clamped); raises on
    structural problems (missing/duplicate blocks, negative values)."""
    blocks = sorted(forecast["block"].tolist())
    if len(blocks) != 96 or blocks != list(range(1, 97)):
        counts = pd.Series(blocks).value_counts()
        dupes = counts[counts > 1].index.tolist()
        missing = sorted(set(range(1, 97)) - set(blocks))
        raise ForecastValidationError(
            f"Forecast must have exactly 96 rows, blocks 1-96 each once. Got {len(blocks)} rows."
            + (f" Duplicated: {dupes}." if dupes else "")
            + (f" Missing: {missing}." if missing else "")
        )
    negative = forecast[forecast["forecast_mw"] < 0]
    if len(negative):
        raise ForecastValidationError(
            f"Forecast has {len(negative)} negative value(s) at block(s) "
            f"{negative['block'].tolist()} - forecast MW cannot be negative."
        )

    warnings = []
    if avc_by_block:
        for _, row in forecast.iterrows():
            avc = avc_by_block.get(int(row["block"]))
            if avc is not None and row["forecast_mw"] > avc:
                warnings.append(
                    f"Block {int(row['block'])}: forecast {row['forecast_mw']:.2f} MW "
                    f"exceeds AvC {avc:.2f} MW - flagged, not clamped (possible forecast "
                    f"or capacity-register problem)."
                )
    return warnings


def build_daily_plan(forecast: pd.DataFrame, calibration: pd.DataFrame,
                      strategy: str = "balanced", overrides: dict | None = None,
                      avc_by_block: dict | None = None) -> dict:
    """
    forecast: DataFrame ['block', 'forecast_mw'] for the delivery date.
    calibration: the 96-row calibration output table.
    strategy: 'conservative' | 'balanced' | 'aggressive'.
    overrides: optional {block: da_share} human overrides (defaults to
               calibrated value when absent, per docx 8.3).
    """
    if strategy not in STRATEGY_SHARE_COLUMN:
        raise ValueError(f"Unknown strategy '{strategy}'. Must be one of {list(STRATEGY_SHARE_COLUMN)}.")
    overrides = overrides or {}
    warnings = validate_forecast(forecast, avc_by_block)

    share_col = STRATEGY_SHARE_COLUMN[strategy]
    merged = forecast.merge(calibration[["block", "time", share_col]], on="block", how="left")
    if merged[share_col].isna().any():
        missing = merged[merged[share_col].isna()]["block"].tolist()
        raise ForecastValidationError(f"No calibrated share for block(s) {missing}.")

    merged["da_share_applied"] = merged["block"].map(overrides).fillna(merged[share_col])
    merged["override_flag"] = merged["block"].isin(overrides.keys())
    merged["da_mw"] = merged["forecast_mw"] * merged["da_share_applied"]
    merged["rtm_mw"] = merged["forecast_mw"] - merged["da_mw"]

    # thumb-rule identity, exactly (docx acceptance test #3 / CIP-T05)
    thumb_rule_error = (merged["da_mw"] + merged["rtm_mw"] - merged["forecast_mw"]).abs().max()
    assert thumb_rule_error < 1e-9, f"Thumb-rule identity violated: max error {thumb_rule_error}"
    assert (merged["da_mw"] >= 0).all() and (merged["rtm_mw"] >= 0).all(), "Negative DA/RTM volume produced."

    total_forecast_mwh = merged["forecast_mw"].sum() * 0.25
    total_da_mwh = merged["da_mw"].sum() * 0.25
    total_rtm_mwh = merged["rtm_mw"].sum() * 0.25
    overall_da_share = total_da_mwh / total_forecast_mwh if total_forecast_mwh else 0.0

    summary = {
        "total_forecast_mwh": round(total_forecast_mwh, 2),
        "total_da_mwh": round(total_da_mwh, 2),
        "total_rtm_mwh": round(total_rtm_mwh, 2),
        "overall_da_share": round(overall_da_share, 4),
        "override_count": int(merged["override_flag"].sum()),
    }

    plan = merged[["block", "time", "forecast_mw", "da_share_applied", "da_mw", "rtm_mw", "override_flag"]].copy()
    return {"plan": plan, "summary": summary, "warnings": warnings, "strategy": strategy}
