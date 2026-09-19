"""
The price forecaster the app and the back-tests share.

For a delivery date T it trains only on days the desk already knows (G-DAM to T-1,
RTM to T-2), forecasts the 96 blocks, then calibrates the P10-P90 band from that same
model's own recent errors:

    for each of the last `calib_days` days d whose price is known at T's decision time,
    forecast d exactly as it would have been forecast then (walk-forward), record
    residual = actual - P50; the band is P50 + the 10th / 90th percentile of those
    residuals, pooled by time window.

This is split-conformal calibration on a rolling window. It replaces the boosting
model's own quantile band, which was too narrow in the back-test (68% / 52% coverage
against an 80% target). Point forecasts are untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bundle import DataBundle
from .features import CAP, WINDOW_LABELS, build_training_frame
from .price_model import ModelConfig, PriceModel, select_training

MIN_CALIB_DAYS = 4
MIN_POOL = 30
BAND = (0.10, 0.90)


class Forecaster:
    def __init__(self, bundle: DataBundle, cfg: ModelConfig = ModelConfig(), calib_days: int = 14):
        self.b, self.cfg, self.K = bundle, cfg, calib_days
        self.frame = build_training_frame(bundle.panel, bundle.policy)
        self._raw: dict[pd.Timestamp, pd.DataFrame] = {}
        self._info: dict[pd.Timestamp, dict] = {}
        self._model: dict[pd.Timestamp, PriceModel] = {}
        self.dates = set(pd.DatetimeIndex(bundle.panel.dates))

    # ------------------------------------------------------------------ raw
    def raw(self, T) -> pd.DataFrame:
        """Model forecast for day T, trained only on what is known at T's decision time."""
        T = pd.Timestamp(T).normalize()
        if T not in self._raw:
            if T not in self.dates:
                raise KeyError(f"{T:%d %b %Y} is outside the data on file.")
            x = self.frame[self.frame["date"] == T]
            model = PriceModel(self.cfg, self.b.features, quantiles=(0.5,)).fit(select_training(self.frame, T, self.b.policy))
            p = model.predict(x)
            keep = ["date", "block", "window", "fc", "y_gdam", "y_rtm"]
            self._raw[T] = pd.concat([x[keep].reset_index(drop=True), p.reset_index(drop=True)], axis=1)
            self._info[T] = {
                "train_days_g": model.n_days["g"], "train_days_r": model.n_days["r"],
                "fitted": bool(model.fitted["g"] and model.fitted["r"]),
                "features_g": list(model.cols.get("g", [])), "features_r": list(model.cols.get("r", [])),
            }
            self._model[T] = model
        return self._raw[T]

    # ------------------------------------------------------------ calibration
    def _pools(self, T) -> dict[str, pd.DataFrame]:
        lag = {"g": self.b.policy.lag_gdam, "r": self.b.policy.lag_rtm}
        ycol = {"g": "y_gdam", "r": "y_rtm"}
        pools = {"g": [], "r": []}
        for back in range(1, self.K + 1):
            day = T - pd.Timedelta(days=back)
            if day not in self.dates:
                continue
            for m in ("g", "r"):
                if back < lag[m]:
                    continue                       # that day's price is not known yet at T's decision time
                r = self.raw(day)
                ok = r[ycol[m]].notna()
                if ok.any():
                    pools[m].append(pd.DataFrame({"window": r.loc[ok, "window"], "resid": r.loc[ok, ycol[m]] - r.loc[ok, f"{m}_p50"]}))
        return {m: (pd.concat(v) if v else pd.DataFrame(columns=["window", "resid"])) for m, v in pools.items()}

    def forecast(self, T) -> pd.DataFrame:
        """Calibrated 96-block forecast. Columns g_/r_ p10, p50, p90, plus the uncalibrated band and anchors."""
        T = pd.Timestamp(T).normalize()
        r = self.raw(T).copy()
        pools = self._pools(T)
        info = dict(self._info[T])
        info["calibrated"] = {}
        for m in ("g", "r"):
            pool = pools[m]
            r[f"{m}_raw_p10"], r[f"{m}_raw_p90"] = r[f"{m}_p10"], r[f"{m}_p90"]
            enough = pool["resid"].size >= MIN_CALIB_DAYS * 96 // 2
            info["calibrated"][m] = bool(enough)
            if not enough:
                continue
            allq = np.quantile(pool["resid"], BAND)
            lo, hi = np.empty(len(r)), np.empty(len(r))
            for w in range(len(WINDOW_LABELS)):
                sel = pool["resid"][pool["window"] == w]
                q = np.quantile(sel, BAND) if len(sel) >= MIN_POOL else allq
                idx = (r["window"] == w).to_numpy()
                lo[idx], hi[idx] = q
            p50 = r[f"{m}_p50"].to_numpy()
            r[f"{m}_p10"] = np.clip(np.minimum(p50 + lo, p50), 0, CAP)
            r[f"{m}_p90"] = np.clip(np.maximum(p50 + hi, p50), 0, CAP)
        info["calib_days_used"] = int(self._n_calib_days(T))
        r.attrs["info"] = info
        return r

    def _n_calib_days(self, T) -> int:
        n = 0
        for back in range(1, self.K + 1):
            day = T - pd.Timedelta(days=back)
            if day in self.dates and back >= self.b.policy.lag_rtm and self.raw(day)["y_rtm"].notna().any():
                n += 1
        return n

    def info(self, T) -> dict:
        return self.forecast(T).attrs["info"]

    # ------------------------------------------------------------------- walk
    def walk(self, start, end) -> pd.DataFrame:
        """Calibrated forecasts for every day in [start, end] that has both prices; one long frame."""
        out = []
        for T in sorted(self.dates):
            if T < pd.Timestamp(start) or T > pd.Timestamp(end):
                continue
            f = self.forecast(T)
            if f["y_gdam"].notna().any() and f["y_rtm"].notna().any():
                out.append(f)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def importance(self, T, market: str = "g", **kw) -> pd.Series:
        self.raw(T)
        return self._model[pd.Timestamp(T).normalize()].importance(market, **kw)


def coverage(pred: pd.DataFrame) -> dict:
    """Share of realised prices inside [P10, P90], per market, on rows with a realised price."""
    out = {}
    for m, y in (("g", "y_gdam"), ("r", "y_rtm")):
        P = pred[pred[y].notna()]
        out[m] = float(((P[f"{m}_p10"] <= P[y]) & (P[y] <= P[f"{m}_p90"])).mean()) if len(P) else float("nan")
        out[m + "_raw"] = float(((P[f"{m}_raw_p10"] <= P[y]) & (P[y] <= P[f"{m}_raw_p90"])).mean()) if len(P) else float("nan")
    return out
