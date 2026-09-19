"""
G-DAM and RTM price forecast, per 15-minute block.

Deliberately small: a few thousand block-days is not enough for anything clever.

  * Three quantile models per market (P10 / P50 / P90): shallow gradient boosting
    on the block-wise features in `engine.features`.
  * The P50 is blended 50/50 with a persistence anchor (same block on the last day
    that market's price is known). Persistence is the strongest naive predictor in
    this data, so the blend can only add to it, not replace it.
  * The blend weight and every model setting below are FIXED, not tuned per run,
    so a 3-week walk-forward cannot be overfitted by adjusting them.
  * Prices are clipped to [0, exchange cap].

If there is too little history to fit (fewer than MIN_TRAIN_ROWS), the forecast
falls back to the anchor with a band from its own past errors, and says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from .features import CAP, FEATURES, Policy

MIN_TRAIN_ROWS = 288        # three days of blocks
QUANTILES = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class ModelConfig:
    max_depth: int = 3
    max_iter: int = 150
    learning_rate: float = 0.05
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    blend_w: float = 0.5        # weight on the boosted model; the rest is the persistence anchor
    seed: int = 0


# market -> (anchor feature order, target column)
MARKETS = {
    "g": (["g_l1", "g_nb", "g_m3", "g_all"], "y_gdam"),
    "r": (["r_l2", "r_nb", "r_m3", "r_all"], "y_rtm"),
}


def select_training(frame: pd.DataFrame, target, policy: Policy = Policy()) -> dict[str, pd.DataFrame]:
    """Rows whose realised price the desk already knows when deciding for `target`."""
    target = pd.Timestamp(target)
    g = frame[(frame["date"] <= target - pd.Timedelta(days=policy.lag_gdam)) & frame["y_gdam"].notna()]
    r = frame[(frame["date"] <= target - pd.Timedelta(days=policy.lag_rtm)) & frame["y_rtm"].notna()]
    return {"g": g, "r": r}


def _anchor(X: pd.DataFrame, order: list[str], fallback: float) -> np.ndarray:
    a = X[order[0]].to_numpy(dtype=float).copy()
    for col in order[1:]:
        a = np.where(np.isfinite(a), a, X[col].to_numpy(dtype=float))
    return np.where(np.isfinite(a), a, fallback)


class PriceModel:
    def __init__(self, cfg: ModelConfig = ModelConfig(), features: list[str] | None = None,
                 quantiles: tuple = QUANTILES):
        self.cfg = cfg
        self.quantiles = tuple(quantiles)
        self.features = list(features) if features is not None else list(FEATURES)
        self.models: dict[str, dict[float, HistGradientBoostingRegressor]] = {}
        self.cols: dict[str, list[str]] = {}     # features that actually vary in the training window
        self.fallback: dict[str, float] = {}
        self.resid_band: dict[str, tuple[float, float]] = {}
        self.fitted: dict[str, bool] = {}
        self.n_rows: dict[str, int] = {}
        self.n_days: dict[str, int] = {}
        self._train: dict[str, pd.DataFrame] = {}

    def _gbm(self, q: float) -> HistGradientBoostingRegressor:
        c = self.cfg
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_depth=c.max_depth, max_iter=c.max_iter,
            learning_rate=c.learning_rate, min_samples_leaf=c.min_samples_leaf,
            l2_regularization=c.l2_regularization, random_state=c.seed, early_stopping=False,
        )

    def fit(self, train: dict[str, pd.DataFrame]) -> "PriceModel":
        for m, (order, ycol) in MARKETS.items():
            df = train[m]
            y = df[ycol].to_numpy(dtype=float)
            self.n_rows[m] = len(df)
            self.n_days[m] = int(df["date"].nunique()) if len(df) else 0
            self.fallback[m] = float(np.nanmedian(y)) if len(y) else 4000.0
            self._train[m] = df
            anchor = _anchor(df, order, self.fallback[m]) if len(df) else np.array([])
            resid = y - anchor if len(df) else np.array([0.0])
            self.resid_band[m] = (float(np.nanquantile(resid, 0.1)), float(np.nanquantile(resid, 0.9))) \
                if len(resid) >= 96 else (-1500.0, 1500.0)
            if len(df) >= MIN_TRAIN_ROWS:
                # a feature with a single value in the window (e.g. no holiday seen yet) carries no
                # information and trips the binning step, so it is left out until it varies
                self.cols[m] = [c for c in self.features if df[c].nunique(dropna=True) >= 2]
                X = df[self.cols[m]]
                self.models[m] = {q: self._gbm(q).fit(X, y) for q in self.quantiles}
                self.fitted[m] = True
            else:
                self.fitted[m] = False
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """X: 96-row (or many-row) frame with FEATURES columns. Returns one row per input row."""
        out = pd.DataFrame(index=X.index)
        w = self.cfg.blend_w
        for m, (order, _) in MARKETS.items():
            anchor = _anchor(X, order, self.fallback.get(m, 4000.0))
            if self.fitted.get(m):
                Xc = X[self.cols[m]]
                q50 = self.models[m][0.5].predict(Xc)
                p50 = w * q50 + (1 - w) * anchor
                if 0.1 in self.models[m] and 0.9 in self.models[m]:
                    lo = p50 - np.maximum(q50 - self.models[m][0.1].predict(Xc), 0.0)
                    hi = p50 + np.maximum(self.models[m][0.9].predict(Xc) - q50, 0.0)
                else:                       # median-only model: band from the anchor's own training residuals
                    lo, hi = p50 + self.resid_band[m][0], p50 + self.resid_band[m][1]
            else:
                p50 = anchor
                lo, hi = anchor + self.resid_band[m][0], anchor + self.resid_band[m][1]
            out[f"{m}_p10"] = np.clip(np.minimum(lo, p50), 0, CAP)
            out[f"{m}_p50"] = np.clip(p50, 0, CAP)
            out[f"{m}_p90"] = np.clip(np.maximum(hi, p50), 0, CAP)
            out[f"{m}_anchor"] = np.clip(anchor, 0, CAP)
        return out

    def importance(self, market: str, n_repeats: int = 3, max_rows: int = 800) -> pd.Series:
        """Permutation importance of each feature on the P50 model (drop in negative
        MAE when the feature is shuffled). Empty if the market was not fitted."""
        if not self.fitted.get(market):
            return pd.Series(dtype=float)
        df = self._train[market]
        df = df.sample(min(len(df), max_rows), random_state=self.cfg.seed)
        ycol = MARKETS[market][1]
        res = permutation_importance(
            self.models[market][0.5], df[self.cols[market]], df[ycol], n_repeats=n_repeats,
            random_state=self.cfg.seed, scoring="neg_mean_absolute_error",
        )
        return pd.Series(res.importances_mean, index=self.cols[market]).sort_values(ascending=False)
