import numpy as np
import pandas as pd

from src.data import EXTERNAL_TICKERS

LAG_DAYS = [1, 2, 3, 5, 10]
ROLL_WINDOWS = [7, 14]
EXTERNAL_LAGS = [1, 2, 5]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gold features: lag, rolling mean/std, return, volatility, momentum slope
    External features: lag t-1,2,5 per series.
    """
    feat = df.copy()
    price = feat["close"]

    for lag in LAG_DAYS:
        feat[f"lag_{lag}"] = price.shift(lag)

    for w in ROLL_WINDOWS:
        feat[f"roll_mean_{w}"] = price.shift(1).rolling(w).mean()
        feat[f"roll_std_{w}"] = price.shift(1).rolling(w).std()

    feat["return_1d"] = price.pct_change(1)
    feat["return_5d"] = price.pct_change(5)
    feat["volatility"] = feat["return_1d"].rolling(10).std()

    def linear_slope(series: pd.Series, window: int) -> pd.Series:
        x = np.arange(window)
        return (
            series.shift(1)
            .rolling(window)
            .apply(
                lambda y: np.polyfit(x, y, 1)[0] / y.mean(),
                raw=True,
            )
        )

    feat["momentum_slope_5"] = linear_slope(price, 5)
    feat["momentum_slope_10"] = linear_slope(price, 10)

    for ext in EXTERNAL_TICKERS:
        if ext not in feat.columns:
            continue
        for lag in EXTERNAL_LAGS:
            feat[f"{ext}_lag{lag}"] = feat[ext].shift(lag)
        feat[f"{ext}_return"] = feat[ext].pct_change(1)

    return feat.dropna()


def get_feature_cols(df_feat: pd.DataFrame) -> list[str]:
    base = (
        [f"lag_{l}" for l in LAG_DAYS]
        + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
        + [f"roll_std_{w}" for w in ROLL_WINDOWS]
        + ["return_1d", "return_5d", "volatility", "momentum_slope_5", "momentum_slope_10"]
    )
    ext = [c for c in df_feat.columns if any(c.startswith(f"{e}_") for e in EXTERNAL_TICKERS)]
    return [c for c in base + ext if c in df_feat.columns]
