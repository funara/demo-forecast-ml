import pandas as pd
import numpy as np
from src.features import build_features, get_feature_cols, LAG_DAYS, ROLL_WINDOWS, EXTERNAL_LAGS


def _make_synthetic_df(n_days=100, include_dxy=True, include_vix=True):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    np.random.seed(42)
    close = 1000 + np.cumsum(np.random.randn(n_days) * 5)
    data = {"close": close}
    if include_dxy:
        data["dxy"] = 104 + np.cumsum(np.random.randn(n_days) * 0.1)
    if include_vix:
        data["vix"] = 15 + np.cumsum(np.random.randn(n_days) * 0.5)
    return pd.DataFrame(data, index=dates)


def test_build_features_returns_expected_columns():
    df = _make_synthetic_df()
    feat = build_features(df)
    cols = feat.columns.tolist()
    for lag in LAG_DAYS:
        assert f"lag_{lag}" in cols
    for w in ROLL_WINDOWS:
        assert f"roll_mean_{w}" in cols
        assert f"roll_std_{w}" in cols
    assert "return_1d" in cols
    assert "return_5d" in cols
    assert "volatility" in cols
    assert "momentum_slope_5" in cols
    assert "momentum_slope_10" in cols


def test_build_features_external_lags():
    df = _make_synthetic_df()
    feat = build_features(df)
    for lag in EXTERNAL_LAGS:
        assert f"dxy_lag{lag}" in feat.columns
        assert f"vix_lag{lag}" in feat.columns
    assert "dxy_return" in feat.columns
    assert "vix_return" in feat.columns


def test_build_features_drops_nan_rows():
    df = _make_synthetic_df(n_days=20)
    feat = build_features(df)
    assert len(feat) < len(df)


def test_build_features_skips_missing_external():
    df = _make_synthetic_df(include_dxy=False, include_vix=False)
    feat = build_features(df)
    assert "dxy_lag1" not in feat.columns
    assert "vix_lag1" not in feat.columns
    # Gold-only features still present
    assert "lag_1" in feat.columns


def test_get_feature_cols_returns_only_present():
    df = _make_synthetic_df(include_dxy=True, include_vix=False)
    feat = build_features(df)
    cols = get_feature_cols(feat)
    assert "dxy_lag1" in cols
    assert "vix_lag1" not in cols
    assert all(c in feat.columns for c in cols)


def test_get_feature_cols_no_external():
    df = _make_synthetic_df(include_dxy=False, include_vix=False)
    feat = build_features(df)
    cols = get_feature_cols(feat)
    assert not any(c.startswith("dxy_") or c.startswith("vix_") for c in cols)
    assert "lag_1" in cols
