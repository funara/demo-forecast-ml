import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data import EXTERNAL_TICKERS
from src.features import EXTERNAL_LAGS, LAG_DAYS, ROLL_WINDOWS

FORECAST_DAYS = 66
N_FOLDS = 5
CI_LOWER = 0.025
CI_UPPER = 0.975


def walk_forward_splits(df: pd.DataFrame, n_folds: int) -> list[tuple]:
    n = len(df)
    min_train = int(n * 0.60)
    test_size = (n - min_train) // n_folds
    splits = []
    for i in range(n_folds):
        train_end = min_train + i * test_size
        test_end = min(train_end + test_size, n)
        splits.append((df.index[:train_end], df.index[train_end:test_end]))
    return splits


def make_mean_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=-1,
    )


def make_quantile_model(q: float) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        objective="reg:quantileerror",
        quantile_alpha=q,
        random_state=42,
        n_jobs=-1,
    )


def make_ar_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
    )


def _prepare_Xy(
    df_feat: pd.DataFrame, mask: pd.Index, feature_cols: list[str], horizon: int
) -> tuple[pd.DataFrame, np.ndarray]:
    X = df_feat.loc[mask, feature_cols]
    y = df_feat["close"].shift(-horizon).loc[mask].values
    ok = ~np.isnan(y)
    return X[ok], y[ok]


def train_direct_suite(
    df_feat: pd.DataFrame, train_idx: pd.Index, feature_cols: list[str]
) -> dict:
    mean_models, lower_models, upper_models, scalers = [], [], [], []
    for h in range(1, FORECAST_DAYS + 1):
        X, y = _prepare_Xy(df_feat, train_idx, feature_cols, h)
        scaler = StandardScaler()
        X_sc = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

        m_mean = make_mean_model()
        m_mean.fit(X_sc, y)
        m_lower = make_quantile_model(CI_LOWER)
        m_lower.fit(X_sc, y)
        m_upper = make_quantile_model(CI_UPPER)
        m_upper.fit(X_sc, y)

        mean_models.append(m_mean)
        lower_models.append(m_lower)
        upper_models.append(m_upper)
        scalers.append(scaler)

    return {"mean": mean_models, "lower": lower_models, "upper": upper_models, "scaler": scalers}


def train_ar_models(df: pd.DataFrame, train_idx: pd.Index) -> dict[str, tuple]:
    ar_models = {}
    ar_lags = [1, 2, 3, 5]
    for name in EXTERNAL_TICKERS:
        if name not in df.columns:
            continue
        series = df[name].loc[train_idx]
        X_rows, y_rows = [], []
        for i in range(max(ar_lags), len(series)):
            row = [series.iloc[i - lag] for lag in ar_lags]
            X_rows.append(row)
            y_rows.append(series.iloc[i])

        X = pd.DataFrame(X_rows, columns=[f"lag_{l}" for l in ar_lags])
        y = np.array(y_rows)

        scaler = StandardScaler()
        X_sc = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        model = make_ar_model()
        model.fit(X_sc, y)
        ar_models[name] = (model, scaler)

    return ar_models


def recursive_forecast_externals(
    df: pd.DataFrame, ar_models: dict[str, tuple], steps: int
) -> pd.DataFrame:
    ar_lags = [1, 2, 3, 5]
    histories = {name: list(df[name].values) for name in ar_models}
    results = {name: [] for name in ar_models}

    for _ in range(steps):
        for name, (model, scaler) in ar_models.items():
            hist = histories[name]
            row = pd.DataFrame(
                [[hist[-lag] for lag in ar_lags]], columns=[f"lag_{l}" for l in ar_lags]
            )
            X_sc = pd.DataFrame(scaler.transform(row), columns=row.columns)
            pred = float(model.predict(X_sc)[0])
            results[name].append(pred)
            hist.append(pred)

    return pd.DataFrame(results)


def walk_forward_evaluate(
    df_feat: pd.DataFrame, feature_cols: list[str], splits: list[tuple]
) -> tuple[float, list[float], list[float]]:
    fold_mapes = []
    horizon_mapies_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        suite = train_direct_suite(df_feat, train_idx, feature_cols)
        fold_horizon = []

        for h_idx in range(FORECAST_DAYS):
            h = h_idx + 1
            truncated_test = test_idx[:-h] if h < len(test_idx) else test_idx[:0]
            if len(truncated_test) == 0:
                fold_horizon.append(np.nan)
                continue

            scaler = suite["scaler"][h_idx]
            model = suite["mean"][h_idx]
            X_test = df_feat.loc[truncated_test, feature_cols]
            X_sc = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
            y_pred = model.predict(X_sc)

            y_true = df_feat["close"].shift(-h).loc[truncated_test].values
            valid = ~np.isnan(y_true)
            if valid.sum() == 0:
                fold_horizon.append(np.nan)
                continue

            mape = mean_absolute_percentage_error(y_true[valid], y_pred[valid]) * 100
            fold_horizon.append(mape)

        valid_h = [m for m in fold_horizon if not np.isnan(m)]
        fold_mapes.append(float(np.mean(valid_h)) if valid_h else 0.0)
        horizon_mapies_per_fold.append(fold_horizon)

    per_horizon = []
    for h_idx in range(FORECAST_DAYS):
        vals = [
            fold[h_idx]
            for fold in horizon_mapies_per_fold
            if h_idx < len(fold) and not np.isnan(fold[h_idx])
        ]
        per_horizon.append(float(np.mean(vals)) if vals else 0.0)

    avg = float(np.mean(per_horizon))
    return avg, fold_mapes, per_horizon


def forecast_with_recursive_externals(
    df: pd.DataFrame, df_feat: pd.DataFrame, suite: dict, ar_models: dict, feature_cols: list[str]
) -> pd.DataFrame:
    last_date = df_feat.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS)

    ext_forecast = recursive_forecast_externals(df, ar_models, steps=FORECAST_DAYS)

    last_row = df_feat[feature_cols].iloc[-1].to_dict()
    gold_hist = list(df["close"].values)
    means, lowers, uppers = [], [], []

    for h in range(FORECAST_DAYS):
        row = last_row.copy()

        for lag in LAG_DAYS:
            if len(means) >= lag:
                row[f"lag_{lag}"] = means[-lag]
            else:
                row[f"lag_{lag}"] = gold_hist[-(lag - len(means))]

        recent = (means + gold_hist)[-max(ROLL_WINDOWS) :]
        for w in ROLL_WINDOWS:
            window_vals = recent[-w:]
            row[f"roll_mean_{w}"] = float(np.mean(window_vals))
            row[f"roll_std_{w}"] = float(np.std(window_vals))

        if len(means) >= 1:
            row["return_1d"] = (means[-1] - (means[-2] if len(means) >= 2 else gold_hist[-1])) / (
                means[-2] if len(means) >= 2 else gold_hist[-1]
            )
        if len(means) >= 5:
            row["return_5d"] = (means[-1] - means[-5]) / means[-5]
        row["volatility"] = (
            float(
                np.std(
                    [
                        (means[i] - means[i - 1]) / means[i - 1]
                        for i in range(1, min(10, len(means)))
                    ]
                )
            )
            if len(means) > 1
            else last_row["volatility"]
        )

        recent_prices = (gold_hist + means)[-(max(10, 1)) :]
        if len(recent_prices) >= 5:
            x5 = np.arange(5)
            y5 = recent_prices[-5:]
            row["momentum_slope_5"] = float(np.polyfit(x5, y5, 1)[0] / np.mean(y5))
        if len(recent_prices) >= 10:
            x10 = np.arange(10)
            y10 = recent_prices[-10:]
            row["momentum_slope_10"] = float(np.polyfit(x10, y10, 1)[0] / np.mean(y10))

        for name in ar_models:
            for lag in EXTERNAL_LAGS:
                col = f"{name}_lag{lag}"
                if col in row:
                    idx = h - lag
                    if idx >= 0:
                        row[col] = ext_forecast[name].iloc[idx]
            ret_col = f"{name}_return"
            if ret_col in row and h >= 1:
                prev = ext_forecast[name].iloc[h - 1] if h >= 1 else df[name].iloc[-1]
                curr = ext_forecast[name].iloc[h]
                row[ret_col] = (curr - prev) / prev if prev != 0 else 0.0

        X_df = pd.DataFrame([row])[feature_cols]
        scaler = suite["scaler"][h]
        X_sc = pd.DataFrame(scaler.transform(X_df), columns=feature_cols)

        means.append(float(suite["mean"][h].predict(X_sc)[0]))
        lowers.append(float(suite["lower"][h].predict(X_sc)[0]))
        uppers.append(float(suite["upper"][h].predict(X_sc)[0]))

    return pd.DataFrame(
        {
            "tanggal": future_dates,
            "prediksi_idr_gram": np.round(means).astype(int),
            "lower_95": np.round(lowers).astype(int),
            "upper_95": np.round(uppers).astype(int),
        }
    )
