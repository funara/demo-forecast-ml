"""
Gold Price Forecasting — XGBoost v2
=====================================
Platform : Google Colab / lokal
Data     : GC=F + DXY + ^TNX (US10Y) + ^VIX via yfinance
           + USDIDR=X (kurs live)
Model    : XGBoost Direct multi-step
Improvements vs v1:
  1. External features  : DXY, US10Y, VIX (lag t-1,2,5)
  2. Recursive forecast : external features diprediksi dulu
                          via model AR terpisah, lalu dipakai
                          sebagai input gold forecast
  3. Walk-forward validation (5 fold, expanding window)
Unit     : IDR per gram
Forecast : 66 business days (~3 bulan) + CI 95%
Output   : gold_forecast_v2.png + gold_forecast_v2_3bulan.csv

Install:
    !pip install xgboost yfinance matplotlib pandas scikit-learn -q
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import yfinance as yf

from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings("ignore", category=UserWarning)

TROY_OZ_TO_GRAM  = 31.1035
TRAIN_START      = "2024-01-01"
FORECAST_DAYS    = 66
N_FOLDS          = 5
ACCENT_COLOR     = "#2A6FBF"   # biru
BACKGROUND_COLOR = "#F8FAFD"
CREDIT_TEXT      = "@adhirahmadian"

LAG_DAYS         = [1, 2, 3, 5, 10]
ROLL_WINDOWS     = [7, 14]
EXTERNAL_LAGS    = [1, 2, 5]
CI_LOWER         = 0.025
CI_UPPER         = 0.975

EXTERNAL_TICKERS = {
    "dxy":   "DX-Y.NYB",   # US Dollar Index
    "us10y": "^TNX",        # US 10-Year Treasury Yield
    "vix":   "^VIX",        # CBOE Volatility Index
}


# ── 1. Data Fetching ──────────────────────────────────────────────────────────

def fetch_usd_idr_rate() -> float:
    rate = yf.Ticker("USDIDR=X").fast_info["last_price"]
    print(f"USD/IDR (live): {rate:,.0f}")
    return float(rate)


def fetch_series(ticker: str) -> pd.Series:
    raw = yf.download(ticker, start=TRAIN_START, auto_adjust=True, progress=False)
    raw.columns = raw.columns.get_level_values(0)
    series = raw["Close"].copy()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return series.dropna().rename(ticker)


def fetch_all_data() -> pd.DataFrame:
    """Fetch gold + external tickers, align pada business day index bersama."""
    gold = fetch_series("GC=F").rename("gold_usd_oz")

    externals = {}
    for name, ticker in EXTERNAL_TICKERS.items():
        try:
            externals[name] = fetch_series(ticker)
            print(f"  {name:6s} ({ticker}): {len(externals[name])} rows")
        except Exception as e:
            print(f"  {name:6s} ({ticker}): GAGAL — {e}")

    df = pd.DataFrame({"gold_usd_oz": gold})
    for name, series in externals.items():
        df[name] = series

    df = df.ffill().dropna()
    return df


def convert_gold_to_idr(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    result = df.copy()
    result["close"] = (result["gold_usd_oz"] / TROY_OZ_TO_GRAM) * rate
    return result


# ── 2. Feature Engineering ────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gold features  : lag, rolling mean/std, return, volatility, momentum slope
    External features: lag t-1,2,5 per series (no lookahead)
    Momentum slope: linear regression slope dari 5 dan 10 hari terakhir —
    eksplisit sinyal ke model apakah harga sedang naik atau turun.
    """
    feat  = df.copy()
    price = feat["close"]

    for lag in LAG_DAYS:
        feat[f"lag_{lag}"] = price.shift(lag)

    for w in ROLL_WINDOWS:
        feat[f"roll_mean_{w}"] = price.shift(1).rolling(w).mean()
        feat[f"roll_std_{w}"]  = price.shift(1).rolling(w).std()

    feat["return_1d"]  = price.pct_change(1)
    feat["return_5d"]  = price.pct_change(5)
    feat["volatility"] = feat["return_1d"].rolling(10).std()

    # momentum slope: koefisien regresi linear harga 5 dan 10 hari terakhir
    # nilai negatif = downtrend, positif = uptrend
    def linear_slope(series: pd.Series, window: int) -> pd.Series:
        x = np.arange(window)
        return series.shift(1).rolling(window).apply(
            lambda y: np.polyfit(x, y, 1)[0] / y.mean(),  # normalized slope
            raw=True,
        )

    feat["momentum_slope_5"]  = linear_slope(price, 5)
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
        [f"lag_{l}"         for l in LAG_DAYS]
        + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
        + [f"roll_std_{w}"  for w in ROLL_WINDOWS]
        + ["return_1d", "return_5d", "volatility",
           "momentum_slope_5", "momentum_slope_10"]
    )
    ext = [
        c for c in df_feat.columns
        if any(c.startswith(f"{e}_") for e in EXTERNAL_TICKERS)
    ]
    return [c for c in base + ext if c in df_feat.columns]


# ── 3. Walk-Forward Validation ────────────────────────────────────────────────

def walk_forward_splits(df: pd.DataFrame, n_folds: int) -> list[tuple]:
    """
    Expanding window: setiap fold menambah data training.
    Minimum train size = 60% data total.
    Returns list of (train_idx, test_idx).
    """
    n          = len(df)
    min_train  = int(n * 0.60)
    test_size  = (n - min_train) // n_folds

    splits = []
    for i in range(n_folds):
        train_end = min_train + i * test_size
        test_end  = min(train_end + test_size, n)
        splits.append((
            df.index[:train_end],
            df.index[train_end:test_end],
        ))
    return splits


# ── 4. Model Factories ────────────────────────────────────────────────────────

def make_mean_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.1,
        random_state=42, n_jobs=-1,
    )


def make_quantile_model(q: float) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.1,
        objective="reg:quantileerror", quantile_alpha=q,
        random_state=42, n_jobs=-1,
    )


def make_ar_model() -> XGBRegressor:
    """Model AR ringan untuk prediksi external features."""
    return XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42, n_jobs=-1,
    )


# ── 5. Training ───────────────────────────────────────────────────────────────

def _prepare_Xy(
    df_feat: pd.DataFrame,
    mask: pd.Index,
    feature_cols: list[str],
    horizon: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    X  = df_feat.loc[mask, feature_cols]
    y  = df_feat["close"].shift(-horizon).loc[mask].values
    ok = ~np.isnan(y)
    return X[ok], y[ok]


def train_direct_suite(
    df_feat: pd.DataFrame,
    train_idx: pd.Index,
    feature_cols: list[str],
) -> dict:
    mean_models, lower_models, upper_models, scalers = [], [], [], []

    for h in range(1, FORECAST_DAYS + 1):
        X, y = _prepare_Xy(df_feat, train_idx, feature_cols, h)

        scaler = StandardScaler()
        X_sc   = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

        m_mean  = make_mean_model();         m_mean.fit(X_sc, y)
        m_lower = make_quantile_model(CI_LOWER); m_lower.fit(X_sc, y)
        m_upper = make_quantile_model(CI_UPPER); m_upper.fit(X_sc, y)

        mean_models.append(m_mean)
        lower_models.append(m_lower)
        upper_models.append(m_upper)
        scalers.append(scaler)

    return {
        "mean":   mean_models,
        "lower":  lower_models,
        "upper":  upper_models,
        "scaler": scalers,
    }


# ── 6. External Feature AR Models ────────────────────────────────────────────

def train_ar_models(df: pd.DataFrame, train_idx: pd.Index) -> dict[str, tuple]:
    """
    Per external series: train AR model dengan lag 1..5
    untuk prediksi 1 step ahead (dipakai secara recursive).
    Returns: {name: (model, scaler)}
    """
    ar_models = {}
    ar_lags   = [1, 2, 3, 5]

    for name in EXTERNAL_TICKERS:
        if name not in df.columns:
            continue

        series = df[name].loc[train_idx]
        X_rows, y_rows = [], []

        for i in range(max(ar_lags), len(series)):
            row = [series.iloc[i - lag] for lag in ar_lags]
            X_rows.append(row)
            y_rows.append(series.iloc[i])

        X  = pd.DataFrame(X_rows, columns=[f"lag_{l}" for l in ar_lags])
        y  = np.array(y_rows)

        scaler = StandardScaler()
        X_sc   = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        model = make_ar_model()
        model.fit(X_sc, y)
        ar_models[name] = (model, scaler)

    return ar_models


def recursive_forecast_externals(
    df: pd.DataFrame,
    ar_models: dict[str, tuple],
    steps: int,
) -> pd.DataFrame:
    """
    Prediksi external features secara recursive untuk `steps` hari ke depan.
    Setiap step menggunakan prediksi sebelumnya sebagai lag input.
    """
    ar_lags   = [1, 2, 3, 5]
    histories = {name: list(df[name].values) for name in ar_models}
    results   = {name: [] for name in ar_models}

    for _ in range(steps):
        for name, (model, scaler) in ar_models.items():
            hist = histories[name]
            row  = pd.DataFrame(
                [[hist[-lag] for lag in ar_lags]],
                columns=[f"lag_{l}" for l in ar_lags],
            )
            X_sc = pd.DataFrame(scaler.transform(row), columns=row.columns)
            pred = float(model.predict(X_sc)[0])
            results[name].append(pred)
            hist.append(pred)

    return pd.DataFrame(results)


# ── 7. Walk-Forward Evaluation ────────────────────────────────────────────────

def walk_forward_evaluate(
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    splits: list[tuple],
) -> float:
    """
    Per fold: train suite pada train_idx, evaluasi MAPE h=1 pada test_idx.
    Return rata-rata MAPE across folds.
    """
    fold_mapes = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        suite = train_direct_suite(df_feat, train_idx, feature_cols)

        X_test = df_feat.loc[test_idx, feature_cols]
        y_true = df_feat["close"].loc[test_idx].values

        # evaluasi h=1 saja per fold (representatif, cepat)
        scaler = suite["scaler"][0]
        model  = suite["mean"][0]
        X_sc   = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
        y_pred = model.predict(X_sc)

        valid = ~np.isnan(y_true)
        mape  = mean_absolute_percentage_error(y_true[valid], y_pred[valid]) * 100
        fold_mapes.append(mape)
        print(f"  Fold {fold}/{N_FOLDS} — train: {len(train_idx)} | test: {len(test_idx)} | MAPE h=1: {mape:.2f}%")

    avg = float(np.mean(fold_mapes))
    print(f"  Walk-forward avg MAPE: {avg:.2f}%")
    return avg


# ── 8. Final Forecast ─────────────────────────────────────────────────────────

def forecast_with_recursive_externals(
    df: pd.DataFrame,
    df_feat: pd.DataFrame,
    suite: dict,
    ar_models: dict,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    1. Prediksi external features untuk FORECAST_DAYS ke depan (recursive AR)
    2. Per horizon h: bangun feature row dengan external lag yang sudah diprediksi
    3. Predict gold price untuk setiap horizon
    """
    last_date    = df_feat.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS)

    ext_forecast = recursive_forecast_externals(df, ar_models, steps=FORECAST_DAYS)

    # seed dari baris terakhir aktual
    last_row  = df_feat[feature_cols].iloc[-1].to_dict()
    gold_hist = list(df["close"].values)

    means, lowers, uppers = [], [], []

    for h in range(FORECAST_DAYS):
        # rebuild feature row untuk horizon h
        # gold lags: pakai prediksi sebelumnya jika sudah ada
        row = last_row.copy()

        for lag in LAG_DAYS:
            if len(means) >= lag:
                row[f"lag_{lag}"] = means[-lag]
            else:
                row[f"lag_{lag}"] = gold_hist[-(lag - len(means))]

        # rolling stats: approximate dari means yang sudah terkumpul
        recent = (means + gold_hist)[-max(ROLL_WINDOWS):]
        for w in ROLL_WINDOWS:
            window_vals = recent[-w:]
            row[f"roll_mean_{w}"] = float(np.mean(window_vals))
            row[f"roll_std_{w}"]  = float(np.std(window_vals))

        if len(means) >= 1:
            row["return_1d"] = (means[-1] - (means[-2] if len(means) >= 2 else gold_hist[-1])) / (means[-2] if len(means) >= 2 else gold_hist[-1])
        if len(means) >= 5:
            row["return_5d"] = (means[-1] - means[-5]) / means[-5]
        row["volatility"] = float(np.std([(means[i] - means[i-1]) / means[i-1]
                                           for i in range(1, min(10, len(means)))])) if len(means) > 1 else last_row["volatility"]

        # momentum slope: gunakan prediksi means terkini
        recent_prices = (gold_hist + means)[-(max(10, 1)):]
        if len(recent_prices) >= 5:
            x5 = np.arange(5)
            y5 = recent_prices[-5:]
            row["momentum_slope_5"] = float(np.polyfit(x5, y5, 1)[0] / np.mean(y5))
        if len(recent_prices) >= 10:
            x10 = np.arange(10)
            y10 = recent_prices[-10:]
            row["momentum_slope_10"] = float(np.polyfit(x10, y10, 1)[0] / np.mean(y10))

        # external features: pakai prediksi AR
        for name in ar_models:
            for lag in EXTERNAL_LAGS:
                col = f"{name}_lag{lag}"
                if col in row:
                    idx = h - lag
                    if idx >= 0:
                        row[col] = ext_forecast[name].iloc[idx]
            ret_col = f"{name}_return"
            if ret_col in row and h >= 1:
                prev = ext_forecast[name].iloc[h-1] if h >= 1 else df[name].iloc[-1]
                curr = ext_forecast[name].iloc[h]
                row[ret_col] = (curr - prev) / prev if prev != 0 else 0.0

        X_df = pd.DataFrame([row])[feature_cols]
        scaler = suite["scaler"][h]
        X_sc   = pd.DataFrame(scaler.transform(X_df), columns=feature_cols)

        means.append(float(suite["mean"][h].predict(X_sc)[0]))
        lowers.append(float(suite["lower"][h].predict(X_sc)[0]))
        uppers.append(float(suite["upper"][h].predict(X_sc)[0]))

    return pd.DataFrame({
        "tanggal": future_dates,
        "prediksi_idr_gram": np.round(means).astype(int),
        "lower_95":          np.round(lowers).astype(int),
        "upper_95":          np.round(uppers).astype(int),
    })


# ── 9. Visualization ──────────────────────────────────────────────────────────


def fmt_idr(value: float, _=None) -> str:
    return f"Rp {value / 1_000_000:.2f}M"


def plot_forecast(
    df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    wf_mape: float,
    output_path: str,
) -> None:
    import datetime

    last_price     = df["close"].iloc[-1]
    last_date      = df.index[-1]
    pred_end       = forecast_df["prediksi_idr_gram"].iloc[-1]
    pred_lower_end = forecast_df["lower_95"].iloc[-1]
    pred_upper_end = forecast_df["upper_95"].iloc[-1]

    hist_max       = df["close"].max()
    hist_min       = df["close"].min()
    hist_max_date  = df["close"].idxmax()
    hist_min_date  = df["close"].idxmin()

    bridge_dates  = [last_date] + list(forecast_df["tanggal"])
    bridge_mean   = [last_price] + list(forecast_df["prediksi_idr_gram"])
    bridge_lower  = [last_price] + list(forecast_df["lower_95"])
    bridge_upper  = [last_price] + list(forecast_df["upper_95"])

    # figure: leave bottom margin for disclaimer
    fig = plt.figure(figsize=(14, 6))
    ax  = fig.add_axes([0.06, 0.18, 0.91, 0.68])

    # background & style
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.tick_params(colors="#666666", labelsize=9)
    ax.yaxis.label.set_color("#666666")
    ax.xaxis.label.set_color("#666666")
    ax.grid(axis="y", color="#EBEBEB", linewidth=0.7, zorder=0)
    ax.grid(axis="x", visible=False)

    # actual price
    ax.plot(df.index, df["close"],
            color="#1A1A2E", linewidth=1.1, label="Actual Price", zorder=3)

    # forecast background zone
    ax.axvspan(forecast_df["tanggal"].iloc[0], forecast_df["tanggal"].iloc[-1],
               color=ACCENT_COLOR, alpha=0.05, zorder=0)

    # CI 95% band
    ax.fill_between(bridge_dates, bridge_lower, bridge_upper,
                    color=ACCENT_COLOR, alpha=0.15, label="95% CI", zorder=2)

    # forecast line
    ax.plot(bridge_dates, bridge_mean,
            color=ACCENT_COLOR, linewidth=2.2, label="3-Month Forecast", zorder=4)

    # cutoff vertical line
    ax.axvline(last_date, color="#AAAAAA", linestyle="--", linewidth=1, zorder=1)

    # ── Annotations ──────────────────────────────────────────────────────────

    # current price
    ax.annotate(f"Now\n{fmt_idr(last_price)}",
                xy=(last_date, last_price),
                xytext=(-58, 18), textcoords="offset points",
                fontsize=7.5, color="#444444",
                arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.8))

    # forecast end
    ax.annotate(
        f"{fmt_idr(pred_end)}\n[{pred_lower_end/1e6:.2f}–{pred_upper_end/1e6:.2f}M]",
        xy=(forecast_df["tanggal"].iloc[-1], pred_end),
        xytext=(8, 0), textcoords="offset points",
        fontsize=8, color=ACCENT_COLOR, fontweight="bold", va="center",
    )

    # historical max
    ax.annotate(f"↑ All-time high\n{fmt_idr(hist_max)}",
                xy=(hist_max_date, hist_max),
                xytext=(0, 14), textcoords="offset points",
                fontsize=7, color="#CC3333", ha="center",
                arrowprops=dict(arrowstyle="-", color="#CC3333", lw=0.7))

    # historical min
    ax.annotate(f"↓ Period low\n{fmt_idr(hist_min)}",
                xy=(hist_min_date, hist_min),
                xytext=(0, -28), textcoords="offset points",
                fontsize=7, color="#777777", ha="center",
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.7))

    # ── Axis formatting ───────────────────────────────────────────────────────
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_idr))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    # ── Title & subtitle ──────────────────────────────────────────────────────
    fig.text(0.5, 0.94,
             "Gold Price Forecast — Next 3 Months",
             ha="center", fontsize=13, fontweight="bold", color="#1A1A2E")

    fig.text(0.5, 0.90,
             f"GC=F (IDR/gram)  ·  XGBoost + DXY / US10Y / VIX  ·  "
             f"Walk-Forward MAPE: {wf_mape:.1f}%  ·  "
             f"Generated: {datetime.date.today().strftime('%d %b %Y')}",
             ha="center", fontsize=8, color="#888888")

    # ── Legend ────────────────────────────────────────────────────────────────
    leg = ax.legend(loc="upper left", frameon=True, fontsize=8.5,
                    framealpha=0.90, edgecolor="#DDDDDD")
    leg.get_frame().set_facecolor(BACKGROUND_COLOR)

    # ── Watermark ─────────────────────────────────────────────────────────────
    fig.text(0.98, 0.955, CREDIT_TEXT,
             ha="right", va="top", fontsize=8,
             color="#AAAAAA", fontstyle="italic")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    fig.text(0.5, 0.04,
             "⚠  This chart is for educational purposes only and does not constitute financial advice. "
             "Gold prices are influenced by geopolitical, macroeconomic, and currency factors not captured by this model. "
             "Past performance does not guarantee future results.",
             ha="center", va="top", fontsize=7, color="#AAAAAA",
             wrap=True, style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0",
                       edgecolor="#DDDDDD", alpha=0.8))

    fig.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=BACKGROUND_COLOR)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── 10. Export ────────────────────────────────────────────────────────────────

def export_csv(forecast_df: pd.DataFrame, output_path: str) -> None:
    forecast_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(forecast_df.head(10).to_string(index=False))


# ── 11. Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Fetch Data ===")
    usd_idr_rate = fetch_usd_idr_rate()
    df_raw       = fetch_all_data()
    df_raw       = convert_gold_to_idr(df_raw, usd_idr_rate)

    print(f"Rows  : {len(df_raw)}")
    print(f"Range : {df_raw.index[0].date()} → {df_raw.index[-1].date()}")
    print(f"Harga terakhir: Rp {df_raw['close'].iloc[-1]:,.0f}/gram")

    print("\n=== Feature Engineering ===")
    df_feat      = build_features(df_raw)
    feature_cols = get_feature_cols(df_feat)
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Rows setelah dropna: {len(df_feat)}")

    print("\n=== Walk-Forward Validation (5 fold) ===")
    splits   = walk_forward_splits(df_feat, N_FOLDS)
    wf_mape  = walk_forward_evaluate(df_feat, feature_cols, splits)

    print("\n=== Train AR Models (External Features) ===")
    ar_models = train_ar_models(df_raw, df_feat.index)
    print(f"  AR models trained: {list(ar_models.keys())}")

    print(f"\n=== Train Final XGBoost Suite ({FORECAST_DAYS * 3} models) ===")
    final_suite = train_direct_suite(df_feat, df_feat.index, feature_cols)
    for h in [1, 10, 20, 30]:
        print(f"  Horizon {h} ready")

    print("\n=== Recursive Forecast (3 Bulan) ===")
    forecast_df = forecast_with_recursive_externals(
        df_raw, df_feat, final_suite, ar_models, feature_cols
    )

    print("\n=== Simpan Output ===")
    plot_forecast(df_raw, forecast_df, wf_mape, "gold_forecast_v2.png")
    export_csv(forecast_df, "gold_forecast_v2_3bulan.csv")

    print("\nSelesai.")


if __name__ == "__main__":
    main()
