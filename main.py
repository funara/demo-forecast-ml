import streamlit as st
import pandas as pd
import joblib
import os
import json
from src.data import fetch_usd_idr_rate, fetch_all_data, convert_gold_to_idr
from src.features import build_features, get_feature_cols
from src.model import walk_forward_splits, walk_forward_evaluate, train_ar_models, train_direct_suite, forecast_with_recursive_externals, FORECAST_DAYS, N_FOLDS
from src.visualize import plot_forecast

st.set_page_config(page_title="Gold Price Forecasting Dashboard", layout="wide", initial_sidebar_state="expanded")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
AR_MODELS_PATH = os.path.join(MODELS_DIR, "ar_models.joblib")
FINAL_SUITE_PATH = os.path.join(MODELS_DIR, "final_suite.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

def load_models():
    if os.path.exists(AR_MODELS_PATH) and os.path.exists(FINAL_SUITE_PATH) and os.path.exists(METRICS_PATH):
        ar_models = joblib.load(AR_MODELS_PATH)
        final_suite = joblib.load(FINAL_SUITE_PATH)
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        return ar_models, final_suite, metrics
    return None, None, None

def save_models(ar_models, final_suite, metrics):
    joblib.dump(ar_models, AR_MODELS_PATH)
    joblib.dump(final_suite, FINAL_SUITE_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)

@st.dialog("Confirm Model Retraining")
def retrain_dialog():
    st.warning("⚠️ Retraining will fetch full historical data and train 198 XGBoost horizons via Walk-Forward validation. This may take a minute or two.")
    st.write("Do you want to proceed?")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Cancel"):
            st.rerun()
    with col_b:
        if st.button("Confirm Retrain", type="primary"):
            st.session_state.execute_retrain = True
            st.rerun()

@st.dialog("About The Model")
def info_dialog():
    st.markdown("### Model Architecture")
    st.write("This forecasting engine uses an advanced **Direct Multi-Step Strategy** paired with nested **Auto-Regressive** engines.")
    st.markdown("""
    - **Base Algorithm:** XGBoost (Extreme Gradient Boosting Regression)
    - **Models Trained:** 198 independent models. 3 models (Median, Upper Quantile, Lower Quantile) for each of the 66 forecast horizons.
    - **Validation:** 5-Fold Walk-Forward expanding window backtesting.
    """)
    st.divider()
    st.markdown("### Data Features")
    st.write("We retrieve live market indicators from Yahoo Finance to understand macroeconomic context.")
    st.markdown("""
    - **GC=F (Gold Futures):** Target feature.
    - **DXY (US Dollar Index):** USD strength directly affects bullion pricing.
    - **^TNX (US 10-Year Yield):** Rates dictate opportunity costs of holding non-yielding gold.
    - **^VIX (Volatility Index):** Tracks fear, which drives safe-haven demand.
    - **USDIDR=X:** Live conversion rate mapping global prices to IDR/gram.
    - Features generated include lags (t-1, 2, 5, 10), momentum slopes, and standard deviation volatilities.
    """)
    st.divider()
    st.markdown("### Limitations & Disclaimer")
    st.info("⚠ **Not Financial Advice.** Gold prices are influenced heavily by sudden geopolitical events, banking crises, and central bank physical buying — variables that a purely quantitative time-series model cannot capture. Real-world accuracy relies on history repeating itself.")

def _show_ticker_warnings(fetch_status: dict[str, bool], sidebar_placeholder):
    failed = [name for name, ok in fetch_status.items() if not ok]
    if failed:
        sidebar_placeholder.warning(
            f"⚠️ External ticker(s) unavailable: {', '.join(failed)}. "
            "Model running without these features."
        )

def execute_training():
    with st.spinner("Fetching full historical data..."):
        usd_idr_rate = fetch_usd_idr_rate()
        df_raw, fetch_status = fetch_all_data()
        df_raw = convert_gold_to_idr(df_raw, usd_idr_rate)
        
    with st.spinner("Building extensive features..."):
        df_feat = build_features(df_raw)
        feature_cols = get_feature_cols(df_feat)
        
    with st.spinner(f"Evaluating Model via Walk-Forward Validation ({N_FOLDS}-fold)..."):
        splits = walk_forward_splits(df_feat, N_FOLDS)
        wf_mape, fold_mapes, horizon_mapes = walk_forward_evaluate(df_feat, feature_cols, splits)
        
    with st.spinner("Training AR models for external futures..."):
        ar_models = train_ar_models(df_raw, df_feat.index)
        
    with st.spinner(f"Training XGBoost suite ({FORECAST_DAYS} horizons)..."):
        final_suite = train_direct_suite(df_feat, df_feat.index, feature_cols)
        
    with st.spinner("Saving models to disk..."):
        metrics = {"wf_mape": wf_mape, "horizon_mapes": horizon_mapes, "feature_cols": feature_cols}
        save_models(ar_models, final_suite, metrics)
        
    st.sidebar.success(f"Models Retrained & Saved Successfully! Avg MAPE: {wf_mape:.2f}%")
    _show_ticker_warnings(fetch_status, st.sidebar)
    st.session_state.execute_retrain = False

def main():
    ar_models, final_suite, metrics = load_models()

    with st.sidebar:
        st.title("Settings & Actions")
        
        if ar_models is None:
            st.error("🔴 Engine Status: Models Missing")
            refresh_disabled = True
        else:
            st.success("🟢 Engine Status: Models Loaded")
            refresh_disabled = False

        st.subheader("Data Control")
        refresh_clicked = st.button("🔄 Refresh Data & Predict", type="primary", disabled=refresh_disabled, width='stretch')
        
        st.subheader("Model Lifecycle")
        if st.button("⚙️ Retrain XGBoost Models", width='stretch'):
            retrain_dialog()
            
        if st.button("ℹ️ Model Information", width='stretch'):
            info_dialog()

    if "execute_retrain" in st.session_state and st.session_state.execute_retrain:
        execute_training()
        ar_models, final_suite, metrics = load_models()

    st.title("Gold Price Forecasting Dashboard")

    if ar_models is None:
        st.warning("⚠️ No trained models found! Please navigate to the sidebar and click **'Retrain XGBoost Models'** to initialize the forecasting engine.")
        return

    auto_trigger = ("dashboard_data" not in st.session_state and ar_models is not None)

    ticker_warnings = st.sidebar.empty()

    if refresh_clicked or auto_trigger:
        with st.spinner("Fetching latest data..."):
            usd_idr_rate = fetch_usd_idr_rate()
            df_raw, fetch_status = fetch_all_data()
            df_raw = convert_gold_to_idr(df_raw, usd_idr_rate)
            _show_ticker_warnings(fetch_status, ticker_warnings)
            
        with st.spinner("Building features for inference..."):
            df_feat = build_features(df_raw)
            feature_cols = get_feature_cols(df_feat)

        trained_cols = metrics.get("feature_cols", [])
        if set(feature_cols) != set(trained_cols):
            missing = set(trained_cols) - set(feature_cols)
            ticker_warnings.warning(
                f"⚠️ Feature mismatch from training. Missing columns: {', '.join(sorted(missing))}. "
                "Retrain the model to incorporate the current data."
            )
        else:
            with st.spinner("Executing recursive forecasting..."):
                forecast_df = forecast_with_recursive_externals(df_raw, df_feat, final_suite, ar_models, feature_cols)

            wf_mape = metrics.get("wf_mape", 0.0)
            horizon_mapes_display = metrics.get("horizon_mapes", [])

            st.session_state.dashboard_data = {
                "df_raw": df_raw,
                "forecast_df": forecast_df,
                "wf_mape": wf_mape,
                "horizon_mapes": horizon_mapes_display,
            }

    if "dashboard_data" in st.session_state:
        data = st.session_state.dashboard_data
        df_raw = data["df_raw"]
        forecast_df = data["forecast_df"]
        wf_mape = data["wf_mape"]
        horizon_mapes_display = data.get("horizon_mapes", [])
        mape_66d = horizon_mapes_display[-1] if len(horizon_mapes_display) >= 66 else wf_mape

        tab_overview, tab_data = st.tabs(["📈 Forecast Overview", "🗃️ Raw Data Extract"])

        with tab_overview:
            c1, c2, c3 = st.columns(3)
            
            with c1:
                with st.container(border=True):
                    current_price = df_raw['close'].iloc[-1]
                    prev_price = df_raw['close'].iloc[-2]
                    delta_price = current_price - prev_price
                    st.metric("Current Gold Price", f"Rp {current_price:,.0f}/g", delta=f"{delta_price:,.0f} from yesterday")
            
            with c2:
                with st.container(border=True):
                    st.metric("66-Day Forecast MAPE", f"{mape_66d:.2f}%", delta="Historical Accuracy (Horizon 66)", delta_color="off")
            
            with c3:
                with st.container(border=True):
                    forecast_end = forecast_df['prediksi_idr_gram'].iloc[-1]
                    delta_forecast = forecast_end - current_price
                    st.metric("3-Month Forecast Prediction", f"Rp {forecast_end:,.0f}/g", delta=f"{delta_forecast:,.0f} projected trajectory")
            
            with st.container(border=True):
                fig = plot_forecast(df_raw, forecast_df, wf_mape)
                st.plotly_chart(fig, width='stretch')

        with tab_data:
            st.subheader("Detailed Forecasting Results")
            st.dataframe(forecast_df, width='stretch')
            
            csv_data = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Forecast Data", data=csv_data, file_name="gold_forecast.csv", mime="text/csv", width='content')
    else:
        st.info("👈 Click **Refresh Data & Predict** in the sidebar to load the dashboard.")

if __name__ == "__main__":
    main()
