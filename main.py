import streamlit as st
import pandas as pd
import joblib
import os
import json
from src.data import fetch_usd_idr_rate, fetch_all_data, convert_gold_to_idr
from src.features import build_features, get_feature_cols
from src.model import walk_forward_splits, walk_forward_evaluate, train_ar_models, train_direct_suite, forecast_with_recursive_externals, FORECAST_DAYS, N_FOLDS
from src.visualize import plot_forecast

st.set_page_config(page_title="Gold Price Forecasting", layout="wide")

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


def execute_training():
    with st.spinner("Fetching full historical data..."):
        usd_idr_rate = fetch_usd_idr_rate()
        df_raw = fetch_all_data()
        df_raw = convert_gold_to_idr(df_raw, usd_idr_rate)
        
    with st.spinner("Building extensive features (lags, rolling averages, momentum)..."):
        df_feat = build_features(df_raw)
        feature_cols = get_feature_cols(df_feat)
        
    with st.spinner(f"Evaluating Model via Walk-Forward Validation ({N_FOLDS}-fold)..."):
        splits = walk_forward_splits(df_feat, N_FOLDS)
        wf_mape, mapes = walk_forward_evaluate(df_feat, feature_cols, splits)
        
    with st.spinner("Training AR models for external futures..."):
        ar_models = train_ar_models(df_raw, df_feat.index)
        
    with st.spinner(f"Training direct multi-step XGBoost suite ({FORECAST_DAYS} horizons)..."):
        final_suite = train_direct_suite(df_feat, df_feat.index, feature_cols)
        
    with st.spinner("Saving models to disk..."):
        metrics = {"wf_mape": wf_mape}
        save_models(ar_models, final_suite, metrics)
        
    st.success(f"Models Retrained & Saved Successfully! Walk-Forward MAPE: {wf_mape:.2f}%")
    st.session_state.execute_retrain = False
    st.balloons()

def main():
    st.title("Gold Price Forecasting via XGBoost")
    st.write("Forecast gold prices in IDR/gram using macroeconomic indicators.")

    # Top level actions
    col_action1, col_action2, col_action3, _ = st.columns([2, 2, 2, 4])
    with col_action1:
        refresh_clicked = st.button("Refresh Forecast & Data", type="primary", use_container_width=True)
    with col_action2:
        if st.button("Retrain XGBoost Models", use_container_width=True):
            retrain_dialog()
    with col_action3:
        if st.button("ℹ️ Model Information", use_container_width=True):
            info_dialog()

    if "execute_retrain" in st.session_state and st.session_state.execute_retrain:
        execute_training()

    st.divider()

    ar_models, final_suite, metrics = load_models()
    
    if ar_models is None:
        st.warning("⚠️ No trained models found! Please click 'Retrain XGBoost Models' to initialize the forecasting engine.")
        return

    # If the user clicks refresh
    if refresh_clicked:
        with st.spinner("Fetching latest data directly from yfinance..."):
            usd_idr_rate = fetch_usd_idr_rate()
            df_raw = fetch_all_data()
            df_raw = convert_gold_to_idr(df_raw, usd_idr_rate)
            
        with st.spinner("Building features for inference..."):
            df_feat = build_features(df_raw)
            feature_cols = get_feature_cols(df_feat)
            
        with st.spinner("Executing targeted recursive forecasting..."):
            forecast_df = forecast_with_recursive_externals(df_raw, df_feat, final_suite, ar_models, feature_cols)
            
        wf_mape = metrics.get("wf_mape", 0.0)
        
        # Save to session state so it persists
        st.session_state.dashboard_data = {
            "df_raw": df_raw,
            "forecast_df": forecast_df,
            "wf_mape": wf_mape
        }

    # Render dashboard if data exists in state
    if "dashboard_data" in st.session_state:
        data = st.session_state.dashboard_data
        df_raw = data["df_raw"]
        forecast_df = data["forecast_df"]
        wf_mape = data["wf_mape"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Gold Price", f"Rp {df_raw['close'].iloc[-1]:,.0f}/g")
        c2.metric("Walk-Forward MAPE", f"{wf_mape:.2f}%")
        c3.metric(f"{FORECAST_DAYS}-Day Forecast Prediction", f"Rp {forecast_df['prediksi_idr_gram'].iloc[-1]:,.0f}/g")
        
        st.subheader("Forecast Chart")
        fig = plot_forecast(df_raw, forecast_df, wf_mape)
        st.plotly_chart(fig, use_container_width=True) # Swap matplotlib for plotly render
        
        st.subheader("Raw Forecast Data")
        st.dataframe(forecast_df)
        
        csv_data = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast Data", data=csv_data, file_name="gold_forecast.csv", mime="text/csv")
    else:
        st.info("👆 Click **Refresh Forecast & Data** to generate latest predictions.")

if __name__ == "__main__":
    main()
