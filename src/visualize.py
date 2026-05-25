import datetime

import pandas as pd
import plotly.graph_objects as go

ACCENT_COLOR = "#2A6FBF"  # biru


def plot_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame, wf_mape: float):
    last_price = df["close"].iloc[-1]
    last_date = df.index[-1]
    pred_end = forecast_df["prediksi_idr_gram"].iloc[-1]
    pred_lower_end = forecast_df["lower_95"].iloc[-1]
    pred_upper_end = forecast_df["upper_95"].iloc[-1]

    hist_max = df["close"].max()
    hist_max_date = df["close"].idxmax()
    hist_min = df["close"].min()
    hist_min_date = df["close"].idxmin()

    bridge_dates = [last_date] + list(forecast_df["tanggal"])
    bridge_mean = [last_price] + list(forecast_df["prediksi_idr_gram"])
    bridge_lower = [last_price] + list(forecast_df["lower_95"])
    bridge_upper = [last_price] + list(forecast_df["upper_95"])

    fig = go.Figure()

    # Invisible Upper bound just to calculate fill range downwards
    fig.add_trace(
        go.Scatter(
            x=bridge_dates,
            y=bridge_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Lower Bound filling upwards to Upper Bound creating the Interval band
    fig.add_trace(
        go.Scatter(
            x=bridge_dates,
            y=bridge_lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(42, 111, 191, 0.15)",
            name="95% Confidence Interval",
            hovertemplate="Lower: Rp %{y:,.0f}<br>Upper: Rp %{customdata:,.0f}<extra></extra>",
            customdata=bridge_upper,
        )
    )

    # Historical Data
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["close"],
            mode="lines",
            line=dict(color="gray", width=1.5),
            name="Actual Price",
            hovertemplate="%{x|%b %d, %Y}<br>Rp %{y:,.0f}<extra></extra>",
        )
    )

    # Prediction Target Line
    fig.add_trace(
        go.Scatter(
            x=bridge_dates,
            y=bridge_mean,
            mode="lines",
            line=dict(color=ACCENT_COLOR, width=3),
            name="3-Month Forecast",
            hovertemplate="%{x|%b %d, %Y}<br>Rp %{y:,.0f}<extra></extra>",
        )
    )

    # Add marker for NOW
    fig.add_annotation(
        x=last_date,
        y=last_price,
        text=f"Now<br>Rp {last_price / 1_000_000:.2f}M",
        showarrow=True,
        arrowhead=1,
        ax=-40,
        ay=30,
    )

    # Add marker for End Prediction
    fig.add_annotation(
        x=forecast_df["tanggal"].iloc[-1],
        y=pred_end,
        text=f"<b>Rp {pred_end / 1_000_000:.2f}M</b><br>[{pred_lower_end / 1_000_000:.2f}M – {pred_upper_end / 1_000_000:.2f}M]",
        showarrow=True,
        arrowhead=1,
        arrowcolor=ACCENT_COLOR,
        ax=30,
        ay=0,
        font=dict(color=ACCENT_COLOR),
    )

    # Add marker for High / Lows
    fig.add_annotation(
        x=hist_max_date,
        y=hist_max,
        text=f"↑ High<br>Rp {hist_max / 1_000_000:.2f}M",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-30,
        font=dict(color="#d44a4a"),
        arrowcolor="#d44a4a",
    )
    fig.add_annotation(
        x=hist_min_date,
        y=hist_min,
        text=f"↓ Low<br>Rp {hist_min / 1_000_000:.2f}M",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=30,
        font=dict(color="gray"),
        arrowcolor="gray",
    )

    # Cutoff line representing "Today"
    fig.add_vline(
        x=last_date.timestamp() * 1000, line_width=1, line_dash="dash", line_color="gray"
    )

    sub = f"GC=F (IDR/gram)  |  XGBoost + DXY / US10Y / VIX  |  Walk-Forward MAPE: {wf_mape:.1f}%  |  Generated: {datetime.date.today().strftime('%d %b %Y')}"

    fig.update_layout(
        title=dict(
            text=f"<b>Gold Price Forecast — Next 3 Months</b><br><sup>{sub}</sup>",
            x=0.5,
            xanchor="center",
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=100, b=40),
        xaxis_title="Date",
        yaxis_title="Price (IDR / gram)",
    )

    fig.update_yaxes(tickprefix="Rp ")

    return fig
