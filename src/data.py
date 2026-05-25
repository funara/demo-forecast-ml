import pandas as pd
import yfinance as yf

TROY_OZ_TO_GRAM = 31.1035
TRAIN_START = "2024-01-01"
EXTERNAL_TICKERS = {
    "dxy": "DX-Y.NYB",  # US Dollar Index
    "us10y": "^TNX",  # US 10-Year Treasury Yield
    "vix": "^VIX",  # CBOE Volatility Index
}


def fetch_usd_idr_rate() -> float:
    rate = yf.Ticker("USDIDR=X").fast_info["last_price"]
    return float(rate)


def fetch_series(ticker: str) -> pd.Series:
    raw = yf.download(ticker, start=TRAIN_START, auto_adjust=True, progress=False)
    raw.columns = raw.columns.get_level_values(0)
    series = raw["Close"].copy()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return series.dropna().rename(ticker)


def fetch_all_data() -> tuple[pd.DataFrame, dict[str, bool]]:
    """Fetch gold + external tickers, return (DataFrame, per-ticker success status)."""
    gold = fetch_series("GC=F").rename("gold_usd_oz")

    status = {}
    externals = {}
    for name, ticker in EXTERNAL_TICKERS.items():
        try:
            externals[name] = fetch_series(ticker)
            status[name] = True
        except Exception:
            status[name] = False

    df = pd.DataFrame({"gold_usd_oz": gold})
    for name, series in externals.items():
        df[name] = series

    df = df.ffill().dropna()
    return df, status


def convert_gold_to_idr(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    result = df.copy()
    result["close"] = (result["gold_usd_oz"] / TROY_OZ_TO_GRAM) * rate
    return result
