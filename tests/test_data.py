import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.data import convert_gold_to_idr, TROY_OZ_TO_GRAM


def test_convert_gold_to_idr():
    df = pd.DataFrame({"gold_usd_oz": [2000.0, 2100.0, 2050.0]})
    rate = 16000.0
    result = convert_gold_to_idr(df, rate)
    expected = (df["gold_usd_oz"] / TROY_OZ_TO_GRAM) * rate
    pd.testing.assert_series_equal(result["close"], expected, check_names=False)
    assert "gold_usd_oz" in result.columns


@patch("src.data.yf.Ticker")
def test_fetch_usd_idr_rate(mock_ticker):
    mock_ticker.return_value.fast_info = {"last_price": 16250.0}
    from src.data import fetch_usd_idr_rate
    assert fetch_usd_idr_rate() == 16250.0


@patch("src.data.yf.download")
def test_fetch_all_data_success(mock_download):
    dates = pd.bdate_range("2024-01-01", periods=10)
    gold_close = pd.Series(np.linspace(2000, 2100, 10), index=dates, name="Close")
    dxy_close = pd.Series(np.linspace(104, 105, 10), index=dates, name="Close")

    def side_effect(ticker, **kwargs):
        raw = pd.DataFrame({"Close": gold_close if ticker == "GC=F" else dxy_close})
        raw.columns = pd.MultiIndex.from_product([raw.columns, [ticker]])
        return raw

    mock_download.side_effect = side_effect

    from src.data import fetch_all_data, EXTERNAL_TICKERS
    df, status = fetch_all_data()

    assert "gold_usd_oz" in df.columns
    assert "dxy" in df.columns
    assert status["dxy"] is True
    assert len(df) > 0


@patch("src.data.yf.download")
def test_fetch_all_data_partial_failure(mock_download):
    dates = pd.bdate_range("2024-01-01", periods=10)
    gold_close = pd.Series(np.linspace(2000, 2100, 10), index=dates, name="Close")

    def side_effect(ticker, **kwargs):
        if ticker == "GC=F":
            raw = pd.DataFrame({"Close": gold_close})
            raw.columns = pd.MultiIndex.from_product([raw.columns, [ticker]])
            return raw
        raise Exception(f"Failed to fetch {ticker}")

    mock_download.side_effect = side_effect

    from src.data import fetch_all_data, EXTERNAL_TICKERS
    df, status = fetch_all_data()

    assert "gold_usd_oz" in df.columns
    for name in EXTERNAL_TICKERS:
        assert status[name] is False
        assert name not in df.columns
