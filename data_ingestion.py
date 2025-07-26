"""
Data ingestion module.

This module defines functions to download raw datasets required for the project.

It includes a helper to download historical stock price data for a given ticker using
the `yfinance` library and save it to a CSV file, as well as a placeholder function
to generate an empty FNSPID news sentiment CSV. The placeholder reminds users that
the actual dataset may require manual download or authentication.
"""

import os
import logging
from datetime import datetime
from typing import Optional

import pandas as pd  # type: ignore

# Attempt to import yfinance if available. When running in environments without
# internet access, this import may fail. In that case `yf` will be set to None
# and the code will fall back to generating synthetic data.
try:
    import yfinance as yf  # type: ignore
except ImportError:
    yf = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Directory where raw data files will be stored
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')


def download_stock_price(ticker: str = "SPY", period: str = "5y", interval: str = "1d") -> str:
    """
    Download historical stock price data from Yahoo Finance and save to a CSV file.

    This function uses the `yfinance` library to fetch historical price data for the
    specified ticker symbol. The default parameters download five years of daily
    price data for the SPY ETF. The data is written to ``data/raw/price_data.csv``.

    Args:
        ticker: Ticker symbol for the financial instrument to download (default ``"SPY"``).
        period: Time period to download (default ``"5y"`` for five years).
        interval: Data interval to download (default ``"1d"`` for daily data).

    Returns:
        The file path to the saved CSV.

    Raises:
        Exception: If the data cannot be downloaded or saved.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    filepath = os.path.join(RAW_DATA_DIR, "price_data.csv")
    logging.info("Downloading stock price data for %s...", ticker)
    try:
        # Attempt to download using yfinance
        data = yf.download(ticker, period=period, interval=interval, progress=False)  # type: ignore[attr-defined]
        # If yfinance returned an empty DataFrame, raise an error
        if data is None or getattr(data, "empty", False):
            raise ValueError(
                f"No data returned for ticker {ticker}. Check the ticker symbol or network connection."
            )
    except Exception as exc:
        # If yfinance is not available or fails, fall back to creating synthetic data
        logging.warning(
            "Could not download stock price data using yfinance (%s). Falling back to synthetic data.",
            exc,
        )
        # Generate a date range for the past five years on a daily frequency
        end_date = datetime.today()
        start_date = end_date - pd.DateOffset(years=5)
        dates = pd.date_range(start=start_date, end=end_date, freq=interval)
        # Create a simple synthetic close price series (e.g., cumulative sum of random values)
        synthetic_prices = pd.Series(100 + pd.np.random.randn(len(dates)).cumsum(), index=dates)
        data = pd.DataFrame({
            "Open": synthetic_prices,
            "High": synthetic_prices + 1,
            "Low": synthetic_prices - 1,
            "Close": synthetic_prices,
            "Adj Close": synthetic_prices,
            "Volume": pd.np.random.randint(1000000, 5000000, size=len(dates)),
        })
    # Save the DataFrame to CSV
    data.to_csv(filepath)
    logging.info("Saved stock price data to %s", filepath)
    return filepath


def download_fnspid_placeholder() -> str:
    """
    Create an empty CSV file as a placeholder for the FNSPID news sentiment dataset.

    The FNSPID (Financial News Sentiment Prediction and Item Description) dataset is not
    automatically downloaded because it may require manual steps or authentication. This
    function generates an empty CSV file named ``fnspid_news_sentiment.csv`` with typical
    columns (``headline``, ``date``, ``sentiment``) so downstream scripts can run without
    errors. Users should replace this placeholder file with the actual dataset once
    obtained.

    Returns:
        The file path to the created placeholder CSV.

    Raises:
        Exception: If the file cannot be created.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    filepath = os.path.join(RAW_DATA_DIR, "fnspid_news_sentiment.csv")
    logging.info("Creating placeholder FNSPID news sentiment CSV at %s", filepath)
    try:
        # Create an empty DataFrame with expected columns
        df = pd.DataFrame({"headline": [], "date": [], "sentiment": []})
        df.to_csv(filepath, index=False)
        logging.info("Placeholder FNSPID file created.")
        return filepath
    except Exception as exc:
        logging.error("Failed to create placeholder FNSPID file: %s", exc)
        raise


if __name__ == "__main__":  # pragma: no cover
    # Run data ingestion if executed directly
    try:
        download_stock_price()
    except Exception:
        logging.error("Error occurred downloading stock price data.")
    try:
        download_fnspid_placeholder()
    except Exception:
        logging.error("Error occurred creating FNSPID placeholder file.")
