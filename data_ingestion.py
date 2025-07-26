import os
import logging
from datetime import datetime
import yfinance as yf
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

def download_stock_price(ticker: str = 'SPY', period: str = '5y', interval: str = '1d') -> str:
    """
    Downloads historical stock price data for a given ticker and saves as CSV.
    Args:
        ticker (str): Stock ticker symbol (default SPY).
        period (str): Data period (e.g., '5y' for 5 years).
        interval (str): Data interval (e.g., '1d' daily).
    Returns:
        str: Filepath of saved CSV.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    filepath = os.path.join(RAW_DATA_DIR, 'price_data.csv')
    
    logging.info(f"Downloading stock price data for {ticker}...")
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            raise ValueError("No data downloaded. Check ticker and connection.")
        data.to_csv(filepath)
        logging.info(f"Saved stock price data to {filepath}")
    except Exception as e:
        logging.error(f"Failed to download stock price data: {e}")
        raise e
    return filepath

def download_fnspid_placeholder():
    """
    Placeholder function for FNSPID news sentiment data ingestion.
    Real dataset requires manual download or authentication.
    Add instructions to README for obtaining this data.
    """
    logging.info("FNSPID dataset requires manual download due to licensing. See README for instructions.")
    # Example: Save placeholder empty CSV to keep pipeline intact
    filepath = os.path.join(RAW_DATA_DIR, 'fnspid_news_sentiment.csv')
    if not os.path.exists(filepath):
        pd.DataFrame().to_csv(filepath)
        logging.info(f"Created placeholder file {filepath}")

if __name__ == "__main__":
    download_stock_price()
    download_fnspid_placeholder()
