"""
Feature engineering module.

This module implements routines to transform the merged market and
sentiment dataset into a rich set of features suitable for machine
learning models. It loads the processed merged data, computes a
variety of technical indicators and sentiment aggregates, creates lag
features, and writes the resulting feature matrix to disk.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

# Configure logger for this module
logger = logging.getLogger(__name__)

# Paths to data directories
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def load_merged_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load the merged dataset produced by the ETL pipeline.

    Args:
        path: Optional custom path to the CSV file. Defaults to
            ``data/processed/merged_data.csv`` relative to the project.

    Returns:
        A pandas DataFrame containing the merged data.

    Raises:
        FileNotFoundError: If the file cannot be found.
        ValueError: If the CSV cannot be parsed.
    """
    if path is None:
        path = os.path.join(PROCESSED_DIR, 'merged_data.csv')
    logger.info("Loading merged data from %s", path)
    if not os.path.exists(path):
        logger.error("Merged data file not found: %s", path)
        raise FileNotFoundError(f"Merged data file not found: {path}")
    try:
        df = pd.read_csv(path, parse_dates=['date'])
    except Exception as exc:
        logger.exception("Error reading merged data from %s", path)
        raise ValueError(f"Failed to read merged data: {exc}")
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) for a price series.

    Args:
        series: A pandas Series of prices.
        period: Window length for RSI calculation.

    Returns:
        A pandas Series containing the RSI values.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a set of features from the merged dataset.

    Features include moving averages, RSI, rolling sentiment averages and
    lagged values for returns and sentiment. Any rows with insufficient
    historical data for feature computation are dropped.

    Args:
        df: The merged DataFrame containing at least ``date``, ``Close``,
            ``returns`` and ``sentiment`` columns.

    Returns:
        A new DataFrame with engineered features and the original target
        information retained.
    """
    logger.info("Engineering features from merged data (%d rows)", len(df))
    df = df.sort_values('date').reset_index(drop=True)
    # Moving averages on closing price
    df['ma_7'] = df['Close'].rolling(window=7, min_periods=1).mean()
    df['ma_21'] = df['Close'].rolling(window=21, min_periods=1).mean()
    # RSI on closing price
    df['rsi_14'] = compute_rsi(df['Close'], period=14)
    # Rolling sentiment average
    df['sentiment_ma_7'] = df['sentiment'].rolling(window=7, min_periods=1).mean()
    # Lag features for sentiment and returns (1-day and 2-day lags)
    df['sentiment_lag1'] = df['sentiment'].shift(1)
    df['sentiment_lag2'] = df['sentiment'].shift(2)
    df['returns_lag1'] = df['returns'].shift(1)
    df['returns_lag2'] = df['returns'].shift(2)
    # Replace NaNs introduced by shifts/rolling with reasonable defaults (0)
    df.fillna(0, inplace=True)
    return df


def save_features(df: pd.DataFrame, path: Optional[str] = None) -> None:
    """Persist the engineered feature set to a CSV file.

    Args:
        df: DataFrame containing the engineered features.
        path: Optional path to save the CSV. Defaults to
            ``data/processed/features.csv``.

    Raises:
        IOError: If the file cannot be written.
    """
    if path is None:
        path = os.path.join(PROCESSED_DIR, 'features.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info("Saving features to %s", path)
    try:
        df.to_csv(path, index=False)
    except Exception as exc:
        logger.exception("Error saving features to %s", path)
        raise IOError(f"Failed to save features: {exc}")


def run_feature_engineering() -> None:
    """End-to-end routine to load, engineer and save features."""
    try:
        merged_df = load_merged_data()
        features_df = engineer_features(merged_df)
        save_features(features_df)
        logger.info("Feature engineering pipeline completed successfully.")
    except Exception as exc:
        logger.error("Feature engineering pipeline failed: %s", exc)
        raise


if __name__ == '__main__':  # pragma: no cover
    run_feature_engineering()
