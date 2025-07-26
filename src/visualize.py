"""
Streamlit dashboard for visualising financial price, sentiment and prediction data.

This module provides an interactive web application built with Streamlit. It
loads the engineered feature dataset produced by the ETL and feature
engineering pipeline, applies the trained machine‑learning model to generate
predictions, and renders a dashboard with line charts and summary statistics.

The dashboard includes sidebar controls for selecting a date range and
displays three main panels: the historical stock price (Close), the
news‑derived sentiment score and the model's predicted direction of the next
day's return. Summary statistics of the filtered data are also shown.

To run the dashboard locally execute:

    streamlit run src/visualize.py

The script expects the processed feature CSV to be located at
``data/processed/features.csv`` relative to the project root and a trained
model pickle at ``models/model.pkl``. If the model file is not present,
predictions will be omitted but the price and sentiment charts will still
render.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Optional, Tuple, List

import pandas as pd  # type: ignore
import streamlit as st  # type: ignore

# Import model utilities for loading and prediction
try:
    from model import load_model  # type: ignore
except Exception:
    # Fallback if import fails when running from different working directory
    from .model import load_model  # type: ignore


# Configure a module‑level logger
logger = logging.getLogger(__name__)

# Paths to data and models relative to this file
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FEATURES_PATH = os.path.join(PROCESSED_DIR, 'features.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pkl')


def load_features(path: Optional[str] = None) -> pd.DataFrame:
    """Load the engineered feature set from a CSV file.

    Args:
        path: Optional path to the features CSV. Defaults to
            ``FEATURES_PATH``.

    Returns:
        A pandas DataFrame with a parsed ``date`` column.
    """
    features_file = path or FEATURES_PATH
    logger.info("Loading features from %s", features_file)
    if not os.path.exists(features_file):
        logger.error("Features file not found: %s", features_file)
        raise FileNotFoundError(f"Features file not found: {features_file}")
    try:
        df = pd.read_csv(features_file, parse_dates=['date'])
    except Exception as exc:
        logger.exception("Failed to read features from %s", features_file)
        raise ValueError(f"Failed to read features: {exc}")
    return df


def load_trained_model(path: Optional[str] = None):
    """Load a trained model from disk.

    Args:
        path: Optional path to the pickle file. Defaults to ``MODEL_PATH``.

    Returns:
        The deserialised model object or ``None`` if loading fails.
    """
    model_file = path or MODEL_PATH
    if not os.path.exists(model_file):
        logger.warning("Model file not found: %s", model_file)
        return None
    try:
        model = load_model(model_file)
    except Exception as exc:
        logger.exception("Failed to load model from %s", model_file)
        return None
    return model


def compute_predictions(df: pd.DataFrame, model) -> pd.Series:
    """Compute prediction labels for a feature DataFrame using a trained model.

    The function drops non‑feature columns and applies the model's ``predict``
    method to generate a binary class for each observation. If the model is
    ``None`` predictions of 0 are returned.

    Args:
        df: DataFrame containing engineered features.
        model: Trained model implementing ``predict``.

    Returns:
        A pandas Series of integer predictions.
    """
    if model is None:
        logger.warning("No trained model provided; returning default predictions of 0")
        return pd.Series([0] * len(df), index=df.index)
    # Determine feature columns: exclude date, returns and sentiment
    feature_cols = [col for col in df.columns if col not in ['date', 'returns', 'sentiment']]
    X = df[feature_cols]
    try:
        preds = model.predict(X)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        return pd.Series([0] * len(df), index=df.index)
    return pd.Series(preds, index=df.index)


def filter_by_date(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Filter a DataFrame between two dates inclusive.

    Args:
        df: DataFrame with a ``date`` column.
        start: Start date.
        end: End date.

    Returns:
        Filtered DataFrame.
    """
    mask = (df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))
    return df.loc[mask].copy()


def main() -> None:
    """Entry point for running the Streamlit dashboard."""
    st.set_page_config(page_title="Finance News Sentiment Dashboard", layout="wide")
    st.title("Finance News Sentiment Dashboard")

    # Load data and model
    try:
        df = load_features()
    except Exception as exc:
        st.error(f"Failed to load features: {exc}")
        return
    model = load_trained_model()

    # Sidebar: date range selection
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    start_date, end_date = st.sidebar.date_input(
        "Select date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )
    # Ensure start_date and end_date are dates, not a tuple of tuples
    if isinstance(start_date, tuple):
        start_date = start_date[0]
    if isinstance(end_date, tuple):
        end_date = end_date[-1]

    # Filter data by date
    df_filtered = filter_by_date(df, start_date, end_date)

    # Compute predictions if model is available
    if model is not None:
        df_filtered['prediction'] = compute_predictions(df_filtered, model)

    # Display line charts
    st.subheader("Stock Price (Close)")
    st.line_chart(df_filtered.set_index('date')['Close'])

    st.subheader("Sentiment Score")
    st.line_chart(df_filtered.set_index('date')['sentiment'])

    if model is not None:
        st.subheader("Predicted Direction (1=up, 0=down)")
        st.line_chart(df_filtered.set_index('date')['prediction'])
    else:
        st.info("Trained model not available – predictions are not displayed.")

    # Summary statistics
    st.subheader("Summary Statistics for Selected Period")
    st.write(df_filtered[['Close', 'sentiment', 'returns']].describe())


if __name__ == '__main__':
    main()
