"""Unit tests for the ETL module.

These tests verify the correctness of the key preprocessing and merging
functions defined in ``src/etl.py``. The tests use small, synthetic
DataFrames to ensure the functions behave as expected without relying on
external data files.
"""

import unittest
import pandas as pd  # type: ignore

# Import functions under test from the ETL module. Since the module is in
# ``src`` relative to the project root, we adjust the import path when
# running the tests from the repository root. Python adds the project
# root to sys.path when using pytest/unittest via discovery.
from src.etl import (
    preprocess_price_data,
    preprocess_news_data,
    merge_data,
)


class TestETLPipeline(unittest.TestCase):
    """Test cases for the ETL preprocessing and merging functions."""

    def test_preprocess_price_data(self) -> None:
        """Verify that price data are reindexed, forward filled and returns computed."""
        # Create a small DataFrame with a missing day and a single column 'Close'
        dates = pd.to_datetime(["2025-01-01", "2025-01-03"])
        df_raw = pd.DataFrame({
            "Open": [100, 103],
            "High": [101, 104],
            "Low": [99, 102],
            "Close": [100, 103],
            "Adj Close": [100, 103],
            "Volume": [1000, 2000],
        }, index=dates)
        df_clean = preprocess_price_data(df_raw)
        # Expect three rows after reindexing to daily frequency (Jan 1, 2, 3)
        self.assertEqual(len(df_clean), 3)
        # Verify that missing day (Jan 2) is forward filled from Jan 1
        jan2_row = df_clean[df_clean['date'] == pd.Timestamp("2025-01-02")].iloc[0]
        self.assertEqual(jan2_row['Close'], 100)
        # Verify returns column exists and first value is zero
        self.assertIn('returns', df_clean.columns)
        self.assertEqual(df_clean.iloc[0]['returns'], 0)

    def test_preprocess_news_data(self) -> None:
        """Verify that news data are aggregated by date and missing values handled."""
        # Create sample news data with duplicate dates and missing sentiment
        df_raw = pd.DataFrame({
            "headline": ["A", "B", "C"],
            "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
            "sentiment": [0.5, None, "0.1"],
        })
        df_clean = preprocess_news_data(df_raw)
        # Two unique dates remain after aggregation
        self.assertEqual(len(df_clean), 2)
        # The sentiment for 2025-01-01 should be the average of 0.5 and 0 (missing filled with 0)
        sentiment_jan1 = df_clean[df_clean['date'] == pd.Timestamp("2025-01-01")]['sentiment'].iloc[0]
        self.assertAlmostEqual(sentiment_jan1, 0.25, places=2)
        # The sentiment for 2025-01-02 should be 0.1
        sentiment_jan2 = df_clean[df_clean['date'] == pd.Timestamp("2025-01-02")]['sentiment'].iloc[0]
        self.assertAlmostEqual(sentiment_jan2, 0.1, places=2)

    def test_merge_data(self) -> None:
        """Verify that merging fills missing sentiment values with 0."""
        price_df = pd.DataFrame({
            'date': pd.to_datetime(["2025-01-01", "2025-01-02"]),
            'Close': [100, 101],
            'returns': [0.0, 0.01],
        })
        news_df = pd.DataFrame({
            'date': pd.to_datetime(["2025-01-01"]),
            'sentiment': [0.5],
        })
        merged = merge_data(price_df, news_df)
        # There should be two rows after merge
        self.assertEqual(len(merged), 2)
        # The sentiment for 2025-01-01 should be 0.5 and for 2025-01-02 should be 0 (filled)
        self.assertEqual(merged.loc[merged['date'] == pd.Timestamp("2025-01-01"), 'sentiment'].iloc[0], 0.5)
        self.assertEqual(merged.loc[merged['date'] == pd.Timestamp("2025-01-02"), 'sentiment'].iloc[0], 0)


if __name__ == '__main__':
    unittest.main()
