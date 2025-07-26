# Finance News Sentiment Predictor

This project aims to build a system that ingests financial market data and news sentiment to predict movements in financial instruments. It leverages public data sources such as Yahoo Finance and the FNSPID (Financial News Sentiment Prediction and Item Description) dataset. The repository is organised to facilitate data ingestion, preprocessing, feature engineering, model training and evaluation, and exposing predictions via a web API.

## Data Sources

- **Stock Prices:** Historical price data is downloaded using the `yfinance` library. By default, the data ingestion script downloads five years of daily price data for the SPY ETF and saves it to `data/raw/price_data.csv`.
- **FNSPID Dataset:** The FNSPID dataset contains news articles labeled with sentiment information. Access to this dataset may require registration or authentication due to licensing restrictions. Please follow the instructions provided by the dataset provider to download the files manually. Once downloaded, place the dataset in the `data/raw/` directory so it can be processed by the ETL pipeline.
