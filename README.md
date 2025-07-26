# Finance News Sentiment Predictor

![Python Version](https://img.shields.io/badge/python-3.9-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build](https://img.shields.io/badge/build-passing-brightgreen)

This project explores whether **news sentiment can help predict short‑term market movements**.
It combines historical price data and time‑aligned news sentiment to train a classifier that
forecasts the next‑day direction of the S&P 500 ETF (SPY).  The repository demonstrates a
complete machine‑learning pipeline from data ingestion and preprocessing through feature
engineering, model training and evaluation, API serving and interactive visualisation.

## Objectives

* **Collect and unify heterogeneous data** – download market prices via *Yahoo Finance* and
  manually obtain the *FNSPID* financial news sentiment dataset.
* **Clean and merge data sources** – parse dates, handle missing values and align
  sentiment scores with corresponding trading days.
* **Engineer informative features** – compute moving averages, relative strength index (RSI),
  rolling sentiment aggregates and lagged features for both returns and sentiment.
* **Train a binary classifier** – predict whether the SPY will close up or down the next day.
* **Expose predictions via an API** – serve a REST endpoint using FastAPI and containerise it
  with Docker.
* **Provide an interactive dashboard** – visualise price, sentiment and model predictions
  over selectable date ranges using Streamlit.

## Tech Stack & Skills Demonstrated

* **Python 3.9** – main programming language.
* **Data Engineering** – `pandas` for ingestion, cleansing and transformation; `yfinance`
  for downloading price data.
* **Feature Engineering** – calculation of technical indicators (moving averages, RSI),
  time‑series lags and rolling statistics.
* **Machine Learning** – model implemented using `scikit‑learn` (Random Forest) but can be
  extended to deep learning (e.g. LSTM with `tensorflow`).
* **API Development** – `FastAPI` provides a production‑ready web service for serving
  predictions.
* **Data Visualisation** – `Streamlit` dashboards to explore features, sentiment and
  predictions interactively.
* **DevOps & Containerisation** – Dockerfile builds a lightweight container for serving
  the API; `tests/` contains unit tests for the ETL pipeline.

## Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your‑username>/finance-news-sentiment-predictor.git
cd finance-news-sentiment-predictor
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the data

Run the data ingestion script to fetch price data and create placeholder sentiment data:

```bash
python src/data_ingestion.py
```

This will download 5 years of SPY price history from Yahoo Finance into
`data/raw/price_data.csv` and generate a placeholder `fnspid_news_sentiment.csv`.  To use the
real FNSPID dataset you must obtain it manually (see **Data Sources** below) and place it
in `data/raw/fnspid_news_sentiment.csv`.

### 3. Build the dataset

Execute the ETL and feature engineering scripts to produce the merged dataset and engineered
features:

```bash
python src/etl.py        # cleans and merges price & sentiment into data/processed/merged_data.csv
python src/features.py   # generates features into data/processed/features.csv
```

### 4. Train the model

Train and evaluate the classifier on the engineered features:

```bash
python src/model.py  # saves trained model to models/model.pkl
```

The script will split the data into training and test sets, train a Random Forest and
report accuracy, precision and recall.  The trained model is serialised to
`models/model.pkl`.

### 5. Run the API

Start the FastAPI server to serve predictions.  You can either run it directly or via Docker:

**Directly with Uvicorn**

```bash
uvicorn src.api:app --reload --port 8000
```

**Using Docker**

```bash
# Build the Docker image
docker build -t finance-news-sentiment-api .

# Run the API container
docker run -p 8000:8000 finance-news-sentiment-api
```

Send a POST request to `/predict` with a JSON body containing the engineered feature values
to obtain a binary prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": {"ma_7": 100.5, "ma_21": 99.8, "rsi_14": 55.2, "sentiment_ma_7": 0.1, "sentiment_lag1": 0.2, "returns_lag1": 0.01}}'
```

The response will include a `prediction` field with `1` for a positive next‑day return and
`0` otherwise.

### 6. Launch the dashboard

Run the Streamlit dashboard to explore prices, sentiment and model outputs interactively:

```bash
streamlit run src/visualize.py
```

The dashboard lets you select a date range and shows time‑series plots of the SPY close
price, sentiment scores and predicted direction along with summary statistics.

## Data Sources

### SPY Price Data

Daily price data for the SPY ETF is downloaded via the open‑source `yfinance`
library, which scrapes historical data from Yahoo Finance.  This data is provided for
informational purposes only and should not be used for trading.  Refer to Yahoo’s
terms of service for usage restrictions.

### FNSPID News Sentiment Dataset

The **FNSPID (Financial News Sentiment Prediction and Item Description)** dataset is a large
collection of time‑aligned financial news articles with sentiment annotations.  Due to
licensing restrictions the dataset is **not included** in this repository.  You must
obtain it manually from the authors’ release and agree to their license before using it.

* Dataset card on Hugging Face: [Zihan1004/FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID)
* Paper: *FNSPID: A Comprehensive Financial News Dataset in Time Series* (Dong et al., 2024)
* **License:** The dataset is distributed under the Creative Commons
  Attribution‑NonCommercial 4.0 International (CC BY‑NC‑4.0) license – commercial use is
  prohibited【397651730625372†L97-L102】.

After downloading, place the raw CSV file at `data/raw/fnspid_news_sentiment.csv` to
ingest it with `src/data_ingestion.py`.

## Notebooks

Two Jupyter notebooks in the `notebooks/` directory walk through the exploratory
data analysis and modelling steps:

* `EDA.ipynb` – examine price, returns and sentiment distributions, compute summary
  statistics and plot time‑series.
* `modeling.ipynb` – train a simple classifier on the engineered features and evaluate
  its performance.

## Screenshots

Below are example outputs from the Streamlit dashboard and API invocation.

### Dashboard

![Dashboard Demo](assets/dashboard_example.png)

### API Example

![API Example](assets/api_example.png)

## Contribution Guidelines

Contributions are welcome!  If you have ideas for new features, bug fixes or
improvements, please open an issue or submit a pull request.  When contributing,
please:

* Create a feature branch off of `data-ingestion` or the most appropriate branch.
* Follow the existing code style and add docstrings where appropriate.
* Write unit tests for any new functionality.
* Ensure that the test suite passes (`pytest`) before submitting.

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for
details.  Note that the FNSPID dataset has its own license (CC BY‑NC‑4.0) which
restricts commercial use【397651730625372†L97-L102】.
