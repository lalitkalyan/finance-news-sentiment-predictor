# Finance News Sentiment Predictor

This project aims to build a system that ingests financial market data and news sentiment to predict movements in financial instruments. It leverages public data sources such as Yahoo Finance and the FNSPID (Financial News Sentiment Prediction and Item Description) dataset. The repository is organised to facilitate data ingestion, preprocessing, feature engineering, model training and evaluation, and exposing predictions via a web API.

## Data Sources

- **Stock Prices:** Historical price data is downloaded using the `yfinance` library. By default, the data ingestion script downloads five years of daily price data for the SPY ETF and saves it to `data/raw/price_data.csv`.
- **FNSPID Dataset:** The FNSPID dataset contains news articles labeled with sentiment information. Access to this dataset may require registration or authentication due to licensing restrictions. Please follow the instructions provided by the dataset provider to download the files manually. Once downloaded, place the dataset in the `data/raw/` directory so it can be processed by the ETL pipeline.


## Running the API

You can serve the trained model using FastAPI. The recommended way is to run it inside a Docker container.

### Build the Docker image

From the project root directory, build the image with:

```sh
docker build -t finance-news-sentiment-api .
```

### Run the API

Run the container and expose port 8000 on your host machine:

```sh
docker run -p 8000:8000 finance-news-sentiment-api
```

This starts the FastAPI server at `http://localhost:8000`. You can check the service status with:

```sh
curl http://localhost:8000/
```

To get a prediction, send a POST request to the `/predict` endpoint with feature values as JSON. For example:

```sh
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": {"moving_average_7": 430.5, "moving_average_21": 425.7, "rsi": 55.3, "rolling_sentiment_7": 0.12, "sentiment_lag_1": 0.15, "sentiment_lag_2": 0.10, "returns_lag_1": 0.001, "returns_lag_2": -0.002}}'
```

The API returns a JSON object containing the predicted direction (0 for down or 1 for up).
