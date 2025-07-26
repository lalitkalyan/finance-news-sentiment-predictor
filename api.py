"""
API module exposing the model via a web server.

This script uses FastAPI to define endpoints for predictions.
"""

from fastapi import FastAPI

app = FastAPI(title="Finance News Sentiment Predictor API")


@app.get("/")
def read_root():
    """
    Root endpoint providing a welcome message.

    Returns:
        A dictionary with a welcome message.
    """
    return {"message": "Welcome to the Finance News Sentiment Predictor API"}


# TODO: Define additional endpoints for making predictions once the model is implemented.
