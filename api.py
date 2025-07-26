"""
REST API for serving the trained finance news sentiment prediction model.

This module defines a FastAPI application that exposes endpoints for
checking service status and generating predictions from the trained
model. It loads the model from `models/model.pkl` on startup.
"""

import logging
import os
from typing import Dict, Any

import pandas as pd  # type: ignore
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import model utilities from the local module
from model import load_model, predict_single

logger = logging.getLogger(__name__)

# Path to the trained model file relative to this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')


def _load_trained_model(path: str) -> Any:
    """Load the trained model from disk, returning None on failure."""
    try:
        return load_model(path)
    except Exception as exc:
        logger.exception("Failed to load trained model from %s", path)
        return None


# Instantiate the FastAPI application
app = FastAPI(title="Finance News Sentiment Predictor API")

# Load the model when the module is imported
model = _load_trained_model(MODEL_PATH)


class PredictionRequest(BaseModel):
    """Schema for prediction requests.

    The `features` field should be a mapping of feature names to their numeric values.
    """
    features: Dict[str, float]


@app.get("/")
def read_root() -> Dict[str, str]:
    """Health check endpoint."""
    return {"message": "Finance News Sentiment Predictor API is running"}


@app.post("/predict")
def make_prediction(request: PredictionRequest) -> Dict[str, int]:
    """Return a prediction for the provided features."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not available.")
    try:
        sample = pd.DataFrame([request.features])
        prediction = predict_single(model, sample)
    except Exception as exc:
        logger.exception("Error during prediction: %s", exc)
        raise HTTPException(status_code=400, detail="Failed to generate prediction.")
    return {"prediction": int(prediction)}
