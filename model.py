"""
Model training and evaluation module.

This module defines a supervised learning pipeline to predict the next‑day
return direction (up/down) from engineered features. It provides
utilities to load the feature set, construct the target variable,
split the data, train a classifier, evaluate its performance and
persist the trained model. A simple RandomForestClassifier from
scikit‑learn is used by default since ``xgboost`` is not listed in
requirements.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional, Tuple, Dict, Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score  # type: ignore

# Configure a module-level logger
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def load_features(path: Optional[str] = None) -> pd.DataFrame:
    """Load the engineered feature set from CSV.

    Args:
        path: Optional custom path to the features file. Defaults to
            ``data/processed/features.csv``.

    Returns:
        DataFrame of features.

    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If the CSV cannot be parsed.
    """
    if path is None:
        path = os.path.join(PROCESSED_DIR, 'features.csv')
    logger.info("Loading features from %s", path)
    if not os.path.exists(path):
        logger.error("Features file not found: %s", path)
        raise FileNotFoundError(f"Features file not found: {path}")
    try:
        df = pd.read_csv(path, parse_dates=['date'])
    except Exception as exc:
        logger.exception("Failed to load features from %s", path)
        raise ValueError(f"Failed to read features: {exc}")
    return df


def construct_target(df: pd.DataFrame) -> pd.Series:
    """Create a binary target indicating whether the next day's return is positive.

    The function uses the ``returns`` column to look ahead one period.

    Args:
        df: DataFrame containing a ``returns`` column.

    Returns:
        A pandas Series of 0/1 values (1 if next day's return > 0).
    """
    if 'returns' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'returns' column to construct target.")
    # Shift returns backward to align next day's return with current row
    next_returns = df['returns'].shift(-1)
    target = (next_returns > 0).astype(int)
    # Drop last value (will be NaN due to shift), fill with 0
    target = target.fillna(0).astype(int)
    return target


def prepare_training_data(df: pd.DataFrame, feature_cols: Optional[list] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target vector from feature DataFrame.

    Args:
        df: DataFrame with engineered features including ``returns`` and ``sentiment``.
        feature_cols: Optional list of column names to use as features. If None,
            all numeric columns except ``date``, ``returns`` and ``sentiment`` are used.

    Returns:
        Tuple (X, y) where X is the feature matrix and y is the target vector.
    """
    target = construct_target(df)
    # Default feature columns exclude date and target/label columns
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in ['date', 'returns', 'sentiment']]
    X = df[feature_cols]
    return X, target


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train a RandomForest classifier on the provided data.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.

    Returns:
        Trained RandomForestClassifier.
    """
    logger.info("Training RandomForest classifier on %d samples", len(X_train))
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate a trained model on the test set.

    Args:
        model: Trained classifier.
        X_test: Feature matrix for testing.
        y_test: True labels for testing.

    Returns:
        Dictionary containing accuracy, precision and recall.
    """
    logger.info("Evaluating model on %d samples", len(X_test))
    predictions = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions, zero_division=0),
    }
    return metrics


def save_model(model: Any, path: Optional[str] = None) -> None:
    """Persist the trained model to disk using pickle.

    Args:
        model: Trained model object.
        path: Optional path to save the model. Defaults to
            ``models/model.pkl``.

    Raises:
        IOError: If the model cannot be saved.
    """
    if path is None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, 'model.pkl')
    logger.info("Saving model to %s", path)
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    except Exception as exc:
        logger.exception("Failed to save model to %s", path)
        raise IOError(f"Failed to save model: {exc}")


def load_model(path: Optional[str] = None) -> Any:
    """Load a previously saved model from disk.

    Args:
        path: Optional path to the saved model. Defaults to
            ``models/model.pkl``.

    Returns:
        Deserialized model object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if path is None:
        path = os.path.join(MODELS_DIR, 'model.pkl')
    logger.info("Loading model from %s", path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_single(model: Any, sample: pd.DataFrame) -> int:
    """Predict the next-day return direction for a single feature row.

    Args:
        model: Trained classifier.
        sample: DataFrame or Series representing a single observation with
            the same feature columns used in training.

    Returns:
        Predicted class label (0 for down/non-positive return, 1 for up).
    """
    if isinstance(sample, pd.Series):
        sample = sample.to_frame().T
    pred = model.predict(sample)[0]
    return int(pred)


def run_training(feature_path: Optional[str] = None) -> Tuple[Any, Dict[str, float]]:
    """High-level function to orchestrate model training and evaluation.

    This function loads the features, constructs the target, splits the
    data, trains the model, evaluates it and saves the trained model.

    Args:
        feature_path: Optional custom path to the features CSV.

    Returns:
        Tuple of (trained_model, evaluation_metrics).
    """
    try:
        df = load_features(feature_path)
        X, y = prepare_training_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        save_model(model)
        logger.info(
            "Training complete. Accuracy: %.3f, Precision: %.3f, Recall: %.3f",
            metrics['accuracy'], metrics['precision'], metrics['recall']
        )
        return model, metrics
    except Exception as exc:
        logger.error("Model training pipeline failed: %s", exc)
        raise


if __name__ == '__main__':  # pragma: no cover
    run_training()
