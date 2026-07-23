"""Thin, backward-compatible training wrapper.

This used to contain its own copy of the preprocessing/sampling/model logic,
which is how it ended up instantiating an ``xgb.XGBRegressor`` and then
calling the classifier-only ``predict_proba`` on it (a guaranteed crash — see
plan.md 2.1). It now delegates everything to ``fraud_detection.pipeline`` and
``fraud_detection.registry`` so there is exactly one implementation of the
model, shared with ``fraud_detection.train`` and the serving API.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from fraud_detection.pipeline import build_pipeline
from fraud_detection.registry import ModelArtifact, load_artifact, save_artifact
from fraud_detection.schema import NON_FEATURE_COLS, TARGET_COL


class FraudDetectionModel:
    """Fraud detection model wrapper: preprocessing, training, prediction."""

    def __init__(self, model_params: Optional[Dict[str, Any]] = None, random_state: int = 42):
        self.model_params = model_params
        self.random_state = random_state
        self.pipeline = build_pipeline(model_params=model_params, random_state=random_state)
        self.feature_columns = None
        self._fitted = False

    def preprocess_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Split a raw dataframe into model features X and label y (if present)."""
        X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
        y = df[TARGET_COL] if TARGET_COL in df.columns else None

        if self.feature_columns is None:
            self.feature_columns = X.columns.tolist()

        return X, y

    def train(self, df: pd.DataFrame) -> None:
        """Fit the pipeline. Undersampling happens inside the pipeline (train-time
        only), not as a manual step against the whole dataframe, so it can never
        leak into evaluation or serving."""
        X, y = self.preprocess_features(df)
        if y is None:
            raise ValueError(f"Training data must include the '{TARGET_COL}' column.")

        self.pipeline.fit(X, y)
        self._fitted = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Fraud probability for each row of ``df``."""
        if not self._fitted:
            raise ValueError("Model not trained. Call train() first.")

        X, _ = self.preprocess_features(df)

        missing_features = set(self.feature_columns) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        X = X[self.feature_columns]
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, output_dir: str, threshold: float = 0.5) -> str:
        if not self._fitted:
            raise ValueError("No model to save. Train the model first.")

        artifact = ModelArtifact(
            pipeline=self.pipeline,
            feature_names=self.feature_columns,
            threshold=threshold,
        )
        return save_artifact(artifact, output_dir)

    @classmethod
    def load(cls, filepath: str) -> "FraudDetectionModel":
        artifact = load_artifact(filepath)

        instance = cls()
        instance.pipeline = artifact.pipeline
        instance.feature_columns = artifact.feature_names
        instance._fitted = True
        return instance
