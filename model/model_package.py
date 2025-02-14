from datetime import datetime
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler


class FraudDetectionModel:
    """
    Fraud Detection model wrapper class that handles preprocessing,
    training, and prediction.
    """

    def __init__(self, model_params: Dict[str, Any] = None):
        self.model = None
        self.model_params = model_params or {
            'learning_rate': 0.01,
            'max_depth': 5,
            'n_estimators': 100
        }
        self.feature_columns = None

    def preprocess_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess features for model training or inference.
        """
        # Remove unnecessary columns
        X = df.drop(['infraction', 'event_created_at', 'merchant_id'], axis=1)
        y = df['infraction'] if 'infraction' in df.columns else None

        # Store feature columns for inference
        if self.feature_columns is None:
            self.feature_columns = X.columns.tolist()

        return X, y

    def train(self, df: pd.DataFrame) -> None:
        """
        Train the model with undersampling.
        """
        # Preprocess data
        X, y = self.preprocess_features(df)

        # Apply undersampling
        rus = RandomUnderSampler(sampling_strategy=0.25)
        X_resampled, y_resampled = rus.fit_resample(X, y)

        # Initialize and train model
        self.model = xgb.XGBRegressor(**self.model_params)
        self.model.fit(X_resampled, y_resampled)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Preprocess features
        X, _ = self.preprocess_features(df)

        # Ensure all expected features are present
        missing_features = set(self.feature_columns) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Align features with training order
        X = X[self.feature_columns]

        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath: str) -> None:
        """
        Save model to disk.
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_params': self.model_params,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(model_data, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'FraudDetectionModel':
        """
        Load model from disk.
        """
        model_data = joblib.load(filepath)

        instance = cls()
        instance.model = model_data['model']
        instance.feature_columns = model_data['feature_columns']
        instance.model_params = model_data['model_params']

        return instance
