from sklearn.model_selection import GridSearchCV, KFold
from joblib import parallel_backend
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any

from .model_package import FraudDetectionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(
        df: pd.DataFrame,
        param_grid: Dict[str, Any] = None,
        cv_folds: int = 5
) -> FraudDetectionModel:
    """
    Train the fraud detection model with grid search CV.
    """
    logger.info("Starting model training process")

    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'learning_rate': [0.01, 0.25],
            'max_depth': [5, 50],
            'n_estimators': [20, 500]
        }

    try:
        # Initialize base model
        base_model = FraudDetectionModel()

        # Prepare data
        X, y = base_model.preprocess_features(df)

        # Setup grid search
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=base_model.model,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

        # Perform grid search with multiprocessing
        logger.info("Starting grid search")
        with parallel_backend('multiprocessing'):
            grid_search.fit(X, y)

        # Get best parameters
        best_params = grid_search.best_params_
        logger.info(f"Best parameters found: {best_params}")

        # Train final model with best parameters
        final_model = FraudDetectionModel(model_params=best_params)
        final_model.train(df)

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"../models/fraud_model_{timestamp}.pkl"
        final_model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        return final_model

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    try:
        # Load data
        df = pd.read_csv("path_to_your_data.csv")

        # Train model
        model = train_model(df)

        logger.info("Model training completed successfully")

    except Exception as e:
        logger.error(f"Training script failed: {str(e)}")
        raise