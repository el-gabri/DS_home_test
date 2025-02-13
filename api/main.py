from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection inference API",
    version="1.0.0"
)

# Load the model at startup
try:
    model = joblib.load('../model/supervised/xgboost_model_20250212_144931.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Expected feature names and types
EXPECTED_FEATURES = {
    'amount': float,
    'hour': int,
    'day_of_week': int,
    'month': int,
    # Add all other required features here
}


class TransactionRequest(BaseModel):
    merchant_id: str = Field(..., description="Unique identifier for the merchant")
    amount: float = Field(..., description="Transaction amount")

    # Add other required fields based on your model's features

    class Config:
        schema_extra = {
            "example": {
                "merchant_id": "MERCH123",
                "amount": 150.00,
                # Add example values for other fields
            }
        }


class InferenceResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    requires_review: bool
    review_priority: str
    inference_time_ms: float


def preprocess_transaction(transaction: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess a single transaction for model inference.
    """
    try:
        # Extract timestamp features
        current_time = datetime.now()

        processed_data = {
            'amount': transaction['amount'],
            'hour': current_time.hour,
            'day_of_week': current_time.weekday(),
            'month': current_time.month,
            # Add processing for other features
        }

        # Create DataFrame with expected features in correct order
        df = pd.DataFrame([processed_data])

        # Ensure all required features are present
        missing_features = set(EXPECTED_FEATURES.keys()) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        return df

    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")


def get_review_priority(probability: float) -> str:
    """
    Determine review priority based on fraud probability.
    """
    if probability >= 0.9:
        return "HIGH"
    elif probability >= 0.7:
        return "MEDIUM"
    else:
        return "LOW"


@app.post("/predict", response_model=InferenceResponse)
async def predict(transaction: TransactionRequest):
    """
    Make real-time fraud prediction for a single transaction.
    """
    start_time = datetime.now()

    try:
        # Generate unique transaction ID
        transaction_id = f"TXN_{start_time.strftime('%Y%m%d_%H%M%S')}_{transaction.merchant_id}"

        # Preprocess transaction
        df = preprocess_transaction(transaction.dict())

        # Make prediction
        probability = model.predict_proba(df)[0, 1]

        # Determine if review is required (using optimal threshold)
        OPTIMAL_THRESHOLD = 0.7  # Replace with your optimal threshold
        requires_review = probability > OPTIMAL_THRESHOLD

        # Calculate inference time
        inference_time = (datetime.now() - start_time).total_seconds() * 1000

        # Prepare response
        response = InferenceResponse(
            transaction_id=transaction_id,
            fraud_probability=float(probability),
            requires_review=requires_review,
            review_priority=get_review_priority(probability),
            inference_time_ms=inference_time
        )

        # Log inference details
        logger.info(
            f"Inference completed - ID: {transaction_id}, "
            f"Probability: {probability:.4f}, "
            f"Review Required: {requires_review}, "
            f"Time: {inference_time:.2f}ms"
        )

        return response

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)