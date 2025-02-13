from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import uvicorn
import sys
import os

# Add parent directory to system path to import model package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_package import FraudDetectionModel

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

# Load model at startup
try:
    MODEL_PATH = os.getenv('MODEL_PATH', '../models/latest_model.pkl')
    model = FraudDetectionModel.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise


class TransactionRequest(BaseModel):
    merchant_id: str = Field(..., description="Unique identifier for the merchant")
    amount: float = Field(..., description="Transaction amount")

    # Add other required fields

    class Config:
        schema_extra = {
            "example": {
                "merchant_id": "MERCH123",
                "amount": 150.00
            }
        }


class InferenceResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    requires_review: bool
    review_priority: str
    inference_time_ms: float


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
        # Generate transaction ID
        transaction_id = f"TXN_{start_time.strftime('%Y%m%d_%H%M%S')}_{transaction.merchant_id}"

        # Create DataFrame for prediction
        df = pd.DataFrame([transaction.dict()])

        # Make prediction
        probability = model.predict(df)[0]

        # Determine if review is required
        OPTIMAL_THRESHOLD = 0.7  # Update based on your analysis
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

        # Log inference
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