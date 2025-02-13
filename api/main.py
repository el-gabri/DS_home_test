import os
from datetime import datetime
from typing import Dict, Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load the trained model
MODEL_PATH = "../models/supervised/xgboost_model_20250212_144931.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at path: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Initialize FastAPI api
app = FastAPI(
    title="Fraud Detection API",
    description="API for real-time fraud detection in financial transactions",
    version="1.0.0"
)


class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in local currency")
    merchant_id: str = Field(..., description="Unique identifier for the merchant")
    is_person: bool = Field(..., description="Whether the merchant is an individual")
    is_legal_entity: bool = Field(..., description="Whether the merchant is a legal entity")
    is_mei: bool = Field(..., description="Whether the merchant is a MEI")
    mc_billpay_br_sum_last_15d: float = Field(0.0, description="Sum of bill payments in last 15 days")
    mc_billpay_br_count_last_1y: int = Field(0, description="Count of bill payments in last year")
    mc_pur_br_sum_last_15d: float = Field(0.0, description="Sum of purchases in last 15 days")
    mc_billpay_br_sum_last_1y: float = Field(0.0, description="Sum of bill payments in last year")
    mc_pur_br_sum_last_1y: float = Field(0.0, description="Sum of purchases in last year")

    # Add other required fields based on model's features

    class Config:
        schema_extra = {
            # "example": {
            #     "amount": 1000.00,
            #     "merchant_id": "MERCH123",
            #     "is_person": True,
            #     "is_legal_entity": False,
            #     "is_mei": False,
            #     "mc_billpay_br_sum_last_15d": 500.0,
            #     "mc_billpay_br_count_last_1y": 12,
            #     "mc_pur_br_sum_last_15d": 300.0,
            #     "mc_billpay_br_sum_last_1y": 5000.0,
            #     "mc_pur_br_sum_last_1y": 3000.0
            # }
        }


class FraudPredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraudulent: bool
    risk_score: int
    timestamp: str
    merchant_id: str


def preprocess_transaction(transaction: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess a single transaction for model inference."""
    # Create a single-row DataFrame
    df = pd.DataFrame([transaction])

    # Add derived temporal features
    now = datetime.now()
    df['is_weekend'] = 1 if now.weekday() >= 5 else 0
    df['is_night'] = 1 if (now.hour < 6 or now.hour >= 18) else 0
    df['month'] = now.month
    df['day'] = now.day
    df['hour'] = now.hour

    df['amount'] = df['amount'] * 0.1

    # Ensure all required columns are present in the correct order
    # Add any missing columns with default values
    # This should match your training data structure

    return df


@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(transaction: Transaction):
    try:
        # Preprocess the transaction
        df = preprocess_transaction(transaction.model_dump())

        # Make prediction
        fraud_probability = model.predict_proba(df)[0, 1]

        # Use 0.7 threshold (based on your model analysis)
        is_fraudulent = fraud_probability > 0.7

        # Calculate risk score (0-100)
        risk_score = int(fraud_probability * 100)

        return FraudPredictionResponse(
            transaction_id=f"TX_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            fraud_probability=float(fraud_probability),
            is_fraudulent=bool(is_fraudulent),
            risk_score=risk_score,
            timestamp=datetime.now().isoformat(),
            merchant_id=transaction.merchant_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_version": "1.0.0"}
