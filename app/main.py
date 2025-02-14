import os
import uuid
from datetime import datetime
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load the trained model
MODEL_PATH = "../model/supervised/xgboost_model_20250212_144931.pkl"
#MODEL_PATH = "../model/supervised/ADA_model_unshuffled20250214_153820.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at path: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Initialize FastAPI app
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

    # Drop unnecessary columns
    df = df.drop(['event_created_at', 'merchant_id'], axis=1)

    # %%
    list_variables_to_drop = ['mc_billpay_br_count_sanction_last_30d',
                              'mc_pep_count_last_2d',
                              'mc_billpay_br_count_pep_br_last_30d',
                              'mc_credit_transfer_in_br_count_pep_br_last_30d',
                              'mc_credit_transfer_out_br_count_pep_br_last_30d',
                              'mc_tx_amount_pend_sum_last_14d',
                              'mc_chip_tx_amount_local_succ_sum_7d']

    df = df.drop(columns=list_variables_to_drop)

    # Ensure all required columns are present in the correct order
    # Add any missing columns with default values
    # This should match your training data structure

    return df


def treat_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive missing value treatment preserving the signal from null values
    """
    # Create a copy to avoid modifying the original dataframe
    df_treated = df.copy()

    # Store columns to process (excluding target and non-numeric columns)
    columns_to_process = df.select_dtypes(include=['float64', 'int64']).columns
    columns_to_process = [col for col in columns_to_process if
                          not col in ['infraction', 'merchant_id', 'event_created_at']]

    # Forward fill for time-dependent features
    df_treated['event_created_at'] = pd.to_datetime(df_treated['event_created_at'])
    df_treated = df_treated.sort_values('event_created_at')

    for column in columns_to_process:
        if df[column].isnull().any():
            # Calculate the fraud rate difference for null vs non-null
            fraud_rate_null = df[df[column].isnull()]['infraction'].mean()
            fraud_rate_non_null = df[df[column].notnull()]['infraction'].mean()
            ratio = fraud_rate_null / fraud_rate_non_null if fraud_rate_non_null > 0 else np.inf

            # Choose imputation strategy based on fraud rate ratio
            if ratio > 10:  # If nulls are much more likely to be fraud
                # Use a special value (e.g., -999) to preserve the signal
                df_treated[column] = df_treated[column].fillna(-999999)
            else:
                df_treated[column] = df_treated[column].fillna(df_treated[column].median())

    return df_treated


@app.post("/api/v1/predict", response_model=FraudPredictionResponse)
async def predict_fraud(transaction: Transaction):
    """
    Predict fraud probability for a PIX transaction.

    Args:
        transaction: Transaction details including amount and merchant information

    Returns:
        Prediction results including fraud probability and review flag
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Preprocess transaction
        features = preprocess_transaction(transaction)

        # Make prediction
        fraud_probability = float(model.predict_proba(features)[0][1])

        # Apply threshold (adjust as needed)
        is_fraud = fraud_probability >  0.7778  # Using threshold from analysis
        review_required = fraud_probability >  0.7778

        return FraudPredictionResponse(
            transaction_id=str(uuid.uuid4()),
            fraud_probability=fraud_probability,
            is_fraud=is_fraud,
            review_required=review_required,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PIX Fraud Detection API",
        "version": "1.0.0",
        "description": "API for real-time fraud detection in PIX transactions",
        "endpoints": {
            "/": "This information",
            "/docs": "OpenAPI documentation",
            "/redoc": "ReDoc documentation",
            "/health": "Health check",
            "/api/v1/predict": "Fraud prediction endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }
