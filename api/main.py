from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Dict, Optional
import joblib
import redis
from datetime import datetime
import json

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection API for PIX transactions"
)

# Redis for caching merchant profiles
redis_client = redis.Redis(host='localhost', port=6379, db=0)


class Transaction(BaseModel):
    merchant_id: str
    amount: float
    event_created_at: str
    # Add other required fields based on your model features


class PredictionResponse(BaseModel):
    transaction_id: str
    risk_score: float
    review_priority: str
    review_needed: bool
    supervised_score: float
    unsupervised_score: float
    prediction_timestamp: str


# Load model on startup
@app.on_event("startup")
async def load_model():
    global model
    model = joblib.load('models/fraud_detection_model.joblib')


@app.post("/predict", response_model=PredictionResponse)
async def predict_transaction(transaction: Transaction):
    try:
        # Convert transaction to DataFrame
        df = pd.DataFrame([transaction.dict()])

        # Preprocess features
        df = model.preprocess_features(df)

        # Get predictions
        predictions = model.predict(df)

        # Get merchant profile from cache
        merchant_profile = redis_client.get(f"merchant:{transaction.merchant_id}")
        if merchant_profile:
            merchant_profile = json.loads(merchant_profile)
        else:
            merchant_profile = {"transaction_count": 0, "avg_amount": 0}

        # Calculate review priority
        risk_score = predictions['combined_score'][0]

        if risk_score > 0.8:
            priority = "HIGH"
            needs_review = True
        elif risk_score > 0.6:
            priority = "MEDIUM"
            needs_review = True
        else:
            priority = "LOW"
            needs_review = False

        # Update merchant profile
        merchant_profile["transaction_count"] += 1
        merchant_profile["avg_amount"] = (
                (merchant_profile["avg_amount"] * (merchant_profile["transaction_count"] - 1) +
                 transaction.amount) / merchant_profile["transaction_count"]
        )

        # Cache updated profile
        redis_client.setex(
            f"merchant:{transaction.merchant_id}",
            3600,  # 1 hour expiry
            json.dumps(merchant_profile)
        )

        return PredictionResponse(
            transaction_id=str(transaction.merchant_id),
            risk_score=float(risk_score),
            review_priority=priority,
            review_needed=needs_review,
            supervised_score=float(predictions['supervised_score'][0]),
            unsupervised_score=float(predictions['unsupervised_score'][0]),
            prediction_timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}