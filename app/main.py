"""Real-time PIX fraud scoring API.

Rewritten end-to-end; the previous version could not serve a single
successful request (see plan.md 2.2 for the full list: it treated the
Pydantic model as a dict, dropped columns that don't exist on the request
schema, expected ~94 model features from a 10-field request body, and
returned a response that didn't match its own ``response_model``). The fixes
here are structural, not patches:

* Preprocessing (temporal features, null handling) lives in
  ``fraud_detection`` and is shared with training — no second copy.
* The ~90 merchant aggregate features a caller cannot supply are resolved
  from ``FeatureStore`` (``app/feature_store.py``), keyed by merchant_id.
* The model artifact is a self-describing ``ModelArtifact`` (feature order +
  threshold + pipeline together), loaded once at startup via ``lifespan``.
* The response model and the returned object are the same shape, checked by
  FastAPI's own validation instead of by hand.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.feature_store import CsvFeatureStore, FeatureStore
from fraud_detection.config import settings
from fraud_detection.features import add_temporal_features
from fraud_detection.registry import ModelArtifact, load_artifact
from fraud_detection.schema import TIMESTAMP_COL

logger = logging.getLogger("fraud_detection.api")

state: dict[str, object] = {"artifact": None, "feature_store": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["artifact"] = load_artifact(settings.model_path)
    except FileNotFoundError:
        logger.error("Model artifact not found at %s; /predict will return 503.", settings.model_path)
        state["artifact"] = None

    state["feature_store"] = CsvFeatureStore(settings.feature_store_path)
    yield
    state.clear()


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring for PIX transactions received by SumUp merchant accounts",
    version="2.0.0",
    lifespan=lifespan,
)


class Transaction(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "merchant_id": "MERCH123",
                "amount": 1000.00,
                "event_created_at": "2026-07-22T14:30:00Z",
            }
        }
    )

    merchant_id: str = Field(..., description="Unique identifier for the merchant")
    amount: float = Field(..., gt=0, description="Transaction amount in local currency")
    event_created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the PIX was received; defaults to now for real-time scoring",
    )


class FraudPredictionResponse(BaseModel):
    transaction_id: str
    merchant_id: str
    fraud_probability: float
    review_required: bool
    threshold: float
    model_version: str
    timestamp: str


def _get_artifact() -> ModelArtifact:
    artifact = state.get("artifact")
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return artifact


def _get_feature_store() -> FeatureStore:
    return state["feature_store"]


def build_feature_row(transaction: Transaction, feature_store: FeatureStore) -> pd.DataFrame:
    """Assemble the single-row feature frame the model expects.

    Merges what the caller sent (amount, timestamp) with what only the
    feature store knows (merchant profile + rolling aggregates), then derives
    the same temporal features used at training time from the transaction's
    own timestamp — never the server's wall clock.
    """
    row = {
        "amount": transaction.amount,
        TIMESTAMP_COL: transaction.event_created_at,
        **feature_store.get_merchant_features(transaction.merchant_id),
    }
    df = pd.DataFrame([row])
    return add_temporal_features(df, timestamp_col=TIMESTAMP_COL)


@app.post("/api/v1/predict", response_model=FraudPredictionResponse)
async def predict_fraud(transaction: Transaction) -> FraudPredictionResponse:
    """Score a single PIX transaction for fraud risk."""
    artifact = _get_artifact()
    feature_store = _get_feature_store()

    try:
        features = build_feature_row(transaction, feature_store)
        fraud_probability = float(artifact.predict_proba(features)[0])
    except Exception:
        logger.exception("Prediction failed for merchant_id=%s", transaction.merchant_id)
        raise HTTPException(status_code=500, detail="Prediction failed") from None

    return FraudPredictionResponse(
        transaction_id=str(uuid.uuid4()),
        merchant_id=transaction.merchant_id,
        fraud_probability=fraud_probability,
        review_required=fraud_probability > artifact.threshold,
        threshold=artifact.threshold,
        model_version=artifact.metadata.get("trained_at", "unknown"),
        timestamp=datetime.now(UTC).isoformat(),
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PIX Fraud Detection API",
        "version": app.version,
        "endpoints": {
            "/": "This information",
            "/docs": "OpenAPI documentation",
            "/redoc": "ReDoc documentation",
            "/health": "Health check",
            "/api/v1/predict": "Fraud prediction endpoint",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if state.get("artifact") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "timestamp": datetime.now(UTC).isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
