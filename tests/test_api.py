import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud_detection.config import settings


@pytest.fixture
def client(artifact_path, merchant_features_csv):
    # Point the already-instantiated settings singleton at the fixtures
    # before the app's lifespan runs, instead of relying on env vars that
    # would only be read at process start.
    settings.model_path = artifact_path
    settings.feature_store_path = merchant_features_csv

    from app.main import app  # imported here so settings above take effect

    with TestClient(app) as test_client:
        yield test_client


def test_health_check_reports_healthy_once_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_lists_endpoints(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "/api/v1/predict" in response.json()["endpoints"]


def test_predict_known_merchant_returns_full_schema(client, synthetic_dataset):
    merchant_id = synthetic_dataset["merchant_id"].iloc[0]
    payload = {
        "merchant_id": merchant_id,
        "amount": 250.0,
        "event_created_at": "2026-07-22T14:30:00Z",
    }

    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == {
        "transaction_id",
        "merchant_id",
        "fraud_probability",
        "review_required",
        "threshold",
        "model_version",
        "timestamp",
    }
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["review_required"] == (body["fraud_probability"] > body["threshold"])
    assert body["merchant_id"] == merchant_id


def test_predict_unknown_merchant_still_returns_a_score(client):
    """An unseen merchant has no feature-store row; the request must still
    succeed by falling back to the imputer's learned strategy, not error."""
    payload = {"merchant_id": "NEVER_SEEN_BEFORE", "amount": 500.0}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    assert 0.0 <= response.json()["fraud_probability"] <= 1.0


def test_predict_rejects_non_positive_amount(client):
    payload = {"merchant_id": "MERCH123", "amount": 0}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_predict_defaults_timestamp_to_now_when_omitted(client):
    payload = {"merchant_id": "MERCH123", "amount": 100.0}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200


def test_health_check_returns_503_when_model_not_loaded(merchant_features_csv):
    settings.model_path = "does/not/exist.pkl"
    settings.feature_store_path = merchant_features_csv

    from app.main import app

    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 503


def test_train_serve_parity(trained_artifact, artifact_path, synthetic_dataset, feature_columns, tmp_path):
    """The score the API returns for a transaction must equal what the
    trained pipeline predicts directly on the same underlying row — i.e.
    scoring through the HTTP layer introduces no drift from the model that
    was actually trained. This is the regression test for plan.md 3.2.

    Uses a feature-store CSV that holds exactly this one row's merchant
    profile (rather than the shared "latest snapshot per merchant" fixture),
    so the test isolates API-vs-pipeline scoring math from feature-store
    freshness semantics, which is a separate concern.
    """
    from fraud_detection.schema import MERCHANT_ID_COL, MERCHANT_PROFILE_COLS

    row = synthetic_dataset.iloc[0]
    merchant_id = row[MERCHANT_ID_COL]

    single_row_store = tmp_path / "single_merchant.csv"
    row.to_frame().T[[MERCHANT_ID_COL, *[c for c in MERCHANT_PROFILE_COLS if c in row.index]]].to_csv(
        single_row_store, index=False
    )

    settings.model_path = artifact_path
    settings.feature_store_path = str(single_row_store)
    from app.main import app

    with TestClient(app) as test_client:
        payload = {
            "merchant_id": merchant_id,
            "amount": float(row["amount"]),
            "event_created_at": row["event_created_at"].isoformat(),
        }
        response = test_client.post("/api/v1/predict", json=payload)

    assert response.status_code == 200
    api_score = response.json()["fraud_probability"]

    direct_score = trained_artifact.predict_proba(synthetic_dataset[feature_columns].iloc[[0]])[0]
    assert np.isclose(api_score, direct_score, atol=1e-6)
