"""Shared fixtures.

There is no raw ``sumup_case.parquet`` in this repository (it's listed in
``.gitignore`` and was never committed), so tests train against a small
synthetic dataset with the same schema instead of mocking the model away.
This is deliberate: it means ``test_train_serve_parity`` in
``test_api.py`` exercises the *real* pipeline code, catching exactly the
kind of training/serving mismatch that made the original API unusable.
"""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features import add_temporal_features
from fraud_detection.pipeline import build_pipeline
from fraud_detection.registry import ModelArtifact, save_artifact
from fraud_detection.schema import (
    MERCHANT_ID_COL,
    MERCHANT_PROFILE_COLS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COL,
    TIMESTAMP_COL,
)


@pytest.fixture
def synthetic_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 1500

    df = pd.DataFrame(
        {
            TIMESTAMP_COL: pd.date_range("2026-01-01", periods=n, freq="5min"),
            MERCHANT_ID_COL: rng.integers(1, 30, n).astype(str),
            "amount": rng.exponential(200, n),
        }
    )
    for col in MERCHANT_PROFILE_COLS:
        if col in ("is_legal_entity", "is_mei", "is_person"):
            df[col] = rng.integers(0, 2, n)
        else:
            df[col] = rng.exponential(50, n)
            null_mask = rng.random(n) < 0.05
            df.loc[null_mask, col] = np.nan

    df[TARGET_COL] = (rng.random(n) < 0.03).astype(int)
    return add_temporal_features(df)


@pytest.fixture
def feature_columns(synthetic_dataset) -> list:
    return [c for c in MODEL_FEATURE_COLUMNS if c in synthetic_dataset.columns]


@pytest.fixture
def trained_artifact(synthetic_dataset, feature_columns) -> ModelArtifact:
    X = synthetic_dataset[feature_columns]
    y = synthetic_dataset[TARGET_COL]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    return ModelArtifact(
        pipeline=pipeline,
        feature_names=feature_columns,
        threshold=0.5,
        metadata={"trained_at": "test-fixture"},
    )


@pytest.fixture
def artifact_path(tmp_path, trained_artifact) -> str:
    return save_artifact(trained_artifact, str(tmp_path))


@pytest.fixture
def merchant_features_csv(tmp_path, synthetic_dataset) -> str:
    """A minimal feature-store CSV: one row per merchant with its latest
    profile, mirroring what an online feature store would return."""
    profile_cols = [c for c in MERCHANT_PROFILE_COLS if c in synthetic_dataset.columns]
    table = (
        synthetic_dataset.sort_values(TIMESTAMP_COL)
        .groupby(MERCHANT_ID_COL)[profile_cols]
        .last()
        .reset_index()
    )
    path = tmp_path / "merchant_features.csv"
    table.to_csv(path, index=False)
    return str(path)
