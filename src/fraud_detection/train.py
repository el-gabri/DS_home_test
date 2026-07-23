"""Training entry point.

Usage::

    python -m fraud_detection.train --data data/raw/sumup_case.parquet --output-dir model/supervised

This script does not exist to be clever — it exists so training happens
through the exact same ``build_pipeline`` / ``add_temporal_features`` /
``NullSignalImputer`` code the serving API loads, with a time-respecting
split and a business-driven threshold, replacing the notebook-only,
copy-pasted version of this logic.
"""

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detection.evaluate import business_cost_curve, precision_at_k, threshold_for_daily_capacity
from fraud_detection.features import add_temporal_features
from fraud_detection.pipeline import build_pipeline
from fraud_detection.registry import ModelArtifact, save_artifact
from fraud_detection.schema import MODEL_FEATURE_COLUMNS, NON_FEATURE_COLS, TARGET_COL, TIMESTAMP_COL


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df = df.drop_duplicates()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return add_temporal_features(df)


def time_series_cross_validate(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> None:
    """Sanity-check average precision across time-respecting folds.

    Uses ``TimeSeriesSplit`` (not ``StratifiedKFold``) precisely because the
    data is temporal and fraud patterns drift: a shuffled CV would let the
    model "see the future" within its own training folds and overstate
    generalization, which is what the original grid search did.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in splitter.split(X):
        pipeline = build_pipeline()
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = pipeline.predict_proba(X.iloc[val_idx])[:, 1]
        scores.append(average_precision_score(y.iloc[val_idx], proba))

    print(f"TimeSeriesSplit average precision: mean={np.mean(scores):.4f} folds={['%.4f' % s for s in scores]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to raw transactions (.parquet or .csv)")
    parser.add_argument("--output-dir", default="model/supervised")
    parser.add_argument("--max-daily-reviews", type=int, default=100)
    parser.add_argument("--fp-review-cost-fraction", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_dataset(args.data)
    feature_cols = [c for c in MODEL_FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols]
    y = df[TARGET_COL]

    # Time-respecting holdout: the last `test_size` fraction of the timeline,
    # never shuffled, so the reported metrics reflect scoring the future
    # from the past, exactly like production inference will.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, shuffle=False)

    time_series_cross_validate(X_train, y_train)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    proba_test = pipeline.predict_proba(X_test)[:, 1]
    n_days = df.loc[X_test.index, TIMESTAMP_COL].dt.date.nunique()

    cost_curve = business_cost_curve(
        y_true=y_test,
        y_proba=proba_test,
        amount=X_test["amount"],
        n_days=n_days,
        fp_review_cost_fraction=args.fp_review_cost_fraction,
    )
    threshold = threshold_for_daily_capacity(cost_curve, args.max_daily_reviews)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
        "average_precision": float(average_precision_score(y_test, proba_test)),
        "precision_at_100": float(precision_at_k(y_test, proba_test, k=100)),
        "n_test_days": int(n_days),
        "chosen_threshold": threshold,
    }
    print("Evaluation metrics:", metrics)

    artifact = ModelArtifact(
        pipeline=pipeline,
        feature_names=feature_cols,
        threshold=threshold,
        metadata={
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "performance_metrics": metrics,
            "max_daily_reviews": args.max_daily_reviews,
            "fp_review_cost_fraction": args.fp_review_cost_fraction,
        },
    )
    model_path = save_artifact(artifact, args.output_dir)
    print(f"Saved artifact to {model_path}")


if __name__ == "__main__":
    main()
