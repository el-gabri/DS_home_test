"""Business-facing evaluation metrics.

The fraud review team can only look at a bounded number of cases per day.
That makes ranking-based metrics (Precision@K, expected loss captured in the
top-K) the metrics that actually matter for this product, not F1 — F1 treats
every transaction as equally reviewable, which the analysts' capacity
contradicts. This module extracts and fixes the ad-hoc threshold analysis
that lived inline in ``notebooks/Model development.ipynb``
(``analyze_xgboost_thresholds``), which mutated its input with a
``SettingWithCopyWarning``-prone assignment and recomputed a fresh
train/test split (and thus a fresh model) every time it was called.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def precision_at_k(y_true: pd.Series, y_proba: np.ndarray, k: int) -> float:
    """Precision among the top-``k`` highest-scored transactions."""
    order = np.argsort(y_proba)[::-1][:k]
    y_true_arr = np.asarray(y_true)
    if len(order) == 0:
        return float("nan")
    return float(y_true_arr[order].mean())


@dataclass
class ThresholdCandidate:
    threshold: float
    recall: float
    precision: float
    n_reviews_per_day: float
    expected_loss_captured: float
    total_cost_per_day: float


def business_cost_curve(
    y_true: pd.Series,
    y_proba: np.ndarray,
    amount: pd.Series,
    n_days: int,
    fp_review_cost_fraction: float = 0.1,
    n_thresholds: int = 100,
) -> pd.DataFrame:
    """Cost/benefit of every candidate threshold over a time-bounded test set.

    Parameters
    ----------
    y_true, y_proba, amount:
        Aligned arrays/series for the evaluation set.
    n_days:
        Number of distinct calendar days the evaluation set spans, used to
        turn totals into a *daily* review load and a *monthly* cost estimate
        (mirrors the original notebook's ``max_daily_reviews`` constraint).
    fp_review_cost_fraction:
        Analyst-hour cost of reviewing a false positive, expressed as a
        fraction of the transaction amount (cost of a missed fraud, i.e. a
        false negative, is the full amount).
    """
    if n_days <= 0:
        raise ValueError("n_days must be positive")

    y_true = np.asarray(y_true)
    amount = np.asarray(amount)
    rows: list[dict] = []

    for threshold in np.linspace(0, 1, n_thresholds):
        predictions = (y_proba > threshold).astype(int)

        tp = int(((predictions == 1) & (y_true == 1)).sum())
        fp = int(((predictions == 1) & (y_true == 0)).sum())
        fn = int(((predictions == 0) & (y_true == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        fp_cost = amount[(predictions == 1) & (y_true == 0)].sum() * fp_review_cost_fraction
        fn_cost = amount[(predictions == 0) & (y_true == 1)].sum()
        total_cost = fp_cost + fn_cost

        caught_fraud_amount = amount[(predictions == 1) & (y_true == 1)].sum()
        total_fraud_amount = amount[y_true == 1].sum()

        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "n_reviews_per_day": predictions.sum() / n_days,
                "monthly_cost": (total_cost / n_days) * 30,
                "expected_loss_captured": (
                    caught_fraud_amount / total_fraud_amount if total_fraud_amount > 0 else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def threshold_for_daily_capacity(cost_curve: pd.DataFrame, max_daily_reviews: int) -> float:
    """Pick the lowest threshold whose review load fits the team's capacity.

    A lower threshold always reviews more (monotonically non-increasing in
    threshold), so among thresholds that respect capacity we want the
    smallest one — it maximizes recall subject to the constraint.
    """
    feasible = cost_curve[cost_curve["n_reviews_per_day"] <= max_daily_reviews]
    if feasible.empty:
        return float(cost_curve["threshold"].max())
    return float(feasible["threshold"].min())
