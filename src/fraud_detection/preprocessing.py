"""Null-value treatment as a proper, leakage-safe sklearn transformer.

The original implementation (see ``app/main.py`` and
``notebooks/Model development.ipynb`` history) decided the imputation
strategy per column, and computed the fallback median, over the *entire*
dataset (train + test) using the label. That leaks test-set and label
information into the features used to train the model, which inflates every
offline metric.

``NullSignalImputer`` fixes this by being a normal ``fit``/``transform``
transformer: ``fit`` only ever sees the training fold (whatever a
``Pipeline``/``GridSearchCV`` hands it), and ``transform`` applies the
strategy learned there — including at serving time, where there is no label
at all.
"""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# If nulls are this many times more likely to be fraud than non-nulls, treat
# "missing" itself as a signal (sentinel value) instead of imputing a normal
# central-tendency value that would erase that signal.
DEFAULT_SIGNAL_RATIO_THRESHOLD = 10.0
SENTINEL_VALUE = -999_999.0


class NullSignalImputer(BaseEstimator, TransformerMixin):
    """Impute numeric nulls, preserving "missingness is a fraud signal".

    Parameters
    ----------
    signal_ratio_threshold:
        Minimum ratio of ``fraud_rate(null) / fraud_rate(not null)`` for a
        column to be treated as "missing is signal" (sentinel fill) rather
        than "missing at random" (median fill).
    """

    def __init__(self, signal_ratio_threshold: float = DEFAULT_SIGNAL_RATIO_THRESHOLD):
        self.signal_ratio_threshold = signal_ratio_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NullSignalImputer":
        if y is None:
            raise ValueError(
                "NullSignalImputer requires y at fit time to decide, per "
                "column, whether missingness correlates with fraud. This is "
                "only ever used during training, never at serving time."
            )

        y = pd.Series(np.asarray(y), index=X.index)
        numeric_cols = X.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns

        self.medians_ = {}
        self.use_sentinel_ = {}
        self.columns_seen_ = list(X.columns)

        for column in numeric_cols:
            is_null = X[column].isnull()
            if not is_null.any():
                continue

            fraud_rate_null = y[is_null].mean()
            fraud_rate_not_null = y[~is_null].mean()
            ratio = (
                fraud_rate_null / fraud_rate_not_null
                if fraud_rate_not_null > 0
                else np.inf
            )

            self.use_sentinel_[column] = ratio > self.signal_ratio_threshold
            if not self.use_sentinel_[column]:
                self.medians_[column] = X[column].median()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()

        missing_cols = set(self.columns_seen_) - set(out.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns at transform time: {sorted(missing_cols)}")

        for column, use_sentinel in self.use_sentinel_.items():
            if column not in out.columns:
                continue
            fill_value = SENTINEL_VALUE if use_sentinel else self.medians_[column]
            out[column] = out[column].fillna(fill_value)

        # Any remaining numeric nulls (column had no nulls in the training
        # fold but does now, e.g. a new feature-store gap) fall back to 0
        # rather than propagating NaN into the model.
        numeric_cols = out.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns
        out[numeric_cols] = out[numeric_cols].fillna(0)

        return out
