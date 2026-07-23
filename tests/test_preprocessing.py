import numpy as np
import pandas as pd
import pytest

from fraud_detection.preprocessing import SENTINEL_VALUE, NullSignalImputer


def _make_signal_dataset(n=1000, seed=0):
    """Column 'signal_col' is null far more often for fraud (ratio > 10);
    column 'random_col' is null at roughly the same rate regardless of fraud."""
    rng = np.random.default_rng(seed)
    y = pd.Series((rng.random(n) < 0.1).astype(int))

    signal_col = rng.normal(size=n)
    signal_null = (rng.random(n) < 0.02) | ((y == 1) & (rng.random(n) < 0.9))
    signal_col = pd.Series(signal_col).mask(signal_null)

    random_col = rng.normal(size=n)
    random_null = rng.random(n) < 0.1
    random_col = pd.Series(random_col).mask(random_null)

    X = pd.DataFrame({"signal_col": signal_col, "random_col": random_col})
    return X, y


def test_requires_y_at_fit_time():
    X, _ = _make_signal_dataset()
    with pytest.raises(ValueError):
        NullSignalImputer().fit(X)


def test_high_fraud_ratio_column_gets_sentinel_not_median():
    X, y = _make_signal_dataset()
    imputer = NullSignalImputer().fit(X, y)

    assert bool(imputer.use_sentinel_["signal_col"]) is True
    transformed = imputer.transform(X)
    assert (transformed.loc[X["signal_col"].isnull(), "signal_col"] == SENTINEL_VALUE).all()


def test_random_missing_column_gets_median():
    X, y = _make_signal_dataset()
    imputer = NullSignalImputer().fit(X, y)

    assert bool(imputer.use_sentinel_.get("random_col", False)) is False
    expected_median = X["random_col"].median()
    transformed = imputer.transform(X)
    assert np.isclose(
        transformed.loc[X["random_col"].isnull(), "random_col"].iloc[0],
        expected_median,
    )


def test_fit_only_uses_the_given_fold_not_full_data():
    """The strategy learned on a train split must not depend on rows that
    were held out — this is the leakage bug described in plan.md 3.1.1."""
    X, y = _make_signal_dataset(n=2000, seed=1)
    X_train, y_train = X.iloc[:1000], y.iloc[:1000]
    X_train_fit = NullSignalImputer().fit(X_train, y_train)

    # The training-fold median must come only from the training rows, not
    # from rows outside the slice it was fit on.
    assert X_train_fit.medians_["random_col"] == pytest.approx(X_train["random_col"].median())


def test_transform_fills_unseen_column_nulls_with_zero():
    X, y = _make_signal_dataset()
    imputer = NullSignalImputer().fit(X, y)

    X_new = X.copy()
    X_new["brand_new_col"] = np.nan
    transformed = imputer.transform(X_new)
    assert (transformed["brand_new_col"] == 0).all()


def test_transform_raises_on_missing_expected_column():
    X, y = _make_signal_dataset()
    imputer = NullSignalImputer().fit(X, y)

    with pytest.raises(ValueError):
        imputer.transform(X.drop(columns=["signal_col"]))
