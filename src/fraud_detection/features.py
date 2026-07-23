"""Feature engineering shared by training and serving.

The one hard rule here: every derived feature must be computable from data
that is genuinely available at prediction time (the transaction itself, plus
whatever the feature store returns for the merchant). Nothing here may look
at the label or at data the server's wall clock happens to show.
"""

import pandas as pd

from fraud_detection.schema import TIMESTAMP_COL


def normalize_amount(df: pd.DataFrame, amount_col: str = "amount") -> pd.DataFrame:
    """Convert the raw ``amount`` column (integer cents, per
    ``sumup_case.parquet``'s ``int32`` dtype and its minimum value of 10000)
    into reais.

    This is a one-time ingestion step applied when loading historical data
    for training (see ``fraud_detection.train.load_dataset``) — it is *not*
    applied again in the serving path, because the API expects callers to
    submit ``amount`` already in reais (as documented in the README example
    and the ``Transaction`` schema). The previous serving code instead
    invented its own ad-hoc scale factor (``amount * 0.1``) that matched
    neither this ``/100`` conversion nor the caller's actual units — silently
    corrupting every prediction.
    """
    out = df.copy()
    out[amount_col] = out[amount_col] / 100
    return out


def add_temporal_features(df: pd.DataFrame, timestamp_col: str = TIMESTAMP_COL) -> pd.DataFrame:
    """Derive weekend/night/month/day/hour flags from a transaction timestamp.

    Critically, this reads ``timestamp_col`` (the transaction's own
    ``event_created_at``), never ``datetime.now()``. The previous serving
    code computed these from the server clock at inference time, which is
    wrong even for real-time scoring (the transaction timestamp and the
    scoring timestamp are not guaranteed to match, e.g. under retries or
    batch backfills) and made offline replay impossible to reproduce.
    """
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col])

    out["is_weekend"] = (ts.dt.weekday >= 5).astype(int)
    out["is_night"] = ((ts.dt.hour < 6) | (ts.dt.hour >= 18)).astype(int)
    out["month"] = ts.dt.month
    out["day"] = ts.dt.day
    out["hour"] = ts.dt.hour

    return out
