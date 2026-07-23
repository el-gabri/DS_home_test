"""Merchant feature lookup for real-time inference.

A PIX-received event only carries ``merchant_code``, ``amount`` and
``event_created_at`` — it cannot carry the ~90 rolling-window aggregates
(``mc_*``/``mcc_*``) the model was trained on, because those are properties
of the merchant's history, not of this one transaction. In production this
class would be a thin client over whatever online feature store (Feast,
Redis, a materialized view) keeps those aggregates fresh. This file was
previously empty, which is why the old API could never have worked even if
every other bug in it were fixed: nothing supplied the ~90 missing columns.

For this take-home, ``CsvFeatureStore`` loads a small merchant -> features
table (see ``model/supervised/merchant_features.csv``, generated from the
same historical data as the training set) into memory at startup. Unknown
merchants resolve to all-NaN, which ``NullSignalImputer`` (part of the
trained pipeline) then fills using the strategy learned at training time —
the same code path as a genuinely missing value during training.
"""

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from fraud_detection.schema import MERCHANT_ID_COL, MERCHANT_PROFILE_COLS


class FeatureStore:
    """Interface every feature store implementation must satisfy."""

    def get_merchant_features(self, merchant_id: str) -> Dict[str, float]:
        raise NotImplementedError


class CsvFeatureStore(FeatureStore):
    """In-memory feature store backed by a CSV keyed by ``merchant_code``."""

    def __init__(self, path: Optional[str] = None):
        self._table: Optional[pd.DataFrame] = None
        if path and os.path.exists(path):
            table = pd.read_csv(path)
            table[MERCHANT_ID_COL] = table[MERCHANT_ID_COL].astype(str)
            self._table = table.set_index(MERCHANT_ID_COL)

    def get_merchant_features(self, merchant_id: str) -> Dict[str, float]:
        if self._table is not None and merchant_id in self._table.index:
            row = self._table.loc[merchant_id]
            return {col: row[col] for col in MERCHANT_PROFILE_COLS if col in row}

        return {col: np.nan for col in MERCHANT_PROFILE_COLS}
