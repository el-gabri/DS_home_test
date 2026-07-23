"""Single factory for the train/serve model pipeline.

Both ``train.py`` and the loaded artifact used by ``app/main.py`` are this
exact object graph: impute -> undersample (train-time only, a no-op at
inference because ``imblearn.Pipeline`` skips samplers in ``predict*``) ->
classify. There is no second copy of this logic anywhere else.
"""

from typing import Any, Dict, Optional

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

from fraud_detection.preprocessing import NullSignalImputer

DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    "learning_rate": 0.1,
    "max_depth": 10,
    "n_estimators": 300,
    "eval_metric": "aucpr",
    "random_state": 42,
}

DEFAULT_SAMPLING_STRATEGY = 0.25
DEFAULT_RANDOM_STATE = 42


def build_pipeline(
    model_params: Optional[Dict[str, Any]] = None,
    sampling_strategy: float = DEFAULT_SAMPLING_STRATEGY,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Build the imputer -> undersampler -> XGBClassifier pipeline.

    Using ``XGBClassifier`` (not ``XGBRegressor``) is deliberate: this is a
    binary classification problem and ``predict_proba`` is required both by
    ``analyze_xgboost_thresholds``-style threshold analysis and by the
    serving API.
    """
    params = {**DEFAULT_MODEL_PARAMS, "random_state": random_state}
    if model_params:
        params.update(model_params)

    return Pipeline(
        steps=[
            ("imputer", NullSignalImputer()),
            ("sampler", RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)),
            ("model", XGBClassifier(**params)),
        ]
    )
