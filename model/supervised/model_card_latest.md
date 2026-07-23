# Fraud Detection Model Card

## Model Information
- Model type: XGBoost Classifier (`fraud_detection.pipeline.build_pipeline`)
- Pipeline steps: `NullSignalImputer` -> `RandomUnderSampler` (train-time only) -> `XGBClassifier`
- Training data: `data/raw/sumup_case.parquet` (991,965 rows after de-duplication, 2023-07-01 to 2024-05-01)
- Artifact: `model_latest.pkl` / `model_config_latest.yaml`

## Evaluation protocol
- Time-respecting holdout: last 20% of the timeline by `event_created_at` (56 days), never shuffled.
- Model-selection sanity check: 5-fold `TimeSeriesSplit` average precision on the training window
  (mean 0.266 across folds) — used to catch overfitting to a single split, not for final reporting.
- This replaces the original notebook's `StratifiedKFold` grid search, which shuffled time within
  training folds and let the model see the future relative to parts of its own validation data.

## Performance on the holdout (last 56 days, never touched during training)
- ROC AUC: 0.978
- Average Precision: 0.423
- **Precision@100 (top 100 highest-scored transactions): 0.77**
- Chosen threshold (lowest threshold whose daily review volume stays within a 100/day capacity): 0.919

These numbers are lower than the previous `Precision@100 = 0.88` reported in earlier model cards.
That is expected and is a feature, not a regression: the earlier number was produced with a null
imputer fit (median + fraud-rate ratio) over train+test combined, and cross-validation that
shuffled time — both leak information unavailable at serving time. This card's numbers reflect what
the model can actually do on unseen future transactions.

## Preprocessing
- `NullSignalImputer` (`fraud_detection/preprocessing.py`): for each numeric column, if
  `fraud_rate(null) / fraud_rate(not null) > 10` on the **training fold only**, missingness is kept
  as a signal (sentinel fill); otherwise the column is median-imputed using the **training fold's**
  median. Never uses information from outside the fold it is fit on.
- `RandomUnderSampler(sampling_strategy=0.25)`: applied only inside `.fit()`; `imblearn.Pipeline`
  skips sampling steps during `.predict()`/`.predict_proba()`, so it never touches
  evaluation or serving data.
- Temporal features (`is_weekend`, `is_night`, `month`, `day`, `hour`) are derived from each
  transaction's own `event_created_at`, both in training and at serving time — see
  `fraud_detection/features.py`.
- `amount` is converted from raw integer cents to reais (`/100`) once, at training data ingestion
  (`fraud_detection/train.py::load_dataset`). The serving API expects `amount` already in reais from
  the caller, matching the `Transaction` schema's documented example.

## Known limitations / next steps (see plan.md for the full list)
- Hyperparameters are the defaults in `fraud_detection/pipeline.py`, not the result of a systematic
  search — Optuna is listed as a dependency but not yet wired into `train.py`.
- `scale_pos_weight`/class-weighting vs. undersampling has not been compared head-to-head.
- No probability calibration; the chosen operating threshold is sensitive to any drift in the raw
  score distribution.
- `merchant_features.csv` (the feature-store fixture used by the demo API) is a static snapshot of
  each merchant's *last known* profile in the training window — a real deployment would query a live
  feature store instead.
