# Fraud Detection System

Real-time fraud detection for PIX transactions received by SumUp merchant accounts. See
[plan.md](plan.md) for the full audit and improvement roadmap this codebase was refactored against.

## Project Structure

```
DS_home_test/
├── data/
│   ├── raw/                     # sumup_case.parquet (gitignored, not committed)
│   └── processed/               # intermediate training artifacts (gitignored)
├── src/fraud_detection/         # shared training + serving logic (single source of truth)
│   ├── schema.py                #   canonical raw feature columns
│   ├── features.py               #   temporal feature engineering, amount unit conversion
│   ├── preprocessing.py          #   leakage-safe null-value imputer (sklearn transformer)
│   ├── pipeline.py               #   imputer -> undersampler -> XGBClassifier factory
│   ├── evaluate.py               #   Precision@K / business cost-curve evaluation
│   ├── registry.py               #   ModelArtifact save/load
│   ├── train.py                  #   training CLI (python -m fraud_detection.train)
│   └── config.py                 #   serving settings (env-configurable)
├── model/
│   ├── model_package.py         # thin FraudDetectionModel wrapper around fraud_detection
│   └── supervised/              # model_latest.pkl, model_config_latest.yaml, model card
├── app/
│   ├── main.py                  # FastAPI serving code
│   └── feature_store.py         # merchant profile lookup (CSV fixture; swap for a real store)
├── notebooks/                   # EDA.ipynb, Model development.ipynb, SHAP.ipynb, Unsupervised.ipynb
├── tests/                       # pytest: features, preprocessing, model, API, train/serve parity
├── pyproject.toml               # dependencies (serving + `[dev]` extra for training/notebooks)
├── Dockerfile                   # serving image
└── plan.md                      # refactor plan and rationale
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/el-gabri/DS_home_test.git
```
2. Create a virtual environment and install dependencies (`[dev]` also pulls in everything needed
   for `fraud_detection.train` and the notebooks):
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Training

Place the raw dataset at `data/raw/sumup_case.parquet`, then:
```bash
python -m fraud_detection.train --data data/raw/sumup_case.parquet --output-dir model/supervised
```
This fits `fraud_detection.pipeline.build_pipeline()` on a time-respecting split (no shuffling),
sanity-checks it with `TimeSeriesSplit`, picks an operating threshold from the daily review capacity
(`--max-daily-reviews`, default 100), and saves a self-describing `ModelArtifact` (pipeline + feature
names + threshold + metrics) as `model/supervised/model_<timestamp>.pkl` plus a matching
`model_config_<timestamp>.yaml`. Copy (or symlink) the artifact you want served to
`model/supervised/model_latest.pkl`, which is what the API loads by default.

## Running the API

```bash
uvicorn app.main:app --reload
```
The API will be available at http://localhost:8000. Configure it via environment variables (see
`fraud_detection/config.py`): `FRAUD_API_MODEL_PATH`, `FRAUD_API_FEATURE_STORE_PATH`.

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

### Making Predictions

A PIX-received event only carries the merchant, the amount and the timestamp — everything else the
model needs (the ~90 `mc_*`/`mcc_*` rolling aggregates) is resolved server-side from the merchant's
profile (see `app/feature_store.py`; a real deployment queries a live feature store here instead of
a CSV).

```python
import requests

transaction = {
    "merchant_id": "MERCH123",
    "amount": 1000.00,
    "event_created_at": "2026-07-22T14:30:00Z",  # optional, defaults to now
}

response = requests.post("http://localhost:8000/api/v1/predict", json=transaction)
print(response.json())
# {
#   "transaction_id": "...", "merchant_id": "MERCH123",
#   "fraud_probability": 0.03, "review_required": false,
#   "threshold": 0.919, "model_version": "...", "timestamp": "..."
# }
```

## Running Tests

```bash
pytest
```
Tests train against a small synthetic dataset with the real schema (there is no raw parquet
committed to the repo), and include a train/serve parity test that fails if the API's scoring path
ever drifts from the trained pipeline.

## Model Information

See [model/supervised/model_card_latest.md](model/supervised/model_card_latest.md) for the full
write-up. Summary, evaluated on a 56-day time-respecting holdout never seen during training:

- ROC AUC: 0.978
- Average Precision: 0.423
- **Precision@100** (top 100 highest-scored transactions per day): **0.77**
- Operating threshold: 0.919, chosen to keep the daily review queue within a 100/day capacity

## Deployment

```bash
docker build -t fraud-detection-app .
docker run -p 8000:8000 fraud-detection-app
```
The image expects `model/supervised/model_latest.pkl` (and, optionally,
`model/supervised/merchant_features.csv`) to exist in the build context.
