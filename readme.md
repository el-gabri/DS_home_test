# Fraud Detection System

This project implements a real-time fraud detection system for financial transactions using machine learning.

## Project Structure

```
fraud-detection/
├── data/
│   ├── raw/                   # Original parquet files
│   └── processed/             # Cleaned features
├── models/
│   └── supervised/            # Trained supervised models
├── notebooks/
│   ├── 1_EDA.ipynb           # Exploratory analysis
│   └── 2_Model development.ipynb     
├── api/
│   ├── main.py               # FastAPI code
│   └── requirements.txt      # Python dependencies
├── tests/
│   ├── test_models.py        # Unit tests
│   └── test_api.py           # API tests
├── config/                   # Feature mappings
├── scripts/                  # Deployment scripts
└── README.md                # Setup instructions
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install requirements.txt
```

3. Run the API:
```bash
cd api
uvicorn main:api --reload
```

The API will be available at http://localhost:8000

## API Documentation

Once the API is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Making Predictions

Send POST requests to `/predict` endpoint with transaction data:

```python
import requests

transaction = {
    "amount": 1000.00,
    "merchant_id": "MERCH123",
    "is_person": True,
    "is_legal_entity": False,
    "is_mei": False,
    "mc_billpay_br_sum_last_15d": 500.0,
    "mc_billpay_br_count_last_1y": 12,
    "mc_pur_br_sum_last_15d": 300.0,
    "mc_billpay_br_sum_last_1y": 5000.0,
    "mc_pur_br_sum_last_1y": 3000.0
}

response = requests.post("http://localhost:8000/predict", json=transaction)
print(response.json())
```

## Running Tests

```bash
pytest tests/
```

## Model Information

The system uses an XGBoost classifier trained on historical transaction data. Key features:

- Probability threshold: 0.7 (optimized for recall)
- Performance metrics:
  - Recall: ~0.75
  - Precision@100: ~0.80
  - ROC AUC: ~0.95

## Deployment

The API is designed for real-time inference, with each prediction taking ~100ms. For production deployment:

1. Build Docker image:
```bash
docker build -t fraud-detection-api .
```

2. Run container:
```bash
docker run -p 8000:8000 fraud-detection-api
```