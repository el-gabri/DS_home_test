# Python 3.9 cannot satisfy numpy~=2.1 (requires >=3.10); 3.12-slim matches
# the versions pinned in pyproject.toml.
FROM python:3.12-slim

WORKDIR /app

# Install only serving dependencies; training-only deps (optuna, jupyter,
# statsmodels, plotly...) don't belong in the runtime image.
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Application code, the shared fraud_detection package, and the model
# artifacts actually needed at serving time.
COPY app/ ./app
COPY src/ ./src
COPY model/ ./model

ENV PYTHONPATH=/app/src
ENV FRAUD_API_MODEL_PATH=/app/model/supervised/model_latest.pkl
ENV FRAUD_API_FEATURE_STORE_PATH=/app/model/supervised/merchant_features.csv

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
