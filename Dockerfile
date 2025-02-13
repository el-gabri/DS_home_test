# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /

# Copy requirements first to leverage Docker cache
COPY /requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY / api
COPY / model
COPY / data
COPY / tests

# Set environment variables
ENV PYTHONPATH=/
ENV MODEL_PATH=/models/supervised/xgboost_model_20250212_144931.pkl

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]