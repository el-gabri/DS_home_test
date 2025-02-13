from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import numpy as np

app = FastAPI(title="Fraud Detection API")

# Define your input schema.
# Adjust the feature names and types according to your data.
class Transaction(BaseModel):
    amount: float
    merchant_id: int
    user_id: int
    transaction_type: str
    # Add additional fields as needed

# Global variable to hold the model
model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "supervised", "model.pkl")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/predict", summary="Get fraud prediction for a transaction")
async def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Prepare the data for prediction.
    # Example: convert Transaction instance into the feature array your model expects.
    try:
        input_features = np.array([[
            transaction.amount,
            transaction.merchant_id,
            transaction.user_id,
            # You might need to convert categorical variables such as transaction_type
            # For demonstration, we convert transaction_type to a dummy numeric value.
            1 if transaction.transaction_type.lower() == "online" else 0
            # Append additional processed features here.
        ]])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")

    # Make prediction.
    try:
        prediction = model.predict(input_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model inference: {e}")

    # Assuming a binary classification: 1 for fraud, 0 for non-fraud.
    return {"prediction": int(prediction[0])}
