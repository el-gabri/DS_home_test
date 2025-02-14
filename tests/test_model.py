# tests/test_models.py
import os
import pickle
import numpy as np
import pytest


@pytest.fixture(scope="module")
def load_model():
    model_path = os.path.join("..", "models", "supervised", "xgboost_model_20250212_144931.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def test_model_prediction_shape(load_model):
    # Create a dummy input (adjust based on your feature count)
    dummy_input = np.array([[100.0, 123, 456, 1]])
    prediction = load_model.predict(dummy_input)
    # Check that the prediction has the expected shape or type.
    assert prediction.shape[0] == 1
