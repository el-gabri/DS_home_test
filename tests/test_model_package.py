import numpy as np
import pytest
from model.model_package import FraudDetectionModel


def test_train_predict_roundtrip(synthetic_dataset):
    model = FraudDetectionModel()
    model.train(synthetic_dataset)

    proba = model.predict(synthetic_dataset)
    assert proba.shape[0] == len(synthetic_dataset)
    assert np.all((proba >= 0) & (proba <= 1))


def test_predict_before_train_raises(synthetic_dataset):
    model = FraudDetectionModel()
    with pytest.raises(ValueError):
        model.predict(synthetic_dataset)


def test_train_without_label_raises(synthetic_dataset):
    model = FraudDetectionModel()
    with pytest.raises(ValueError):
        model.train(synthetic_dataset.drop(columns=["infraction"]))


def test_save_load_roundtrip_gives_same_predictions(synthetic_dataset, tmp_path):
    model = FraudDetectionModel()
    model.train(synthetic_dataset)
    proba_before = model.predict(synthetic_dataset)

    model.save(str(tmp_path), threshold=0.6)
    saved_files = list(tmp_path.glob("model_*.pkl"))
    assert len(saved_files) == 1

    reloaded = FraudDetectionModel.load(str(saved_files[0]))
    proba_after = reloaded.predict(synthetic_dataset)

    assert np.allclose(proba_before, proba_after)


def test_predict_missing_feature_raises(synthetic_dataset):
    model = FraudDetectionModel()
    model.train(synthetic_dataset)

    incomplete = synthetic_dataset.drop(columns=["amount"])
    with pytest.raises(ValueError, match="Missing features"):
        model.predict(incomplete)
