"""Save/load a self-describing model artifact.

The previous artifacts (``model/supervised/*.pkl`` + a sibling
``model_config_*.yaml``) went out of sync with each other: the YAML claimed
``model_type: XGBoost`` and a threshold lived only as a magic number
duplicated in ``app/main.py``, while the pickle itself was whatever
``best_model`` happened to be from the last notebook run (an AdaBoost
pipeline, per the filename). ``ModelArtifact`` bundles the fitted pipeline
together with everything needed to serve it correctly, derived from the
pipeline object itself rather than typed by hand, so the two cannot diverge.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
import yaml
from imblearn.pipeline import Pipeline


@dataclass
class ModelArtifact:
    pipeline: Pipeline
    feature_names: List[str]
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X) -> Any:
        return self.pipeline.predict_proba(X[self.feature_names])[:, 1]

    @property
    def model_type(self) -> str:
        return type(self.pipeline.named_steps["model"]).__name__


def save_artifact(
    artifact: ModelArtifact,
    output_dir: str,
    version: Optional[str] = None,
) -> str:
    """Persist a ``ModelArtifact`` as a single pickle plus a readable YAML card.

    Returns the path to the saved pickle.
    """
    os.makedirs(output_dir, exist_ok=True)
    version = version or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    model_path = os.path.join(output_dir, f"model_{version}.pkl")
    joblib.dump(artifact, model_path)

    config = {
        "version": version,
        "model_type": artifact.model_type,
        "feature_names": artifact.feature_names,
        "threshold": artifact.threshold,
        "model_parameters": artifact.pipeline.named_steps["model"].get_params(),
        "sampling_strategy": artifact.pipeline.named_steps["sampler"].get_params()["sampling_strategy"],
        **artifact.metadata,
    }
    config_path = os.path.join(output_dir, f"model_config_{version}.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return model_path


def load_artifact(model_path: str) -> ModelArtifact:
    artifact = joblib.load(model_path)
    if not isinstance(artifact, ModelArtifact):
        raise TypeError(
            f"{model_path} does not contain a fraud_detection.registry.ModelArtifact "
            f"(got {type(artifact)!r}). Legacy artifacts must be re-saved with "
            "fraud_detection.registry.save_artifact before they can be served."
        )
    return artifact
