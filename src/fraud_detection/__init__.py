"""Shared training/serving logic for the PIX fraud detection model.

This package exists so that feature engineering and preprocessing are defined
exactly once and imported by both the training pipeline (notebooks, train.py)
and the serving API (app/main.py). Duplicating this logic across the two was
the root cause of the training-serving skew documented in plan.md.
"""

__version__ = "0.1.0"
