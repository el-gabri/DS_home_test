"""Runtime configuration for the serving API.

Centralizing this in one ``pydantic-settings`` object replaces the previous
pattern of hardcoded, commented-in/out ``MODEL_PATH`` strings at the top of
``app/main.py`` and a threshold literal (``0.7778``) duplicated twice in the
same file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAUD_API_")

    model_path: str = "model/supervised/model_latest.pkl"
    feature_store_path: str = "model/supervised/merchant_features.csv"
    # Fallback used only if the loaded artifact has no threshold recorded.
    default_threshold: float = 0.7778


settings = Settings()
