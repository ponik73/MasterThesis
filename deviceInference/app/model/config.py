from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """API global configuration properties."""

    MODEL_DIR: Path = Path("./model_storage")
    TEMP_DIR: Path = Path("./tmp")
    SUPPORTED_FRAMEWORKS: list[str] = ["tflite"]

@lru_cache
def get_settings() -> Settings:
    """Cache function for Setting class instance."""
    return Settings()