from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """API global configuration properties."""

    MODEL_DIR: Path = Path("./model_storage")
    EXECUTABLES_DIR: Path = Path("./executables")
    TEMP_DIR: Path = Path("./tmp")

@lru_cache
def getSettings() -> Settings:
    """Cache function for Setting class instance."""
    return Settings()