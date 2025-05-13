from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """API global configuration properties."""

    MODEL_DIR: Path = Path("./model_storage")
    EXECUTABLES_DIR: Path = Path("./executables")
    TEMP_DIR: Path = Path("./tmp")


@lru_cache
def get_settings() -> Settings:
    """Cache function for Setting class instance."""
    return Settings()
