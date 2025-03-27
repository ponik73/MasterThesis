from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_core import Url
from pydantic_settings import BaseSettings


class DockerHubPath(BaseModel):
    username: str
    image: str
    version: Literal["latest"]

    def __str__(self):
        return f"{self.username}/{self.image}:{self.version}"


class Settings(BaseSettings):
    """App global configuration properties."""

    DOCKER_DEVICE_API_PATH: DockerHubPath = DockerHubPath(username="ponik73", image="device-api", version="latest")

    REPORT_API_URL: Url = Url("http://localhost:8001")
    REPORT_SAVE_PATH: Path = Path("./reports")
    REPORT_OPEN_IN_BROWSER: bool = True


@lru_cache
def get_settings() -> Settings:
    """Cache function for Setting class instance."""
    return Settings()
