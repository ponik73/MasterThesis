from pathlib import Path
from functools import lru_cache
from pydantic import BaseModel
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_core import Url

class DockerHubPath(BaseModel):
    username: str
    image: str
    version: Literal["latest"]

    def __str__(self):
        return f"{self.username}/{self.image}:{self.version}"

class Settings(BaseSettings):
    """App global configuration properties."""

    DOCKER_DEVICE_API_PATH : DockerHubPath = DockerHubPath(
        username="ponik73",
        image="device-api",
        version="latest"
        )
    
    REPORT_API_URL : Url = Url("http://localhost:8001")
    REPORT_SAVE_PATH : Path = Path("./reports")
    REPORT_OPEN_IN_BROWSER : bool = True

@lru_cache
def getSettings() -> Settings:
    """Cache function for Setting class instance."""
    return Settings()