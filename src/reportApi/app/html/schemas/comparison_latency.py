from database.enumerated import SupportedPlatforms
from database.models import Model
from pydantic import BaseModel


class ComparisonLatencyContext(BaseModel):
    """Objects used by comparison latency template."""

    models_by_platform: dict[SupportedPlatforms, list[Model]]
    sys_names: list[str]
    kernel_versions: list[str]
    table_rows: list[tuple[tuple[str, str | None], str, str, float]]  # (name, unit); value A; value B; difference B-A

    selected_model_id: int | None = None
    selected_device_name: str | None = None
    selected_kernel_a: str | None = None
    selected_kernel_b: str | None = None
