from database.enumerated import SupportedPlatforms
from database.models import Dataset

from .comparison_latency import ComparisonLatencyContext


class ComparisonAccuracyContext(ComparisonLatencyContext):
    """Objects used by comparison accuracy template."""

    datasets_by_platform: dict[SupportedPlatforms, list[Dataset]]

    selected_dataset_id: int | None = None
