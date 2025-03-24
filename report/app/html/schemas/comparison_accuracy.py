from typing import List, Dict

from database.models import Dataset
from database.enumerated import SupportedPlatforms
from .comparison_latency import ComparisonLatencyContext

class ComparisonAccuracyContext(ComparisonLatencyContext):
      datasets_by_platform: Dict[SupportedPlatforms, List[Dataset]]

      selected_dataset_id: int | None = None
