from typing import List, Dict, Tuple
from pydantic import BaseModel

from database.models import Model, Metric, Dataset, Device
from database.enumerated import SupportedPlatforms


class ComparisonLatencyContext(BaseModel):
      models_by_platform: Dict[SupportedPlatforms, List[Model]]
      sys_names: List[str]
      kernel_versions: List[str]
      table_rows: List[Tuple[Tuple[str, str], str, str, float]] # (name, unit); value A; value B; difference B-A

      selected_model_id: int | None = None
      selected_device_name: str | None = None
      selected_kernel_a: str | None = None
      selected_kernel_b: str | None = None