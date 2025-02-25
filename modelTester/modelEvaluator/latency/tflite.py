from pathlib import Path
import re
from typing import List

from .parser import LatencyParser, LatencyAssessmentResult, LatencyMetric

class TFliteLatencyParser(LatencyParser):
    def parse(self, data: str) -> LatencyAssessmentResult:
        """Parses TFLite latency data."""
        # Regexes for the metrics:
        metricRe = {
            "init": r"Init:\s*(?P<value>[\d\.]+)",
            "firstInference": r"First inference:\s*(?P<value>[\d\.]+)",
            "warmup": r"Warmup \(avg\):\s*(?P<value>[\d\.]+)",
            "inference": r"Inference \(avg\):\s*(?P<value>[\d\.]+)"
        }
        UNIT = "ms"
        metrics : List[LatencyMetric] = []
        
        for metric, pattern in metricRe.items():
            # Search for the pattern and extract the value
            match = re.search(pattern, data)
            # Skip if metric not matched:
            if not match:
                continue
            # Add metric to the list:
            metrics.append(
                LatencyMetric(
                    name=metric,
                    value=match.group("value").strip(),
                    unit=UNIT
                )                
            )

        return LatencyAssessmentResult(metrics=metrics)
