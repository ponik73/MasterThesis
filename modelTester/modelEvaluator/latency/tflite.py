import re

from modelEvaluator.accuracy.metrics.metric import MetricResult

from .parser import LatencyParser


class TFliteLatencyParser(LatencyParser):
    """Parser for raw latency assessment results concerning TFLite framework."""

    def parse(self, data: str) -> list[MetricResult]:
        """Parse TFLite latency data."""
        # Regexes for the metrics:
        metric_re = {
            "init": r"Init:\s*(?P<value>[\d\.]+)",
            "firstInference": r"First inference:\s*(?P<value>[\d\.]+)",
            "warmup": r"Warmup \(avg\):\s*(?P<value>[\d\.]+)",
            "inference": r"Inference \(avg\):\s*(?P<value>[\d\.]+)",
        }
        unit = "us"
        metrics: list[MetricResult] = []

        for metric, pattern in metric_re.items():
            # Search for the pattern and extract the value
            match = re.search(pattern, data)
            # Skip if metric not matched:
            if not match:
                continue
            # Add metric to the list:
            metrics.append(MetricResult(name=metric, value=match.group("value").strip(), unit=unit))

        return metrics
