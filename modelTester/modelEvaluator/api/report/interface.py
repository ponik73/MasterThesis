from configuration import get_settings
from configurationHandler.schemas import Dataset, Device, Model
from modelEvaluator.accuracy.metrics.metric import MetricResult
from modelEvaluator.api.report.calls import generate_report_call, register_run_call


class ReportInterface:
    """Interface for interacting with the reporting API."""

    def __init__(self) -> None:
        """ReportInterface constructor."""
        self.url = get_settings().REPORT_API_URL

    def register_run(
        self, device: Device, model: Model, dataset: Dataset, metrics: tuple[list[MetricResult], list[MetricResult]]
    ) -> int | None:
        """Register a run to the database.

        :param device: The device used during evaluation.
        :param model: The model used during evaluation.
        :param dataset: The dataset used during evaluation.
        :param metrics: A tuple of two metric result lists: (latency_metrics, accuracy_metrics).
        :return: The run ID assigned by the report server.
        """
        latency_metrics, accuracy_metrics = metrics
        # latency_metrics_dict = [x.model_dump() for x in latency_metrics]
        call_args = {
            "headers": {"accept": "application/json"},
            "json": {
                "device": {"sys_name": device.sys_name, "kernel_version": device.kernel_version},
                "model": {
                    "framework": model.framework.value,
                    "task": model.task.value,
                    "path": model.uri,
                    "platform": model.platform,
                },
                "latency_metrics": [x.model_dump() for x in latency_metrics],
                "dataset": {
                    "path": dataset.uri,
                    "platform": dataset.platform,
                },
                "accuracy_metrics": [x.model_dump() for x in accuracy_metrics],
            },
        }

        return register_run_call(url=self.url, **call_args)

    def generate_report(self, run_ids: list[int]) -> str | None:
        """Generate an HTML report for the given run IDs."""
        call_args = {
            "headers": {
                "accept": "text/html",
            },
            "params": {
                "run_ids": run_ids,
            },
        }

        return generate_report_call(url=self.url, **call_args)
