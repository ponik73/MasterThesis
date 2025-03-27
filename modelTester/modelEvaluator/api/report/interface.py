from configuration import get_settings
from configurationHandler.schemas import Dataset, Device, Model

from modelEvaluator.accuracy.metrics.metric import MetricResult
from modelEvaluator.api.report.calls import generate_report_call, register_run_call


class ReportInterface:
    def __init__(self):
        self.url = get_settings().REPORT_API_URL

    def register_run(
        self, device: Device, model: Model, dataset: Dataset, metrics: tuple[list[MetricResult], list[MetricResult]]
    ) -> int | None:
        try:
            latency_metrics, accuracy_metrics = metrics
            # latency_metrics_dict = [x.model_dump() for x in latency_metrics]
            callArgs = {
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

            return register_run_call(url=self.url, **callArgs)
        except Exception as e:
            print(f"Unable to register run. Reason: {e}")
            return None

    def generate_report(self, run_ids: list[int]) -> str | None:
        try:
            callArgs = {
                "headers": {
                    "accept": "text/html",
                },
                "params": {
                    "run_ids": run_ids,
                },
            }

            return generate_report_call(url=self.url, **callArgs)
        except Exception as e:
            print(f"Unable to generate report. Reason: {e}")
            return None
