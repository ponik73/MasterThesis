from typing import Any

from modelEvaluator.exceptions import GenerateReportCallError, RegisterRunCallError
from pydantic_core import Url
from requests import Response

from ..apiCall import api_call


@api_call(method="POST", endpoint="run")
def register_run_call(response: Response | None, url: Url, **kwargs: dict[str, Any]) -> str:
    """Call endpoint for storing run to database."""
    if not response:
        raise RegisterRunCallError

    return int(response.json()["run_id"])


@api_call(method="GET", endpoint="report")
def generate_report_call(response: Response | None, url: Url, **kwargs: dict[str, Any]) -> str:
    """Call endpoint for generation of report for the experiment."""
    if not response:
        raise GenerateReportCallError

    return response.text
