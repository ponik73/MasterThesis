from requests import Response
from pydantic_core import Url

from ..apiCall import apiCall

@apiCall(
    method="POST",
    endpoint="run"
)
def register_run_call(response: Response | None, url: Url, **kwargs) -> str:
    if not response:
        raise
    
    return int(response.json()["run_id"])

@apiCall(
    method="GET",
    endpoint="report"
)
def generate_report_call(response: Response | None, url: Url, **kwargs) -> str:
    if not response:
        raise

    return response.text
