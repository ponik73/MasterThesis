import functools
from typing import Literal
import requests
from requests import Response
from pydantic_core import Url


def apiCall(
        method: Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
        endpoint: str       
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(url: Url, **kwargs) -> Response | None:
            try:
                url = Url.build(
                    scheme=url.scheme,
                    host=url.host,
                    port=url.port,
                    path=endpoint
                )
                # Execute request:
                response = requests.request(method, url, **kwargs)
                response.raise_for_status()
                return func(response, url, **kwargs)
            except Exception as e:
                return func(None, url, **kwargs)
        
        return wrapper
    return decorator
