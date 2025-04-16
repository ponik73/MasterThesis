import functools
from collections.abc import Callable
from typing import Any, Literal

import requests
from pydantic_core import Url
from requests import Response


def api_call(method: Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"], endpoint: str):
    """A decorator for making HTTP API calls with error handling.

    Constructs a full URL using the provided `endpoint`, performs an HTTP request with the specified `method`,
    and then passes the response to the decorated function. If the request fails, the decorated function is still called but with `None` as the response.

    :param method: The HTTP method to use for the request.
    :param endpoint: The API endpoint path (appended to the base URL).
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(url: Url, **kwargs: dict[str, Any]) -> Response | None:
            try:
                url = Url.build(scheme=url.scheme, host=url.host, port=url.port, path=endpoint)
                # Execute request:
                response = requests.request(method, url, **kwargs)
                response.raise_for_status()
                return func(response, url, **kwargs)
            except requests.exceptions.RequestException:
                return func(None, url, **kwargs)

        return wrapper

    return decorator
