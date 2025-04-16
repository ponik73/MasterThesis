from modelEvaluator.exceptions import FrameworkNotSupportedError

from .parser import LatencyParser
from .tflite import TFliteLatencyParser

PARSER_REGISTRY = {
    "tflite": TFliteLatencyParser,
}


def latency_parser_factory(model_framework: str) -> LatencyParser:
    """Return the corresponding parser instance based on model framework."""
    parser_class = PARSER_REGISTRY.get(model_framework.lower())
    if not parser_class:
        raise FrameworkNotSupportedError(model_framework)
    return parser_class()
