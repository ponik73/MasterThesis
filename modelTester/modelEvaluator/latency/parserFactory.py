from .parser import LatencyParser
from .tflite import TFliteLatencyParser

PARSER_REGISTRY = {
    "tflite": TFliteLatencyParser,
}

def latencyParserFactory(modelFramework: str) -> LatencyParser:
    """Returns the corresponding parser instance based on model framework."""
    parser_class = PARSER_REGISTRY.get(modelFramework.lower())
    if not parser_class:
        raise ValueError(f"Model framework `{modelFramework}` not supported.")
    return parser_class()
