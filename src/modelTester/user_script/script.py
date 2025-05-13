from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from importlib.util import module_from_spec, spec_from_file_location
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import ModuleType

from user_script.exceptions import (
    RequiredFunctionInvalidParametersError,
    RequiredFunctionNotPresentError,
    ScriptLoadingError,
)


class RequiredFunction(ABC):
    """Abstract base class for defining a required function interface."""

    def __init__(self) -> None:
        """Initialize the RequiredFunction with no loaded callable."""
        self.loaded_callable: Callable | None = None

    @abstractmethod
    def name(self) -> str:
        """Name of the required function."""

    @abstractmethod
    def param_names(self) -> list[str]:
        """List of expected parameter names for the required function."""

    def _create_params_dict(self) -> dict[str, Any]:
        return dict.fromkeys(self.param_names())

    @abstractmethod
    def call(self) -> None:
        """Execute the required function logic."""


class UserScript:
    """Abstract base class for loading and validating user-defined Python scripts."""

    def __init__(self, script_path: Path) -> None:
        """Initialize the UserScript with the path to the user-defined script."""
        self.script_path = script_path
        self.module: ModuleType | None = None
        self.required_functions: list[RequiredFunction] = []

    def get_function_by_name(self, name: str) -> RequiredFunction | None:
        """Retrieve a required function object by its name."""
        for f in self.required_functions:
            if f.name() == name:
                return f
        return None

    def load(self) -> None:
        """Load the user script as a module and validate all required functions."""
        spec = spec_from_file_location("user_script", self.script_path)
        self.module = module_from_spec(spec)
        spec.loader.exec_module(self.module)
        if self.module is None:
            raise ScriptLoadingError(script_path=self.script_path)

        self._validate()

    def _validate(self) -> bool:
        for f in self.required_functions:
            # Module contains object with the required name:
            if not hasattr(self.module, f.name()):
                raise RequiredFunctionNotPresentError(script_path=self.script_path, function_name=f.name())
            module_f = getattr(self.module, f.name())

            # The object is callable:
            if not callable(module_f):
                raise RequiredFunctionNotPresentError(script_path=self.script_path, function_name=f.name())
            module_f_signature = inspect.signature(module_f)

            # The callable parameters match the required parameters:
            module_f_param_names = list(module_f_signature.parameters.keys())
            if len(module_f_param_names) != len(f.param_names()) or set(module_f_param_names) != set(f.param_names()):
                raise RequiredFunctionInvalidParametersError(script_path=self.script_path, function_name=f.name())

            f.loaded_callable = module_f
