from pathlib import Path


class UserScriptError(Exception):
    """Base class for user script related errors."""


class ScriptLoadingError(UserScriptError):
    """User script could not be loaded."""

    def __init__(self, script_path: Path) -> None:
        """ScriptLoadingError constructor."""
        super().__init__(f"User script `{script_path}` could not be loaded.")


class RequiredFunctionNotPresentError(UserScriptError):
    """User script does not contain required function."""

    def __init__(self, script_path: Path, function_name: str) -> None:
        """RequiredFunctionNotPresentError constructor."""
        super().__init__(f"User script `{script_path}` does not contain required function `{function_name}.`")


class RequiredFunctionInvalidParametersError(UserScriptError):
    """User script function signature does not contain required parameters."""

    def __init__(self, script_path: Path, function_name: str) -> None:
        """RequiredFunctionInvalidParametersError constructor."""
        super().__init__(
            f"User script `{script_path}`: function {function_name} signature does not contain required parameters."
        )


class FunctionLoadingError(UserScriptError):
    """Function could not be loaded."""

    def __init__(self, function_name: str) -> None:
        """FunctionLoadingError constructor."""
        super().__init__(f"Function {function_name} could not be loaded.")
