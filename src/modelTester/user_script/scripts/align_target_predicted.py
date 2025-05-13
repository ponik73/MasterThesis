from pathlib import Path
from typing import Any, Literal

from numpy.typing import NDArray
from user_script.script import RequiredFunction, UserScript


class AlignTargetPredictedFunction(RequiredFunction):
    """Implementation of RequiredFunction for aligning model predictions and targets."""

    def name(self) -> Literal["align_target_predicted"]:
        """Get the expected function name to look for in the user script."""
        return "align_target_predicted"

    def param_names(self) -> list[str]:
        """Get the list of expected parameter names for the function."""
        return ["predictions", "targets"]

    def call(self, predictions: NDArray[Any], targets: NDArray[Any]) -> Any:  # noqa: ANN401
        """Call the loaded user-defined align function with the given predictions and targets.

        :param predictions: The model's raw predictions.
        :param targets: The ground truth annotations or labels.
        """
        kwargs = self._create_params_dict()
        kwargs["predictions"] = predictions
        kwargs["targets"] = targets

        return self.loaded_callable(**kwargs)


class AlignTargetPredictedScript(UserScript):
    """UserScript specialization for loading and validating the 'align_target_predicted' function.

    This script is expected to contain the `align_target_predicted(predictions, targets)` function,
    which aligns raw predictions with target labels for evaluation.
    """

    def __init__(self, script_path: Path) -> None:
        """AlignTargetPredictedScript constructor."""
        super().__init__(script_path)
        self.required_functions.append(AlignTargetPredictedFunction())
