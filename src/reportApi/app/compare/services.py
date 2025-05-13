from collections import defaultdict
from html.schemas.comparison_accuracy import ComparisonAccuracyContext
from html.schemas.comparison_latency import ComparisonLatencyContext
from typing import Any, Literal

from database.enumerated import MetricCategory
from database.models import Dataset, Device, Model, Run
from database.services import get_all_objects_by_properties, get_latest_run, get_runs_with_sys_name
from exceptions import InvalidDatasetIdError, InvalidDeviceNameError, InvalidKernelError, InvalidModelIdError
from jinja2 import Template
from sqlmodel import Session, SQLModel


def generate_comparison(
    db: Session,
    template: Template,
    comparison_type: Literal["accuracy", "latency"],
    model_id: int | None = None,
    dataset_id: int | None = None,
    device_name: str | None = None,
    kernel_version_a: str | None = None,
    kernel_version_b: str | None = None,
) -> str:
    """Generate an HTML comparison report (either latency or accuracy) using provided filters.

    :param db: SQLAlchemy database session.
    :param template: Jinja2 template used for rendering the comparison report.
    :param comparison_type: The type of comparison report to generate. Either 'accuracy' or 'latency'.
    :param model_id: ID of the model to filter results by.
    :param dataset_id: ID of the dataset (only applicable for accuracy comparison).
    :param device_name: System name of the device to filter results by.
    :param kernel_version_a: Kernel version for the first side of comparison.
    :param kernel_version_b: Kernel version for the second side of comparison.

    :return: Rendered HTML string of the comparison report.
    """
    # Fetch base objects
    all_models: list[Model] = get_all_objects_by_properties(db_object_class=Model, db=db)
    all_devices: list[Device] = get_all_objects_by_properties(db_object_class=Device, db=db)

    # Fetch datasets for accuracy comparison
    all_datasets: list[Dataset] = (
        get_all_objects_by_properties(db_object_class=Dataset, db=db) if comparison_type == "accuracy" else []
    )

    kernel_versions: list[str] = []
    table_rows: list[tuple[tuple[str, str | None], str, str, float]] = []

    # Model ID filtering:
    all_devices, all_datasets = _filter_by_model_id(
        db=db,
        comparison_type=comparison_type,
        model_id=model_id,
        list_model_ids=[x.id for x in all_models],
        list_devices=all_devices,
        list_datasets=all_datasets,
    )

    # Dataset ID filtering (only for accuracy):
    if comparison_type == "accuracy" and dataset_id is not None:
        all_models, all_devices = _filter_by_dataset_id(
            db=db,
            dataset_id=dataset_id,
            list_dataset_ids=[x.id for x in all_datasets],
            list_devices=all_devices,
            list_models=all_models,
        )

    # Device name filtering
    all_models, all_datasets, kernel_versions = _filter_by_device_name(
        db=db,
        comparison_type=comparison_type,
        device_name=device_name,
        list_devices=all_devices,
        list_datasets=all_datasets,
        list_models=all_models,
    )

    # Kernel version validation
    _validate_kernel_versions(kernel_version_a, kernel_version_b, kernel_versions)

    # Generate table rows when all required parameters are provided
    required_params = [model_id, device_name, kernel_version_a, kernel_version_b]
    if comparison_type == "accuracy":
        required_params.append(dataset_id)

    if all(x is not None for x in required_params):
        table_rows = _get_table_rows(
            db=db,
            model_id=model_id,
            sys_name=device_name,
            kernel_version_a=kernel_version_a,
            kernel_version_b=kernel_version_b,
            dataset_id=dataset_id,
            comparison_type=comparison_type,
        )

    # Create template context based on comparison type
    if comparison_type == "latency":
        template_context = ComparisonLatencyContext(
            models_by_platform=_sort_by_platform(all_models),
            sys_names=list({x.sys_name for x in all_devices}),
            kernel_versions=kernel_versions,
            table_rows=table_rows,
            selected_model_id=model_id,
            selected_device_name=device_name,
            selected_kernel_a=kernel_version_a,
            selected_kernel_b=kernel_version_b,
        )
    else:  # accuracy
        template_context = ComparisonAccuracyContext(
            models_by_platform=_sort_by_platform(all_models),
            datasets_by_platform=_sort_by_platform(all_datasets),
            sys_names=list({x.sys_name for x in all_devices}),
            kernel_versions=kernel_versions,
            table_rows=table_rows,
            selected_model_id=model_id,
            selected_dataset_id=dataset_id,
            selected_device_name=device_name,
            selected_kernel_a=kernel_version_a,
            selected_kernel_b=kernel_version_b,
        )

    # Render and return HTML content
    return template.render(template_context)


def _filter_by_device_name(
    db: Session,
    comparison_type: Literal["accuracy", "latency"],
    device_name: str,
    list_devices: list[Device],
    list_datasets: list[Dataset],
    list_models: list[Model],
) -> tuple[list[Model], list[Dataset], list[str]]:
    if device_name is None:
        return list_models, list_datasets, []
    # Validate device name
    if device_name not in [x.sys_name for x in list_devices]:
        raise InvalidDeviceNameError

    related_runs = get_runs_with_sys_name(db=db, sys_name=device_name)

    # Filter models and datasets
    list_models = [model for model in list_models if model.id in {run.model_id for run in related_runs}]

    if comparison_type == "accuracy":
        list_datasets = [ds for ds in list_datasets if ds.id in {run.dataset_id for run in related_runs}]

    # Get available kernel versions
    return list_models, list_datasets, [x.kernel_version for x in list_devices]


def _filter_by_dataset_id(
    db: Session, dataset_id: int, list_dataset_ids: list[int], list_devices: list[Device], list_models: list[Model]
) -> tuple[list[Model], list[Device]]:
    # Validate dataset ID
    if dataset_id not in list_dataset_ids:
        raise InvalidDatasetIdError

    # Filter models and devices
    list_models = _filter_by_related_runs(db=db, list_filtered=list_models, filter_pairs=[("dataset_id", dataset_id)])

    list_devices = _filter_by_related_runs(db=db, list_filtered=list_devices, filter_pairs=[("dataset_id", dataset_id)])

    return list_models, list_devices


def _filter_by_model_id(
    db: Session,
    comparison_type: Literal["accuracy", "latency"],
    model_id: int,
    list_model_ids: list[int],
    list_devices: list[Device],
    list_datasets: list[Dataset],
) -> tuple[list[Device], list[Dataset]]:
    if model_id is None:
        return list_devices, list_datasets

    # Validate model ID
    if model_id not in list_model_ids:
        raise InvalidModelIdError

    list_devices = _filter_by_related_runs(db=db, list_filtered=list_devices, filter_pairs=[("model_id", model_id)])

    # Filter datasets for accuracy comparison
    if comparison_type == "accuracy":
        list_datasets = _filter_by_related_runs(
            db=db, list_filtered=list_datasets, filter_pairs=[("model_id", model_id)]
        )

    return list_devices, list_datasets


def _validate_kernel_versions(version_a: str | None, version_b: str | None, list_versions: list[str]) -> None:
    if version_a is not None and version_a not in list_versions:
        raise InvalidKernelError
    if version_b is not None and version_b not in list_versions:
        raise InvalidKernelError


def _sort_by_platform(objects: list[Model | Dataset]) -> dict[str, list[Model | Dataset]]:
    sorted_by_platform: dict[str, list[Model | Dataset]] = defaultdict(list)
    for obj in objects:
        sorted_by_platform[obj.platform.value].append(obj)
    return dict(sorted_by_platform)


def _filter_by_related_runs(
    db: Session, list_filtered: list[SQLModel], filter_pairs: list[tuple[str, Any]]
) -> list[SQLModel]:
    # Get runs related:
    related_runs = get_all_objects_by_properties(db_object_class=Run, db=db, properties=dict(filter_pairs))
    # Filter list:
    attr_name_mapping = {Model: "model_id", Device: "device_id", Dataset: "dataset_id"}
    attr_name = attr_name_mapping.get(type(list_filtered[0]))
    return [x for x in list_filtered if x.id in {getattr(run, attr_name) for run in related_runs}]


def _get_table_rows(
    db: Session,
    comparison_type: Literal["accuracy", "latency"],
    model_id: int,
    sys_name: str,
    kernel_version_a: str,
    kernel_version_b: str,
    dataset_id: int | None = None,
) -> list[tuple[str, str, str, float]]:
    run_a = get_latest_run(
        db=db, model_id=model_id, sys_name=sys_name, kernel_version=kernel_version_a, dataset_id=dataset_id
    )
    run_b = get_latest_run(
        db=db, model_id=model_id, sys_name=sys_name, kernel_version=kernel_version_b, dataset_id=dataset_id
    )

    metric_category = MetricCategory.ACCURACY if comparison_type == "accuracy" else MetricCategory.LATENCY

    metrics_a: dict[str, tuple[float, str]] = {
        metric.name: (metric.unit, metric.value) for metric in run_a.metrics if metric.category == metric_category
    }
    metrics_b: dict[str, tuple[float, str]] = {
        metric.name: (metric.unit, metric.value) for metric in run_b.metrics if metric.category == metric_category
    }

    # Get all unique metric names from both runs
    table_rows: list[tuple[tuple[str, str | None], str, str, float]] = []

    for metric_name, (unit, value_a) in metrics_a.items():
        _, value_b = metrics_b.get(metric_name)
        if value_b is None:
            continue
        abs_diff = abs(value_a - value_b)
        table_rows.append(((metric_name, unit), str(value_a), str(value_b), abs_diff))

    return table_rows
