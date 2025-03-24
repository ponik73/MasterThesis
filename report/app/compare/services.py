from sqlmodel import Session, SQLModel
from jinja2 import Template
from typing import List, Dict, Tuple, Any, Literal
from collections import defaultdict

from database.services import get_all_objects_by_properties, get_runs_with_sys_name, get_latest_run
from database.models import Device, Model, Run, Metric, Dataset
from database.enumerated import MetricCategory
from html.schemas.comparison_latency import ComparisonLatencyContext
from html.schemas.comparison_accuracy import ComparisonAccuracyContext
    
def generate_comparison(
    db: Session,
    template: Template,
    comparison_type: Literal['accuracy', 'latency'],
    model_id: int | None = None,
    dataset_id: int | None = None,
    device_name: str | None = None,
    kernel_version_a: str | None = None,
    kernel_version_b: str | None = None
) -> str:
    # Fetch base objects
    all_models: List[Model] = get_all_objects_by_properties(db_object_class=Model, db=db)
    all_devices: List[Device] = get_all_objects_by_properties(db_object_class=Device, db=db)
    
    # Fetch datasets for accuracy comparison
    all_datasets: List[Dataset] = (
        get_all_objects_by_properties(db_object_class=Dataset, db=db) 
        if comparison_type == 'accuracy' 
        else []
    )

    kernel_versions: List[str] = []
    table_rows: List[Tuple[str, str, str, float]] = []

    # Model ID filtering
    if model_id is not None:
        # Validate model ID
        assert model_id in [x.id for x in all_models], "Invalid model ID"
        
        all_devices = _filter_by_related_runs(
            db=db,
            list_filtered=all_devices,
            filter_pairs=[("model_id", model_id)]
        )
        
        # Filter datasets for accuracy comparison
        if comparison_type == 'accuracy':
            all_datasets = _filter_by_related_runs(
                db=db,
                list_filtered=all_datasets,
                filter_pairs=[("model_id", model_id)]
            )

    # Dataset ID filtering (only for accuracy)
    if comparison_type == 'accuracy' and dataset_id is not None:
        # Validate dataset ID
        assert dataset_id in [x.id for x in all_datasets], "Invalid dataset ID"

        # Filter models and devices
        all_models = _filter_by_related_runs(
            db=db,
            list_filtered=all_models,
            filter_pairs=[("dataset_id", dataset_id)]
        )
        
        all_devices = _filter_by_related_runs(
            db=db,
            list_filtered=all_devices,
            filter_pairs=[("dataset_id", dataset_id)]
        )

    # Device name filtering
    if device_name is not None:
        # Validate device name
        assert device_name in [x.sys_name for x in all_devices], "Invalid device name"
        related_runs = get_runs_with_sys_name(db=db, sys_name=device_name)

        # Filter models and datasets
        all_models = [model for model in all_models if model.id in {run.model_id for run in related_runs}]
        
        if comparison_type == 'accuracy':
            all_datasets = [ds for ds in all_datasets if ds.id in {run.dataset_id for run in related_runs}]

        # Get available kernel versions
        kernel_versions = [x.kernel_version for x in all_devices]

    # Kernel version validation
    if kernel_version_a is not None:
        assert kernel_version_a in kernel_versions, "Invalid kernel version A"
    if kernel_version_b is not None:
        assert kernel_version_b in kernel_versions, "Invalid kernel version B"

    # Generate table rows when all required parameters are provided
    required_params = [model_id, device_name, kernel_version_a, kernel_version_b]
    if comparison_type == 'accuracy':
        required_params.append(dataset_id)
    
    if all(x is not None for x in required_params):
        table_rows = _get_table_rows(
            db=db,
            model_id=model_id,
            sys_name=device_name,
            kernel_version_a=kernel_version_a,
            kernel_version_b=kernel_version_b,
            dataset_id=dataset_id,
            comparison_type=comparison_type
        )

    # Create template context based on comparison type
    if comparison_type == 'latency':
        template_context = ComparisonLatencyContext(
            models_by_platform=_sort_by_platform(all_models),
            sys_names=list({x.sys_name for x in all_devices}),
            kernel_versions=kernel_versions,
            table_rows=table_rows,
            selected_model_id=model_id,
            selected_device_name=device_name,
            selected_kernel_a=kernel_version_a,
            selected_kernel_b=kernel_version_b
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
            selected_kernel_b=kernel_version_b
        )

    # Render and return HTML content
    return template.render(template_context)

def _sort_by_platform(objects: List[Model | Dataset]) -> Dict[str, List[Model | Dataset]]:
    sorted_by_platform : Dict[str, List[Model | Dataset]] = defaultdict(list)
    for obj in objects:
        sorted_by_platform[obj.platform.value].append(obj)
    return dict(sorted_by_platform)

def _filter_by_related_runs(
        db: Session,
        list_filtered: List[SQLModel],
        filter_pairs: List[Tuple[str, Any]]
):
    # Get runs related:
    related_runs = get_all_objects_by_properties(
        db_object_class=Run,
        db=db,
        properties=dict(filter_pairs)
    )
    # Filter list:
    attr_name_mapping = {
        Model: "model_id",
        Device: "device_id",
        Dataset: "dataset_id"
    }
    attr_name = attr_name_mapping.get(type(list_filtered[0]))
    return [x for x in list_filtered if x.id in {getattr(run, attr_name) for run in related_runs}]

def _get_table_rows(
        db: Session,
        comparison_type: Literal['accuracy', 'latency'],
        model_id: int,
        sys_name: str,
        kernel_version_a: str,
        kernel_version_b: str,
        dataset_id: int | None = None
) -> List[Tuple[str, str, str, float]]:
    run_a = get_latest_run(db=db, model_id=model_id, sys_name=sys_name, kernel_version=kernel_version_a, dataset_id=dataset_id)
    run_b = get_latest_run(db=db, model_id=model_id, sys_name=sys_name, kernel_version=kernel_version_b, dataset_id=dataset_id)

    metric_category = (
        MetricCategory.ACCURACY if comparison_type == 'accuracy' 
        else MetricCategory.LATENCY
    )

    metrics_a: Dict[str, float] = {
        metric.name: metric.value 
        for metric in run_a.metrics 
        if metric.category == metric_category
    }
    metrics_b: Dict[str, float] = {
        metric.name: metric.value 
        for metric in run_b.metrics 
        if metric.category == metric_category
    }

    # Get all unique metric names from both runs
    table_rows: List[Tuple[str, str, str, float]] = []

    for metric_name, value_a in metrics_a.items():
        value_b = metrics_b.get(metric_name, None)
        if value_b is None:
            continue
        # TODO: prettify
        abs_diff = abs(value_a - value_b)
        value_a = str(value_a)
        value_b = str(value_b)
        table_rows.append(
            (metric_name, value_a, value_b, abs_diff)
        )

    return table_rows
