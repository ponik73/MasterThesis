from html.schemas.report import DeviceResults, RunResults
from itertools import groupby

from database.enumerated import MetricCategory
from database.models import Dataset, DatasetBase, Device, DeviceBase, Metric, MetricBase, Model, ModelBase, Run
from database.services import get_multiple_by_ids, get_or_create_record, insert_multiple_into_db, insert_single_into_db
from jinja2.environment import Template
from sqlmodel import Session


def register_run_service(
    device: DeviceBase,
    model: ModelBase,
    metrics: tuple[list[MetricBase], list[MetricBase]],
    dataset: DatasetBase,
    db: Session,
) -> int:
    """Register a new run, creating dependencies if necessary."""
    latency_metrics, accuracy_metrics = metrics

    # Get device id by fingerprint; create if not exists:
    device_db_object: Device = get_or_create_record(object_data=device, db_object_class=Device, db=db)
    # Get model id by resource; create if not exists:
    model_db_object: Model = get_or_create_record(object_data=model, db_object_class=Model, db=db)
    # Get dataset id by resource; create if not exists:
    dataset_db_object: Dataset = get_or_create_record(object_data=dataset, db_object_class=Dataset, db=db)

    # Create record of the run:
    run_db_object: Run = insert_single_into_db(
        db_object=Run(device_id=device_db_object.id, model_id=model_db_object.id, dataset_id=dataset_db_object.id),
        db=db,
    )

    # Create record for each metric:
    insert_multiple_into_db(
        db=db,
        db_object=[
            Metric(name=metric.name, value=metric.value, run_id=run_db_object.id, category="latency", unit=metric.unit)
            for metric in latency_metrics
        ],
    )
    insert_multiple_into_db(
        db=db,
        db_object=[
            Metric(name=metric.name, value=metric.value, run_id=run_db_object.id, category="accuracy")
            for metric in accuracy_metrics
        ],
    )

    return run_db_object.id


def generate_report_service(run_ids: list[int], db: Session, template: Template) -> str:
    """Generate a report html page for given runs."""
    # Get Run database object for each identifier:
    runs: list[Run] = get_multiple_by_ids(db_object_class=Run, identifiers=run_ids, db=db)

    # Group runs by device_id:
    runs_by_device: dict[int, list[Run]] = {
        device_id: list(group) for device_id, group in groupby(runs, key=lambda r: r.device_id)
    }

    # Parse the runs into template objects:
    results: list[DeviceResults] = []
    for runs in runs_by_device.values():
        device: Device | None = None
        results_device: list[RunResults] = []

        for run in runs:
            metrics_by_category: dict[MetricCategory, list[Metric]] = {
                category: list(group) for category, group in groupby(run.metrics, key=lambda r: r.category)
            }

            results_device.append(
                RunResults(
                    model=run.model,
                    latency_results=metrics_by_category.get(MetricCategory.LATENCY),
                    dataset=run.dataset,
                    accuracy_results=metrics_by_category.get(MetricCategory.ACCURACY),
                )
            )
            if device is None:
                device = run.device

        results.append(DeviceResults(device=device, results=results_device))

    # Create html report:
    return template.render(results=results)
