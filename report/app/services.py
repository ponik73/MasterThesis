from sqlmodel import Session
from fastapi import HTTPException
from typing import List, Tuple, Dict
from jinja2.environment import Template
from pydantic import BaseModel
from itertools import groupby

from database.models import Device, Model, ModelBase, Dataset, DatasetBase, Run, Metric, DeviceBase, MetricBase
from database.services import get_or_create_record, insert_multiple_into_db, insert_single_into_db, get_multiple_by_ids
from database.enumerated import MetricCategory
from html.schemas.report import DeviceResults, RunResults

def register_run_service(
    device: DeviceBase,
    model: ModelBase,
    latency_metrics: List[MetricBase],
    dataset: DatasetBase,
    accuracy_metrics: List[MetricBase],
    db: Session
) -> int:
    """Register a new run, creating dependencies if necessary."""
    try:
        # Get device id by fingerprint; create if not exists:
        device_db_object : Device = get_or_create_record(
            object_data=device,
            db_object_class=Device,
            db=db
        )
        # Get model id by resource; create if not exists:
        model_db_object : Model = get_or_create_record(
            object_data=model,
            db_object_class=Model,
            db=db
        )
        # Get dataset id by resource; create if not exists:
        dataset_db_object : Dataset = get_or_create_record(
            object_data=dataset,
            db_object_class=Dataset,
            db=db
        )

        # Create record of the run:
        run_db_object : Run = insert_single_into_db(
            db_object=Run(
                device_id=device_db_object.id,
                model_id=model_db_object.id,
                dataset_id=dataset_db_object.id
            ),
            db=db
        )

        # Create record for each metric:
        insert_multiple_into_db(
            db=db,
            db_object=[
                Metric(name=metric.name, value=metric.value, run_id=run_db_object.id, category="latency") for metric in latency_metrics
            ]
        )
        insert_multiple_into_db(
            db=db,
            db_object=[
                Metric(name=metric.name, value=metric.value, run_id=run_db_object.id, category="accuracy") for metric in accuracy_metrics
            ]
        )

        return run_db_object.id
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    

def generate_report_service(
    run_ids: List[int],
    db: Session,
    template: Template
):
    # Get Run database object for each identifier:
    runs : List[Run] = get_multiple_by_ids(
        db_object_class=Run,
        identifiers=run_ids,
        db=db
    )

    # Group runs by device_id:
    runs_by_device: Dict[int, List[Run]] = {
        device_id: list(group) for device_id, group in groupby(runs, key=lambda r: r.device_id)
    }

    # Parse the runs into template objects:
    results : List[DeviceResults] = []
    for _, runs in runs_by_device.items():
        device : Device | None = None
        results_device : List[RunResults] = []

        for run in runs:
            metrics_by_category : Dict[MetricCategory, List[Metric]] = {
               category: list(group) for category, group in groupby(run.metrics, key=lambda r: r.category)
            }

            results_device.append(
                RunResults(
                    model=run.model,
                    latency_results=metrics_by_category.get(MetricCategory.LATENCY),
                    dataset=run.dataset,
                    accuracy_results=metrics_by_category.get(MetricCategory.ACCURACY)
                )
            )
            if device is None:
                device = run.device
        
        results.append(
            DeviceResults(device=device, results=results_device)
        )
    
    # Create html report:
    html_content = template.render(results=results)
    return html_content