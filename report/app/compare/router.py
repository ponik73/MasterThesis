from typing import Annotated
from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlmodel import Session

from database.services import get_session
from html.environment import jinja_env
from .services import generate_comparison


compare_router = APIRouter(
    prefix="/compare",
    tags=["compare"]
)

@compare_router.get("/latency", response_class=HTMLResponse)
def get_comparison_latency(
    db: Annotated[Session, Depends(get_session)],
    model_id: int | None = None,
    device_name: str | None = None,
    kernel_version_a: str | None = None,
    kernel_version_b: str | None = None,
    # best_run: bool | None = None,
    # date_from: str | None = None,
    # date_until: str | None = None,
):
    template = jinja_env.get_template("comparison_latency.html")
    return HTMLResponse(
        content=generate_comparison(
            comparison_type='latency',
            db=db,
            template=template,
            model_id=model_id,
            device_name=device_name,
            kernel_version_a=kernel_version_a,
            kernel_version_b=kernel_version_b
        )
    )

@compare_router.get("/accuracy", response_class=HTMLResponse)
def get_comparison_accuracy(
    db: Annotated[Session, Depends(get_session)],
    model_id: int | None = None,
    dataset_id: int | None = None,
    device_name: str | None = None,
    kernel_version_a: str | None = None,
    kernel_version_b: str | None = None,
    # best_run: bool | None = None,
    # date_from: str | None = None,
    # date_until: str | None = None,
):
    template = jinja_env.get_template("comparison_accuracy.html")
    return HTMLResponse(
        content=generate_comparison(
            comparison_type='accuracy',
            db=db,
            template=template,
            model_id=model_id,
            dataset_id=dataset_id,
            device_name=device_name,
            kernel_version_a=kernel_version_a,
            kernel_version_b=kernel_version_b
        )
    )