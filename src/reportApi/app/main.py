from html.environment import jinja_env
from typing import Annotated

import uvicorn
from compare.router import compare_router
from database.services import create_db_and_tables, get_session
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
from schemas import GetReportInput, RegisterRunInput, RegisterRunOutput
from services import generate_report_service, register_run_service
from sqlmodel import Session
from starlette.requests import Request
from starlette.responses import JSONResponse

app = FastAPI()
app.include_router(compare_router)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database on server startup."""
    create_db_and_tables()


################## ROUTER


@app.post("/run/", response_model=RegisterRunOutput)
def register_run(run: RegisterRunInput, db: Annotated[Session, Depends(get_session)]) -> RegisterRunOutput:
    """Write a run into database."""
    return RegisterRunOutput(
        run_id=register_run_service(
            device=run.device,
            model=run.model,
            metrics=(run.latency_metrics, run.accuracy_metrics),
            dataset=run.dataset,
            db=db,
        )
    )


@app.get("/report/", response_class=HTMLResponse)
def get_report(
    experiment: Annotated[GetReportInput, Query(..., description="List of run IDs")],
    db: Annotated[Session, Depends(get_session)],
) -> HTMLResponse:
    """Get a html report for given runs."""
    template = jinja_env.get_template("report.html")
    return HTMLResponse(
        content=generate_report_service(run_ids=list(set(experiment.run_ids)), db=db, template=template)
    )


################## EXCEPTIONS


@app.exception_handler(500)
async def internal_exception_handler(_: Request, __: Exception) -> JSONResponse:
    """Exception handler for internal server JSONResponse (500)."""
    return JSONResponse(status_code=500, content=jsonable_encoder({"error_message": "Internal Server Error"}))


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Exception handler for HTTPException (400, 404, etc.)."""
    return JSONResponse(status_code=exc.status_code, content=jsonable_encoder({"error_message": str(exc.detail)}))


################## MAIN

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--port", "-p", default=8000, type=int)
    # args = parser.parse_args()
    # uvicorn.run(app, host="127.0.0.1", port=args.port)

    PORT = 8001
    uvicorn.run(app, host="127.0.0.1", port=PORT)
