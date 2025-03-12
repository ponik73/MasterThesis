import uvicorn
from starlette.responses import JSONResponse
from starlette.requests import Request
from typing import Annotated
from sqlmodel import Session, SQLModel, create_engine
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader

from schemas import RegisterRunInput, RegisterRunOutput, GetReportInput
from services import register_run_service, generate_report_service

app = FastAPI(
    # summary="REST API for latency and accuracy assessment of neural network models on embedded platforms."
)

################## DATABASE

engine = create_engine(
    "sqlite:///database.db",
    connect_args={
        "check_same_thread": False
    }
)
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
def get_session():
    with Session(engine) as session:
        yield session

################## HTML

jinja_env = Environment(loader=FileSystemLoader("html/templates"))

################## LIFESPAN TODO       

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

################## ROUTER

@app.post("/run/", response_model=RegisterRunOutput)
def register_run(
    run: RegisterRunInput,
    db: Annotated[Session, Depends(get_session)]
):
    return RegisterRunOutput(
        run_id=register_run_service(
            device=run.device,
            model=run.model,
            latency_metrics=run.latency_metrics,
            dataset=run.dataset,
            accuracy_metrics=run.accuracy_metrics,
            db=db
        )
    )

@app.get("/report/", response_class=HTMLResponse)
def generate_report(
    experiment: Annotated[GetReportInput, Query(..., description="List of run IDs")],
    db: Annotated[Session, Depends(get_session)],
):
    template = jinja_env.get_template("report.html")
    return HTMLResponse(
        content=generate_report_service(
            run_ids=list(set(experiment.run_ids)),
            db=db,
            template=template
        )
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
    