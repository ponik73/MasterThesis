import subprocess

import uvicorn
from evaluate.router import evaluate_router
from exceptions import FingerprintError
from fastapi import FastAPI, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from model.router import model_router
from schemas import FingerprintOutput
from starlette.requests import Request
from starlette.responses import JSONResponse

app = FastAPI(summary="REST API for latency and accuracy assessment of neural network models on embedded platforms.")

# DEBUG:
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.include_router(model_router)
app.include_router(evaluate_router)


@app.exception_handler(500)
async def internal_exception_handler(_: Request, __: Exception) -> JSONResponse:
    """Exception handler for internal server JSONResponse (500)."""
    return JSONResponse(status_code=500, content=jsonable_encoder({"error_message": "Internal Server Error"}))


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Exception handler for HTTPException (400, 404, etc.)."""
    return JSONResponse(status_code=exc.status_code, content=jsonable_encoder({"error_message": str(exc.detail)}))


@app.get(
    "/",
    description="Sever alive endpoint",
)
def alive() -> Response:
    """Return 200 OK if server is up."""
    return Response(status_code=200)


@app.get(
    "/fingerprint",
    description="Run profiling with parameters defined by request body.",
    response_model=FingerprintOutput,
)
def get_fingerprint() -> FingerprintOutput:
    """Get device name and kernel version."""
    cmd_result = subprocess.run("echo $SYS_NAME", capture_output=True, text=True, shell=True, check=False)

    if cmd_result.returncode != 0:
        error_message = f"Unable to indentify the device. Reason {cmd_result.stderr}"
        raise FingerprintError(error_message)

    sys_name = cmd_result.stdout

    cmd_result = subprocess.run("uname -r", capture_output=True, text=True, shell=True, check=False)

    if cmd_result.returncode != 0:
        error_message = f"Unable to indentify the device. Reason {cmd_result.stderr}"
        raise FingerprintError(error_message)

    kernel_version = cmd_result.stdout

    return FingerprintOutput(sys_name=sys_name, kernel_version=kernel_version)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--port", "-p", default=8000, type=int)
    # args = parser.parse_args()
    # uvicorn.run(app, host="127.0.0.1", port=args.port)

    PORT = 8000
    uvicorn.run(app, host="127.0.0.1", port=PORT)
