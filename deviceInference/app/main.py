import subprocess
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from starlette.responses import JSONResponse
from starlette.requests import Request
from fastapi.encoders import jsonable_encoder
from schemas import FingerprintOutput

from model.router import modelRouter
from evaluate.router import evaluateRouter
from exceptions import FingerprintError

app = FastAPI(
    summary="REST API for latency and accuracy assessment of neural network models on embedded platforms."
)

# DEBUG:
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.include_router(modelRouter)
app.include_router(evaluateRouter)

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
def alive():
    return Response(status_code=200)

@app.get(
        "/fingerprint",
        description="Run profiling with parameters defined by request body.",
        response_model=FingerprintOutput
)
async def getFingerprint():
    cmdResult = subprocess.run('echo $SYS_NAME', capture_output=True, text=True, shell = True)

    if cmdResult.returncode != 0:
        raise FingerprintError(f"Unable to indentify the device. Reason {cmdResult.stderr}")
    
    sys_name = cmdResult.stdout
    
    cmdResult = subprocess.run('uname -r', capture_output=True, text=True, shell = True)

    if cmdResult.returncode != 0:
        raise FingerprintError(f"Unable to indentify the device. Reason {cmdResult.stderr}")
    
    kernel_version = cmdResult.stdout

    return FingerprintOutput(sys_name=sys_name, kernel_version=kernel_version)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--port", "-p", default=8000, type=int)
    # args = parser.parse_args()
    # uvicorn.run(app, host="127.0.0.1", port=args.port)

    PORT = 8000
    uvicorn.run(app, host="127.0.0.1", port=PORT)
    