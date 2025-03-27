import asyncio
import re
import time
from http.client import HTTPConnection
from pathlib import Path
from typing import Literal, cast

import asyncssh
from configuration import DockerHubPath, get_settings
from configurationHandler.schemas import Device
from pydantic_core import Url

from modelEvaluator.api.device.calls import (
    evaluateLatencyTfliteCall,
    getFingerprintCall,
    getModelInputInfoTfliteCall,
    isServerUpCall,
    predictTfliteCall,
    uploadModelCall,
)
from modelEvaluator.exceptions import DeviceApiInitError


class DeviceInterface:
    def __init__(self, device: Device):
        self.url: Url | None = None
        self.device = device
        self.conn: HTTPConnection | None = None

        self.url = Url.build(scheme="http", host=self.device.uri, port=8000)

        asyncio.get_event_loop().run_until_complete(self._initializeApi())

    async def _createSshConnection(self) -> asyncssh.SSHClientConnection:
        """Create SSH connection with remote device.

        :return: Estabilished ssh communication channel between PC and remote device.
        """
        # try:
        return await asyncssh.connect(
            self.device.uri,
            username=self.device.credentials.user_name,
            password=self.device.credentials.password,
            connect_timeout=30,
            known_hosts=None,
        )

    async def _get_free_port(self, conn: asyncssh.SSHClientConnection) -> int:
        """Obtain free network port for new modelrunner process.

        Query remote device for list of the occupied ports which are not available for new modelrunner instance.

        :param conn: Opened SSH client connection.
        :return: Available port for new modelrunner instance.
        """
        result = await conn.run(
            "netstat -tlnp | tail -n +3 | grep \"^tcp \"  | awk '{print $4}' | cut -d : -f 2", check=True, timeout=3
        )

        if not result.stdout:
            raise DeviceApiInitError("Could not list remote device free ports.")

        result_std_output = result.stdout.decode() if type(result.stdout) is bytes else cast(str, result.stdout)
        # strip newline char from each record
        filtered = filter(None, result_std_output.split("\n"))
        occupied_ports = (int(port) for port in filtered)
        default_port = 8000
        while default_port in occupied_ports:
            default_port += 1

        return default_port

    async def _check_for_docker_container(self, conn: asyncssh.SSHClientConnection, image: DockerHubPath) -> int | None:
        port: int | None = None

        result = await conn.run(f'docker container ls --filter=ancestor={image} --format "{{{{.ID}}}}\n{{{{.Ports}}}}"')
        if result.exit_status != 0:
            raise DeviceApiInitError("Unable to check if the device API container is running.")
        lines = result.stdout.splitlines()
        if len(lines) > 1:
            container_id = lines[0]

            port_match = re.search(r"0.0.0.0:\S+\->(?P<port>.*?)/\S+", lines[1])
            if port_match is not None:
                port = int(port_match.group("port"))

        return port

    async def _pull_and_run_docker(self, conn: asyncssh.SSHClientConnection, image: DockerHubPath) -> int:
        result = await conn.run(f"docker pull {image}")
        if result.exit_status != 0:
            raise DeviceApiInitError("Unable to pull the device API image.")

        port = await self._get_free_port(conn)

        # Run the API in container:
        result = await conn.run(f"docker run -e SYS_NAME=`uname -n` -d -p {port}:{port} {image}")
        if result.exit_status != 0:
            raise DeviceApiInitError("Unable to start the device API container.")

        return port

    async def _initializeApi(self):
        settings = get_settings()
        port: int | None = None

        async with await self._createSshConnection() as conn:
            port = await self._check_for_docker_container(conn=conn, image=settings.DOCKER_DEVICE_API_PATH)
            if port is None:
                port = await self._pull_and_run_docker(conn=conn, image=settings.DOCKER_DEVICE_API_PATH)

            # Store the port where container is running:
            self.url = Url.build(scheme="http", host=self.device.uri, port=port)

            # Check if server is up:
            start_time = time.time()
            timeout = 60
            server_up = False

            while (time.time() - start_time < timeout) and (not server_up):
                time.sleep(1)
                server_up = isServerUpCall(url=self.url)

            if not server_up:
                raise DeviceApiInitError("Device API container is not running.")

    def get_fingerprint(self):
        try:
            callArgs = {"headers": {"accept": "application/json"}}

            sys_name, kernel_version = getFingerprintCall(url=self.url, **callArgs)

            if sys_name is None or kernel_version is None:
                raise

            self.device.sys_name = sys_name.strip()
            self.device.kernel_version = kernel_version.strip()
        except Exception as e:
            print(f"Unable to retrieve fingerprint of device: `{self.device.name}` at `{self.url}`. Reason: {e}")

    def upload_model(self, model_path: Path, model_custom_name: str) -> bool:
        try:
            callArgs = {
                "headers": {
                    "accept": "application/json",
                    # requests won't add a boundary if this header is set when you pass files=
                    # 'Content-Type': 'multipart/form-data',
                },
                "files": {
                    "customName": (None, model_custom_name),
                    "modelFile": open(model_path, "rb"),
                },
            }

            uploadModelCall(url=self.url, **callArgs)
            return True
        except Exception as e:
            print(f"Unable to upload model `{model_custom_name}` to device: `{self.name}` at `{self.url}`. Reason: {e}")
            return False

    def getModelInputInfo(self, model_custom_name: str, modelFramework: Literal["tflite"]) -> list[dict]:
        try:
            callArgs = {
                "headers": {
                    "accept": "application/json",
                },
                "params": {"model_custom_name": model_custom_name},
            }

            if modelFramework == "tflite":
                return getModelInputInfoTfliteCall(url=self.url, **callArgs)
            raise Exception(f"{modelFramework} framework not supported.")
        except Exception as e:
            print(f"Unable to retrieve model input details `{model_custom_name}`. Reason: {e}")
            return []

    def evaluateLatency(self, model_custom_name: str, modelFramework: Literal["tflite"]) -> str | None:
        callArgs = {
            "headers": {
                "accept": "application/json",
                # requests won't add a boundary if this header is set when you pass files=
                # 'Content-Type': 'multipart/form-data',
            },
            "files": {"model_custom_name": (None, model_custom_name)},
        }

        if modelFramework == "tflite":
            return evaluateLatencyTfliteCall(url=self.url, **callArgs)
        raise Exception(f"{modelFramework} framework not supported.")

    def predict(self, model_custom_name: str, modelFramework: Literal["tflite"], encoded_batch: bytes) -> bytes | None:
        try:
            callArgs = {
                "headers": {
                    "accept": "application/json",
                    # requests won't add a boundary if this header is set when you pass files=
                    # 'Content-Type': 'multipart/form-data',
                },
                "files": {
                    "batch": encoded_batch,  # open('bla.npy', 'rb'),
                    "model_custom_name": (None, model_custom_name),
                },
            }

            if modelFramework == "tflite":
                return predictTfliteCall(url=self.url, **callArgs)
            raise Exception(f"{modelFramework} framework not supported.")

        except Exception as e:
            print(
                f"Unable to evaluate accuracy for model `{model_custom_name}` on device: `{self.device.name}` at `{self.url}`. Reason: {e}"
            )
            return None
