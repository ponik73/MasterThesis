from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import asyncssh
from pydantic_core import Url

if TYPE_CHECKING:
    from http.client import HTTPConnection

    from configurationHandler.schemas import Device

from configuration import DockerHubPath, get_settings
from modelEvaluator.api.device.calls import (
    evaluate_latency_tflite_call,
    get_fingerprint_call,
    get_model_input_info_tflite_call,
    is_server_up_call,
    predict_tflite_call,
    upload_model_call,
)
from modelEvaluator.exceptions import (
    DeviceApiError,
    DeviceApiInitError,
    FrameworkNotSupportedError,
    ModelUploadError,
)


class DeviceInterface:
    """Class defining interface for communication with embedded device."""

    def __init__(self, device: Device) -> None:
        """DeviceInterface constructor."""
        self.url: Url | None = None
        self.device = device
        self.conn: HTTPConnection | None = None

        self.url = Url.build(scheme="http", host=self.device.uri, port=8000)

        asyncio.get_event_loop().run_until_complete(self._initialize_api())

    async def _create_ssh_connection(self) -> asyncssh.SSHClientConnection:
        """Create SSH connection with remote device.

        :return: Estabilished ssh communication channel between PC and remote device.
        """
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
            error_message = "Could not list remote device free ports."
            raise DeviceApiInitError(error_message)

        result_std_output = result.stdout.decode() if type(result.stdout) is bytes else cast(str, result.stdout)
        # strip newline char from each record
        filtered = filter(None, result_std_output.split("\n"))
        occupied_ports = (int(port) for port in filtered)
        default_port = 8000
        while default_port in occupied_ports:
            default_port += 1

        return default_port

    async def _check_for_docker_container(self, conn: asyncssh.SSHClientConnection, image: DockerHubPath) -> tuple[str | None, int | None]:
        port: int | None = None
        container_id: str | None = None

        result = await conn.run(f'docker container ls --filter=ancestor={image} --format "{{{{.ID}}}}\n{{{{.Ports}}}}"')
        if result.exit_status != 0:
            error_message = "Unable to check if the device API container is running."
            raise DeviceApiInitError(error_message)
        lines = result.stdout.splitlines()
        if len(lines) > 1:
            port_match = re.search(r"0.0.0.0:\S+\->(?P<port>.*?)/\S+", lines[1])
            if port_match is not None:
                port = int(port_match.group("port"))

            container_id = lines[0].strip()
        
        return container_id, port

    async def _pull_and_run_docker(self, conn: asyncssh.SSHClientConnection, image: DockerHubPath) -> int:
        result = await conn.run(f"docker pull {image}")
        if result.exit_status != 0:
            error_message = "Unable to pull the device API image."
            raise DeviceApiInitError(error_message)

        port = await self._get_free_port(conn)

        # Run the API in container:
        result = await conn.run(f"docker run -e SYS_NAME=`uname -n` -d -p {port}:{port} {image}")
        if result.exit_status != 0:
            error_message = "Unable to start the device API container."
            raise DeviceApiInitError(error_message)

        return port
    
    async def _stop_docker_container(self) -> None:
        settings = get_settings()

        async with await self._create_ssh_connection() as conn:
            # Get container ID:
            container_id, _ = await self._check_for_docker_container(conn=conn, image=settings.DOCKER_DEVICE_API_PATH)

            if container_id:
                # Call docker container stop command:
                result = await conn.run(f'docker container stop {container_id}')
                if result.exit_status != 0:
                    error_message = "Unable to stop the device API container."
                    raise DeviceApiError(error_message)


    async def _initialize_api(self) -> None:
        settings = get_settings()
        port: int | None = None

        # self.url = Url.build(scheme="http", host=self.device.uri, port=8000)

        # return

        async with await self._create_ssh_connection() as conn:
            _, port = await self._check_for_docker_container(conn=conn, image=settings.DOCKER_DEVICE_API_PATH)
            if port is None:
                port = await self._pull_and_run_docker(conn=conn, image=settings.DOCKER_DEVICE_API_PATH)

            # Store the port where container is running:
            self.url = Url.build(scheme="http", host=self.device.uri, port=port)

            # Check if server is up:
            start_time = time.time()
            timeout = 60
            server_up = False

            while (time.time() - start_time < timeout) and (not server_up):
                asyncio.sleep(1)
                server_up = is_server_up_call(url=self.url)

            if not server_up:
                error_message = "Device API container is not running."
                raise DeviceApiInitError(error_message)

    def get_fingerprint(self) -> None:
        """Get device fingerprint."""
        call_args = {"headers": {"accept": "application/json"}}

        sys_name, kernel_version = get_fingerprint_call(url=self.url, **call_args)

        self.device.sys_name = sys_name.strip()
        self.device.kernel_version = kernel_version.strip()

    def upload_model(self, model_path: Path, model_custom_name: str) -> bool:
        """Upload model to device."""
        try:
            with Path(model_path).open("rb") as model_file:
                call_args = {
                    "headers": {
                        "accept": "application/json",
                        # requests won't add a boundary if this header is set when you pass files=
                        # 'Content-Type': 'multipart/form-data',
                    },
                    "files": {
                        "custom_name": (None, model_custom_name),
                        "model_file": model_file,
                    },
                }

                upload_model_call(url=self.url, **call_args)
        except ModelUploadError:
            print(f"Unable to upload model `{model_custom_name}` to device: `{self.name}` at `{self.url}`.")
            return False
        else:
            return True

    def get_model_input_info(self, model_custom_name: str, model_framework: Literal["tflite"]) -> list[dict]:
        """Get the model's input node info."""
        call_args = {
            "headers": {
                "accept": "application/json",
            },
            "params": {"model_custom_name": model_custom_name},
        }

        if model_framework == "tflite":
            return get_model_input_info_tflite_call(url=self.url, **call_args)
        raise FrameworkNotSupportedError(model_framework)

    def evaluate_latency(self, model_custom_name: str, model_framework: Literal["tflite"]) -> str | None:
        """Evaluate latency ondevice."""
        call_args = {
            "headers": {
                "accept": "application/json",
                # requests won't add a boundary if this header is set when you pass files=
                # 'Content-Type': 'multipart/form-data',
            },
            "files": {"model_custom_name": (None, model_custom_name)},
        }

        if model_framework == "tflite":
            return evaluate_latency_tflite_call(url=self.url, **call_args)
        raise FrameworkNotSupportedError(model_framework)

    def predict(self, model_custom_name: str, model_framework: Literal["tflite"], encoded_batch: bytes) -> bytes | None:
        """Execute ondevice model inference for given batch."""
        call_args = {
            "headers": {
                "accept": "application/json",
                # requests won't add a boundary if this header is set when you pass files=
                # 'Content-Type': 'multipart/form-data',
            },
            "files": {
                "batch": encoded_batch,
                "model_custom_name": (None, model_custom_name),
            },
        }

        if model_framework == "tflite":
            return predict_tflite_call(url=self.url, **call_args)
        raise FrameworkNotSupportedError(model_framework)
    
    def shutdownApi(self) -> None:
        """Shutdown Device API (intended to use at the end of the evaluation pipeline)."""
        asyncio.get_event_loop().run_until_complete(self._stop_docker_container())