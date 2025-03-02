import asyncio, asyncssh
import requests
from http.client import HTTPConnection
from pathlib import Path
from pydantic_core import Url
from typing import Literal, List, Dict

from configuration import getSettings
from .apiCalls import getFingerprintCall, uploadModelCall, getModelInputInfoTfliteCall, evaluateLatencyTfliteCall, evaluateAccuracyTfliteCall

class DeviceInterface():
    def __init__(self, url: Url, name: str):
        self.url = url
        self.name = name
        self.conn : HTTPConnection | None = None
        
        # try:
        #     # TODO: connect to the device: download, build, run docker fastapi
        #     asyncio.get_event_loop().run_until_complete(self._initializeApi())
        # except Exception as e:
        #     raise Exception("TODO Message (maybe exceptions.py)")

    async def _initializeApi(self):
        # docker pull your-dockerhub-username/fastapi-tf:latest
        # docker run -d -p 8000:8000 your-dockerhub-username/fastapi-tf:latest

        settings = getSettings()
        port : int | None = None

        async def _createSshConnection(url) -> asyncssh.SSHClientConnection:
            """Create SSH connection with remote device.

            :return: Estabilished ssh communication channel between PC and remote device.
            """
            # try:
            return await asyncssh.connect(
                url,
                username="root",#self.username,
                password="", #self.password,
                connect_timeout=3,
                known_hosts=None
            )
            # except asyncssh.TimeoutError as e:
            #     raise Exception(e)

        async with await _createSshConnection(str(self.url)) as conn:
            # Pull device API docker image:
            result = await conn.run(f'docker pull {settings.DOCKER_DEVICE_API_PATH}')
            if result.exit_status != 0:
                raise Exception("Unable to pull the device API image.")
            
            port = 8000 # TODO: get free port

            # Run the API in container:
            result = await conn.run(f'docker run -d -p {port}:{port} {settings.DOCKER_DEVICE_API_PATH}')
            if result.exit_status != 0:
                raise Exception("Unable to pull the device API image.")
            
            # Store the port where container is running:
            self.url = Url.build(
                scheme=self.url.scheme,
                host=self.url.host,
                port=port
            )
        
    def getFingerprint(self) -> str | None:
        try:
            callArgs = {
                "headers": {
                    'accept': 'application/json'
                }
            }
            
            fingerprint = getFingerprintCall(
                url=self.url,
                **callArgs
            )
            if not fingerprint:
                raise
            
            return fingerprint
        except Exception as e:
            print(f'Unable to retrieve fingerprint of device: `{self.name}` at `{self.url}`. Reason: {e}')
            return None

    def uploadModel(
            self,
            modelPath: Path,
            modelCustomName: str
    ) -> bool:
        try:
            callArgs = {
                "headers": {
                    'accept': 'application/json',
                    # requests won't add a boundary if this header is set when you pass files=
                    # 'Content-Type': 'multipart/form-data',
                },
                "files": {
                    'customName': (None, modelCustomName),
                    'modelFile': open(modelPath, 'rb'),
                }
            }

            uploadModelCall(
                url=self.url,
                **callArgs
            )
            return True
        except Exception as e:
            print(f'Unable to upload model `{modelCustomName}` to device: `{self.name}` at `{self.url}`. Reason: {e}')
            return False
        
    def getModelInputInfo(
            self,
            modelCustomName: str,
            modelFramework: Literal["tflite"]
    ) -> List[Dict]:
        try:
            callArgs = {
                "headers": {
                    'accept': 'application/json',
                },
                "params": {
                    "modelCustomName": modelCustomName
                }
            }

            if modelFramework == "tflite":
                return getModelInputInfoTfliteCall(
                    url=self.url,
                    **callArgs
                )
            else:
                raise Exception(f'{modelFramework} framework not supported.')
        except Exception as e:
            print(f'Unable to retrieve model input details `{modelCustomName}`. Reason: {e}')
            return []

    def evaluateLatency(
            self,
            modelCustomName: str,
            modelFramework: Literal["tflite"]
    ) -> str | None:
        callArgs = {
            "headers" : {
                'accept': 'application/json',
                # requests won't add a boundary if this header is set when you pass files=
                # 'Content-Type': 'multipart/form-data',
            },
            "files" : {
                'modelCustomName': (None, modelCustomName)
            }
        }

        if modelFramework == "tflite":
            return evaluateLatencyTfliteCall(
                url=self.url,
                **callArgs
            )
        else:
            raise Exception(f'{modelFramework} framework not supported.')
    
    def evaluateAccuracy(
            self,
            modelCustomName: str,
            modelFramework: Literal["tflite"],
            encodedBatch: bytes
    ) -> bytes | None:
        try:
            callArgs = {
                "headers" : {
                    'accept': 'application/json',
                    # requests won't add a boundary if this header is set when you pass files=
                    # 'Content-Type': 'multipart/form-data',
                },
                "files" : {
                    'batch': encodedBatch,#open('bla.npy', 'rb'),
                    'modelCustomName': (None, modelCustomName),
                }
            }

            if modelFramework == "tflite":
                return evaluateAccuracyTfliteCall(
                    url=self.url,
                    **callArgs
                )
            else:
                raise Exception(f'{modelFramework} framework not supported.')

        except Exception as e:
            print(f'Unable to evaluate accuracy for model `{modelCustomName}` on device: `{self.name}` at `{self.url}`. Reason: {e}')
            return None