# from ...configurationHandler.configurations import Device
from configurationHandler.configurations import Device
import asyncio, asyncssh
from pathlib import Path

import modelEvaluator.latency.tflite
# 10.171.65.19

class DeviceController():
    def __init__(self,
                 device: Device
        # name: str,
        # uri:str,
        # username: str = "root",
        # password: str = "",
        # fingerprint: str | None = None
        ):
        # self.name = name
        # self.uri = uri
        # self.username = username
        # self.password = password
        # self.fingerprint = fingerprint
        self.device = device

    async def getFingerprint(self) -> bool:
        async with await self._createSshConnection() as conn:
            result = await conn.run("uname -r;exit", timeout=2)
            if result.exit_status == 0:
                self.fingerprint = result.stdout
        return self.fingerprint is not None
    
    async def evaluateLatency(self, modelRemotePath: Path) -> str | None:
        
        evalFunc, evalFuncArgs = None, None
        if str(modelRemotePath).endswith(".tflite"):
            evalFunc = modelEvaluator.latency.tflite.evaluateLatencyTFLite
            evalFuncArgs = {
                "executablePath": "/usr/bin/tensorflow-lite-2.16.2/examples/./benchmark_model" #TODO: get information from Device
            }
        else:
            raise NotImplementedError
        
        async with await self._createSshConnection() as conn:
            result = await evalFunc(conn, modelRemotePath, **evalFuncArgs)
        
        return result

    
    async def _createSshConnection(self) -> asyncssh.SSHClientConnection:
        """Create SSH connection with remote device.

        :return: Estabilished ssh communication channel between PC and remote device.
        """
        try:
            return await asyncssh.connect(
                self.device.uri,
                username="root",#self.username,
                password="", #self.password,
                connect_timeout=3,
                known_hosts=None
            )
        except asyncssh.TimeoutError:
            message = f"Unable to reach device {self.device.name}"
            raise Exception(message)

    async def uploadModel(self, modelLocalPath: Path):
        remoteDir = "$HOME" # TODO: device.modelDir
        
        modelRemotePath = Path(remoteDir) / modelLocalPath.name

        async with await self._createSshConnection() as conn:
            await asyncssh.scp(modelLocalPath, (conn, modelRemotePath.as_posix()))

        return modelRemotePath
    
    # async def removeModel(self, modelRemotePath: Path) -> bool:
    #     removed = False
    #     async with await self._createSshConnection() as conn:
    #         result = await conn.run(f'rm {modelRemotePath}')
    #         if result.exit_status == 0:
    #             removed = True
    #     return removed
    
