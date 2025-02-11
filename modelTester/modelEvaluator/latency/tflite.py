# TFLite tool `benchmark_model` is used for latency evaluation of TFLite models
import asyncio, asyncssh
from pathlib import Path
import re

async def evaluateLatencyTFLite(
            conn: asyncssh.SSHClientConnection,
            modelRemotePath: Path,
            executablePath: str
            ) -> dict:
        # latencyExecutable = "/usr/bin/tensorflow-lite-2.16.2/examples/./benchmark_model"
        # onDeviceModelPath = "/usr/bin/tensorflow-lite-2.16.2/examples/mobilenet_v1_1.0_224_quant.tflite"
        assessment = {}

        result = await conn.run(f'{executablePath} --graph={modelRemotePath.as_posix()}', timeout=160)
        if result.exit_status == 0:
            assessment = _parseLatencyTFLite(result.stdout)

        return assessment

def _parseLatencyTFLite(executableOutput: str) -> dict:
        metricRe = {
            "init": ("Init:", ","),
            "firstInference": ("First inference:", ","),
            "warmup": ("Warmup \\(avg\\):", ","),
            "inference": ("Inference \\(avg\\):", "\n")
        }

        metricValue = {}
        for metric in metricRe.keys():
            # Metric regex (identifier) strings:
            startRe, endRe = metricRe[metric]
            # Indexes of the metric identifier strings:
            startIdx, endIdx = re.search(f'{startRe}(.*?){endRe}', executableOutput).span()
            # Substring containing the metric information:
            information = executableOutput[startIdx : endIdx]
            # Index of metric start regex in substring:
            valueStart = re.search(f'{startRe}', information).end()
            # Index of metric end regex in substring:
            valueEnd = re.search(f'{endRe}', information).start()
            # Substring except regexes:
            value = information[valueStart : valueEnd]
            # Add value to the result dict:
            metricValue[metric] = value.strip()

        return metricValue