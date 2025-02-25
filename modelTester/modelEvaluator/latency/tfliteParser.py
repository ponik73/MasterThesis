from pathlib import Path
import re

# TODO: tflite latency parser

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