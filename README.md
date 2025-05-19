# MasterThesis

This repository contains modules developed for the master’s thesis. The recommended approach for launching services is via Docker. Python-based initialization is also available.

> For the validation experiment: Initialize the Report API, update the provided configuration file (`test/experiment.json`) with the IP address of the embedded device(s), and run the Model Tester script using this configuration.

## Report API

### Docker (recommended)

Navigate to the `src/reportApi` directory and follow these steps:

1. Build the Docker image:
```
docker build --no-cache -t report-api .
```

2. Run the image in container:
```
docker run -d -p 8001:8001 report-api
```

### Python

Navigate to the `src/reportApi/app` directory and execute the following:

1. Install the required dependencies:
```
pip install -r ../requirements.txt
```

2. Start the API server:
```
python -m uvicorn main:app --port 8001
```

## Model Tester

Navigate to the `src/modelTester` directory:

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Run the orchestrator script:
```
python modelTester.py --cfg=<PATH>
```

An example configuration file is provided in `test/experiment.json`, which was used during the validation experiment.
To use this file, replace the `<URI>` placeholder with the IP address of the embedded device:
```
"devices": [
        {
            "uri": "<URI>",
            "name": "device"
        }
    ],
```
To run the script with this configuration, execute the following from the `src/modelTester` directory:
```
python modelTester.py --cfg=../../test/experiment.json
```

## Device API

> **Note**: Initialization of the Device API is not required unless you wish to explore or test it locally. The orchestrator script is designed to work exclusively with embedded devices.

Once the Device API is running, access the interactive documentation via http://localhost:8000/docs or send REST requests using a client of your choice.

### Docker (recommended)

You may either build the Docker image locally or pull it from Docker Hub.

#### Building the image

In `src/deviceApi` directory:

1. Build the Docker image:
```
docker build --no-cache device-api:latest .
```

2. Run the container:
```
docker run -e SYS_NAME=`uname -n` -d -p 8000:8000 device-api:latest
```

#### Use pre-built image from Docker Hub

1. Pull the image:
```
docker pull ponik73/device-api
```

2. Run the container:
```
docker run -e SYS_NAME=`uname -n` -d -p 8000:8000 ponik73/device-api
```

### Python

> **Note**: The machine must support the `tflite-runtime` package.

Navigate to the `src/deviceApi/app` directory:

1. Install the required dependencies:
```
pip install -r ../requirements.txt
```

2. Start the API server:
```
python -m uvicorn main:app --port 8000
```

