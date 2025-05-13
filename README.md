# MasterThesis

Report API and Model Tester steps required. Device API only if you want to check it out.

## Report API

### Docker (recommended)

In `src/reportApi` directory:

1. Build the Docker image
```
docker build --no-cache -t report-api .
```

2. Run the image in container
```
docker run -d -p 8001:8001 report-api
```

### Python

In `src/reportApi/app` directory:

1. Install dependencies
```
pip install -r ../requirements.txt
```

2. Initialize the API
```
python -m uvicorn main:app --port 8001
```

## Model Tester

In `src/modelTester` directory:

1. Install dependencies
```
pip install -r requirements.txt
```

2. Run the orchestrator script:
```
python modelTester.py --cfg=<PATH>
```

The repository contains `test/experiment.json` file which was used during validation experiment.
To use this file, modify the `<URI>` placeholder to the IP address of the embedded device.
```
"devices": [
        {
            "uri": "localhost",
            "name": "<URI>"
        }
    ],
```
Run the orchestrator script with `test/experiment.json` file. In the `src/modelTester` directory:
```
python modelTester.py --cfg=../../test/experiment.json
```

## Device API (ak si to chces pustit lokalne)

You don't need to initialize the API, this tutorial is only useful if you want to use Device API as is (the orchestrator script works only with embedded devices).

Initialize the API either via Docker or Python. Once the API is initialized open `localhost:8000/docs` in the browser or sent REST requests via some client.

### Docker (recommended)

Either build the image locally or use the uploaded image on docker hub.

#### Building the image

In `src/deviceApi` directory:

1. Build the Docker image
```
docker build --no-cache device-api:latest .
```

2. Run the image in container
```
docker run -e SYS_NAME=`uname -n` -d -p 8000:8000 device-api:latest
```

#### Accessing the image from Docker Hub

1. Pull the image
```
docker pull ponik73/device-api
```

2. Run the image in container
```
docker run -e SYS_NAME=`uname -n` -d -p 8000:8000 ponik73/device-api
```

### Python

Machine must support `tflite-runtime` package! In `src/deviceApi/app` directory:

1. Install dependencies
```
pip install -r ../requirements.txt
```

2. Initialize the API
```
python -m uvicorn main:app --port 8000
```

