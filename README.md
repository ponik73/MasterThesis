# MasterThesis

`mypy --strict --ignore-missing-imports .`

`pre-commit run --all-files`

## Device API

either:
- docker build (recommended)
- install dependencies and run uvicorn (machine must support tflite runtime)

## Report API

either:
- install dependencies and run uvicorn
- docker build

## Model Tester

install dependencies; setup config (urls); create configuration files

In `src/modelTester` folder:
```
python modelTester.py --cfg=<PATH>
```