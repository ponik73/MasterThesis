#!/bin/bash

docker build --no-cache -t ponik73/device-api .

docker push ponik73/device-api
