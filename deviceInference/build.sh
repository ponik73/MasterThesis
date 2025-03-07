#!/bin/bash

docker buildx create --use

# docker buildx build --platform linux/amd64,linux/arm64,linux/arm/v7 -t ponik73/device-api:latest --push .

docker buildx build --platform linux/arm64,linux/amd64 -t ponik73/device-api:latest --push .
