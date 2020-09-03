#!/bin/bash

echo "creating directories..."
mkdir ./dataset/extracted

echo "extracting features..."
docker run \
    --gpus device=0 \
    --volume $(pwd):/home/emg-prediction \
    --workdir /home/emg-prediction \
    --user $(id -u):$(id -g) \
    --detach-keys="a" \
    emg-prediction python3 /home/emg-prediction/dataset/feature_extraction.py

