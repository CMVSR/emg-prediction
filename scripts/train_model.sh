#!/bin/bash

echo "training extraction model..."
docker run \
    --gpus device=0 \
    --volume $(pwd):/home/emg-prediction \
    --workdir /home/emg-prediction \
    --user $(id -u):$(id -g) \
    --detach-keys="a" \
    emg-prediction-extraction-model
