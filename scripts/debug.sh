#!/bin/bash
docker run \
    --gpus device=2 \
    --volume $(pwd):/home/emg-prediction \
    --workdir /home/emg-prediction \
    -p 127.0.0.1:5678:5678 \
    --rm \
    emg-prediction
