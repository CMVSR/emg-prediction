#!/bin/bash

# to run on one gpu
docker run \
    --gpus device=0 \
    --volume $(pwd):/home/emg-prediction \
    --workdir /home/emg-prediction \
    --user $(id -u):$(id -g) \
    -d -it \
    --detach-keys="a" \
    emg-prediction
