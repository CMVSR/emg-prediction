#!/bin/bash

# to run multiple containers on different gpus
for i in {0..3}
do
    docker run \
        --gpus device=$i \
        --volume $(pwd):/home/emg-prediction \
        --workdir /home/emg-prediction \
        --user $(id -u):$(id -g) \
        -d -it \
        --detach-keys="a" \
        emg-prediction python3 /home/emg-prediction/app/app.py
done
