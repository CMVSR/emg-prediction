#!/bin/bash

cp ./requirements.txt ./deployment/debug/requirements.txt
docker build --no-cache --force-rm -t emg-prediction-debug ./deployment/debug/
rm ./deployment/debug/requirements.txt
