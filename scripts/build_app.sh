#!/bin/bash

cp ./requirements.txt ./deployment/app/requirements.txt
docker build --no-cache --force-rm -t emg-prediction-app ./deployment/app/
rm ./deployment/app/requirements.txt
