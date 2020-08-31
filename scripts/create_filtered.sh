#!/bin/bash

echo "creating directories..."
mkdir ./dataset/filtered
mkdir ./dataset/filtered/move_up
mkdir ./dataset/filtered/move_down
mkdir ./dataset/filtered/no_movement

echo "resampling and filtering data..."
python3 ./dataset/resample_data.py

echo "combining patient data..."
python3 ./dataset/build_filtered.py
