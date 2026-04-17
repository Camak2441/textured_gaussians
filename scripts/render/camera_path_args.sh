#!/usr/bin/bash

CAMERA_PATHS=$(echo $1 | tr "," "\n")
CAMERA_PATHS_ARG=""

for CAMERA_PATH in $CAMERA_PATHS
do
    if [ -n "$CAMERA_PATHS_ARG" ]; then
        CAMERA_PATHS_ARG="$CAMERA_PATHS_ARG ../examples/results/camera_paths/$CAMERA_PATH.json"
    else
        CAMERA_PATHS_ARG="../examples/results/camera_paths/$CAMERA_PATH.json"
    fi
done

echo $CAMERA_PATHS_ARG
